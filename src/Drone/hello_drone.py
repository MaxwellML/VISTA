import asyncio
import csv
from pathlib import Path

import numpy as np

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import SEGMENTATION_PALLETE
from projectairsim.types import ImageType
from projectairsim.utils import projectairsim_log, unpack_image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLIGHTPATH_CSV = Path(__file__).resolve().parent / "flightpath.csv"

# Unreal reads this file and displays it using the DroneTelemetryHUD actor.
UNREAL_PROJECT_DIR = Path(
    r"C:\Users\cmdp7\OneDrive\Documents\Unreal Projects\RiveleroCaseStudy"
)
TELEMETRY_FILE = UNREAL_PROJECT_DIR / "Saved" / "drone_telemetry.txt"

SCENE_CONFIG = "scene_basic_drone.jsonc"
DRONE_NAME = "Drone1"
SEGMENTATION_CAMERA = "DownCamera"

FLIGHT_ALTITUDE_M = 30.0
FLIGHT_VELOCITY_MPS = 17.0
TELEMETRY_INTERVAL_SEC = 0.25

# Precision settings for waypoint arrival.
POSITION_TOLERANCE_M = 0.20
FINAL_APPROACH_VELOCITY_MPS = 1.0
MAX_POSITION_CORRECTIONS = 5

# After Project AirSim reports that a move has completed, keep watching the
# ground-truth position for a short period instead of immediately consuming
# another correction attempt.
POSITION_CHECK_INTERVAL_SEC = 0.10
SETTLE_TIMEOUT_SEC = 3.0
REQUIRED_GOOD_SAMPLES = 3


# ---------------------------------------------------------------------------
# Load plant targets
# ---------------------------------------------------------------------------

def load_flightpath():
    """Load plant IDs and NED X/Y targets from flightpath.csv."""
    targets = []

    with FLIGHTPATH_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"id", "x_m", "y_m"}
        if reader.fieldnames is None:
            raise RuntimeError("flightpath.csv has no header row.")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "flightpath.csv is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            plant_id = row["id"].strip()

            # Example:
            # Othonna_017 -> segmentation ID 17
            try:
                segmentation_id = int(plant_id.rsplit("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not derive a segmentation ID from plant ID "
                    f"{plant_id!r}. Expected a name such as Othonna_017."
                ) from exc

            if not 1 <= segmentation_id <= 255:
                raise RuntimeError(
                    f"{plant_id!r} produced segmentation ID "
                    f"{segmentation_id}, but stencil IDs must be 1-255 "
                    f"for this script."
                )

            targets.append({
                "id": plant_id,
                "segmentation_id": segmentation_id,
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                # Project AirSim uses NED coordinates:
                # negative Z means above the starting point.
                "z": -FLIGHT_ALTITUDE_M,
            })

    return targets


# ---------------------------------------------------------------------------
# Segmentation setup
# ---------------------------------------------------------------------------

def configure_plant_segmentation(world, targets):
    """
    Give the rest of the scene segmentation ID 0, then give each plant
    the numeric ID derived from its CSV name.

    Example:
        Othonna_001 -> 1
        Othonna_017 -> 17
        Othonna_030 -> 30
    """

    projectairsim_log().info(
        "Resetting scene segmentation IDs to 0"
    )

    # This is the same all-object regex pattern used by Project AirSim's
    # official segmentation example.
    world.set_segmentation_id_by_name(
        r"[\w]*",
        0,
        True,   # regex
        True,   # match owner/actor names
    )

    missing_plants = []

    for target in targets:
        found = world.set_segmentation_id_by_name(
            target["id"],
            target["segmentation_id"],
            False,  # exact name, not regex
            True,   # match the owning Unreal actor's name
        )

        projectairsim_log().info(
            "Segmentation: %s -> ID %d | found=%s",
            target["id"],
            target["segmentation_id"],
            found,
        )

        if not found:
            missing_plants.append(target["id"])

    if missing_plants:
        raise RuntimeError(
            "Project AirSim could not find these Unreal plant actors by "
            "their CSV IDs: "
            + ", ".join(missing_plants)
        )


def get_visible_plant_pixels(drone, target):
    """
    Return the number of currently visible DownCamera pixels belonging
    to the target plant.
    """

    responses = drone.get_images(
        SEGMENTATION_CAMERA,
        [ImageType.SEGMENTATION],
    )

    response = responses[ImageType.SEGMENTATION]
    segmentation_image = unpack_image(response)

    # The Project AirSim segmentation palette is RGB.
    plant_colour = np.asarray(
        SEGMENTATION_PALLETE[target["segmentation_id"]],
        dtype=np.uint8,
    )

    # Use the first three channels in case the returned image happens to
    # contain an alpha channel.
    segmentation_rgb = segmentation_image[:, :, :3]

    plant_mask = np.all(
        segmentation_rgb == plant_colour,
        axis=2,
    )

    return int(np.count_nonzero(plant_mask))


# ---------------------------------------------------------------------------
# Live telemetry
# ---------------------------------------------------------------------------

async def display_telemetry(drone, target):
    """
    Continuously show:
      - current target plant ID
      - drone coordinates
      - visible pixels of the current target plant

    The message is:
      1. printed in the Python terminal; and
      2. written to Saved/drone_telemetry.txt for Unreal to display.
    """

    while True:
        pose = drone.get_ground_truth_pose()
        position = pose["translation"]

        x = float(position["x"])
        y = float(position["y"])
        z = float(position["z"])

        visible_plant_pixels = get_visible_plant_pixels(
            drone,
            target,
        )

        message = (
            f"Target: {target['id']}\n"
            f"X: {x:.2f} m\n"
            f"Y: {y:.2f} m\n"
            f"Z: {z:.2f} m\n"
            f"Visible pixels: {visible_plant_pixels}"
        )

        # Update one line in the Python terminal.
        print(
            message.replace("\n", " | ").ljust(140),
            end="\r",
            flush=True,
        )

        # Unreal's DroneTelemetryHUD actor reads this file.
        TELEMETRY_FILE.write_text(
            message,
            encoding="utf-8",
        )

        # Refresh at the configured telemetry interval.
        await asyncio.sleep(TELEMETRY_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Flight helpers
# ---------------------------------------------------------------------------

async def move_to_target_precisely(drone, target):
    """
    Fly to a target, verify the ground-truth position, and readjust if needed.

    The first approach uses FLIGHT_VELOCITY_MPS. Subsequent corrections use
    FINAL_APPROACH_VELOCITY_MPS.

    After each move command completes, watch the real ground-truth position for
    up to SETTLE_TIMEOUT_SEC. The waypoint is accepted only after
    REQUIRED_GOOD_SAMPLES consecutive readings are within tolerance.
    """
    loop = asyncio.get_running_loop()

    last_x = None
    last_y = None
    last_z = None
    last_horizontal_error = None
    last_vertical_error = None

    for attempt in range(MAX_POSITION_CORRECTIONS + 1):
        velocity = (
            FLIGHT_VELOCITY_MPS
            if attempt == 0
            else FINAL_APPROACH_VELOCITY_MPS
        )

        projectairsim_log().info(
            "Positioning %s: attempt %d/%d at %.1f m/s | "
            "target NED=(%.7f, %.7f, %.7f)",
            target["id"],
            attempt + 1,
            MAX_POSITION_CORRECTIONS + 1,
            velocity,
            target["x"],
            target["y"],
            target["z"],
        )

        move_task = await drone.move_to_position_async(
            north=target["x"],
            east=target["y"],
            down=target["z"],
            velocity=velocity,
        )
        await move_task

        projectairsim_log().info(
            "Controller reports move complete for %s; "
            "watching position for up to %.1f s",
            target["id"],
            SETTLE_TIMEOUT_SEC,
        )

        good_samples = 0
        settle_deadline = loop.time() + SETTLE_TIMEOUT_SEC

        while loop.time() < settle_deadline:
            pose = drone.get_ground_truth_pose()
            position = pose["translation"]

            current_x = float(position["x"])
            current_y = float(position["y"])
            current_z = float(position["z"])

            dx = target["x"] - current_x
            dy = target["y"] - current_y
            dz = target["z"] - current_z

            horizontal_error = (dx**2 + dy**2) ** 0.5
            vertical_error = abs(dz)

            last_x = current_x
            last_y = current_y
            last_z = current_z
            last_horizontal_error = horizontal_error
            last_vertical_error = vertical_error

            inside_tolerance = (
                horizontal_error <= POSITION_TOLERANCE_M
                and vertical_error <= POSITION_TOLERANCE_M
            )

            if inside_tolerance:
                good_samples += 1
            else:
                good_samples = 0

            projectairsim_log().info(
                "Settling %s | actual=(%.7f, %.7f, %.7f) | "
                "horizontal error=%.3f m vertical error=%.3f m | "
                "good samples=%d/%d",
                target["id"],
                current_x,
                current_y,
                current_z,
                horizontal_error,
                vertical_error,
                good_samples,
                REQUIRED_GOOD_SAMPLES,
            )

            if good_samples >= REQUIRED_GOOD_SAMPLES:
                projectairsim_log().info(
                    "Confirmed %s within %.2f m tolerance for %d "
                    "consecutive samples | accepted position="
                    "(%.7f, %.7f, %.7f)",
                    target["id"],
                    POSITION_TOLERANCE_M,
                    REQUIRED_GOOD_SAMPLES,
                    current_x,
                    current_y,
                    current_z,
                )
                return current_x, current_y, current_z

            await asyncio.sleep(POSITION_CHECK_INTERVAL_SEC)

        if attempt < MAX_POSITION_CORRECTIONS:
            projectairsim_log().warning(
                "%s did not settle within %.2f m after %.1f s | "
                "last horizontal error=%.3f m vertical error=%.3f m; "
                "starting correction %d/%d at %.1f m/s",
                target["id"],
                POSITION_TOLERANCE_M,
                SETTLE_TIMEOUT_SEC,
                last_horizontal_error,
                last_vertical_error,
                attempt + 1,
                MAX_POSITION_CORRECTIONS,
                FINAL_APPROACH_VELOCITY_MPS,
            )

    raise RuntimeError(
        f"Could not settle within {POSITION_TOLERANCE_M:.2f} m of "
        f"{target['id']} after {MAX_POSITION_CORRECTIONS + 1} attempts. "
        f"Last position=({last_x:.6f}, {last_y:.6f}, {last_z:.6f}), "
        f"horizontal error={last_horizontal_error:.3f} m, "
        f"vertical error={last_vertical_error:.3f} m."
    )


async def climb_to_flight_altitude(drone):
    """
    Climb vertically to FLIGHT_ALTITUDE_M before beginning horizontal flight.

    Project AirSim uses NED coordinates, so altitude above the start point is
    represented by a negative Down value.
    """
    pose = drone.get_ground_truth_pose()
    position = pose["translation"]

    current_x = float(position["x"])
    current_y = float(position["y"])

    projectairsim_log().info(
        "Climbing vertically to %.1f m before horizontal flight",
        FLIGHT_ALTITUDE_M,
    )

    climb_task = await drone.move_to_position_async(
        north=current_x,
        east=current_y,
        down=-FLIGHT_ALTITUDE_M,
        velocity=FLIGHT_VELOCITY_MPS,
    )
    await climb_task

    projectairsim_log().info(
        "Reached flight altitude %.1f m",
        FLIGHT_ALTITUDE_M,
    )


# ---------------------------------------------------------------------------
# Main flight
# ---------------------------------------------------------------------------

async def main():
    client = ProjectAirSimClient()

    try:
        client.connect()

        world = World(
            client,
            SCENE_CONFIG,
            delay_after_load_sec=2,
        )

        drone = Drone(
            client,
            world,
            DRONE_NAME,
        )

        drone.enable_api_control()
        drone.arm()

        # Ensure the Unreal Saved directory exists before writing telemetry.
        TELEMETRY_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        targets = load_flightpath()

        projectairsim_log().info(
            "Loaded %d plant targets",
            len(targets),
        )

        if not targets:
            projectairsim_log().warning(
                "No targets found in flightpath.csv"
            )
            return

        # Assign segmentation IDs once, before the flight begins.
        configure_plant_segmentation(
            world,
            targets,
        )

        # Initial takeoff.
        projectairsim_log().info("Taking off")

        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        # First climb vertically above the takeoff point. Only once the drone
        # is clear of the trees do we allow any horizontal movement.
        await climb_to_flight_altitude(drone)

        # Visit every plant individually so we always know which ID
        # the drone is currently flying towards.
        for index, target in enumerate(targets):
            projectairsim_log().info(
                "Flying to %s at X=%.3f, Y=%.3f",
                target["id"],
                target["x"],
                target["y"],
            )

            # Run telemetry + visible-pixel measurement at the same time
            # as the flight command.
            telemetry_task = asyncio.create_task(
                display_telemetry(
                    drone,
                    target,
                )
            )

            try:
                # Fly to 30 m directly above the current plant, then verify the
                # ground-truth position and make slower corrections if needed.
                current_x, current_y, current_z = (
                    await move_to_target_precisely(drone, target)
                )

                projectairsim_log().info(
                    "Waypoint accepted for %s - beginning descent | "
                    "target=(%.6f, %.6f, %.6f) | "
                    "actual=(%.6f, %.6f, %.6f)",
                    target["id"],
                    target["x"],
                    target["y"],
                    target["z"],
                    current_x,
                    current_y,
                    current_z,
                )

                # Descend quickly most of the way.
                descent_task = await drone.move_by_velocity_async(
                    v_north=0.0,
                    v_east=0.0,
                    v_down=5.0,
                    duration=5.5,
                )
                await descent_task

                # Then land at the plant's X/Y position for debugging.
                land_task = await drone.land_async()
                await land_task

                projectairsim_log().info(
                    "Landed at %s",
                    target["id"],
                )

            finally:
                telemetry_task.cancel()
                try:
                    await telemetry_task
                except asyncio.CancelledError:
                    pass

                # Move the terminal cursor to a fresh line.
                print()

            # If another target remains, take off again.
            if index < len(targets) - 1:
                projectairsim_log().info(
                    "Taking off for next target"
                )

                takeoff_task = await drone.takeoff_async()
                await takeoff_task

                # Again, gain full altitude directly above the current plant
                # before setting off horizontally toward the next one.
                await climb_to_flight_altitude(drone)

        projectairsim_log().info("Flight path complete")

        drone.disarm()
        drone.disable_api_control()

    except Exception as err:
        projectairsim_log().error(
            f"Exception occurred: {err}",
            exc_info=True,
        )

    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())