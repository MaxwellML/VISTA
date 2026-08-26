import argparse  
import asyncio
import csv
import msvcrt  # Windows CMD keyboard input
from pathlib import Path

import numpy as np

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import SEGMENTATION_PALLETE
from projectairsim.types import ImageType
from projectairsim.utils import projectairsim_log, unpack_image

import cv2
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OTHONNA_CSV = Path(__file__).resolve().parent / "flightpath.csv"
SWEEP_PATH_CSV = Path(__file__).resolve().parent / "sweep_path.csv"

# Unreal reads this file and displays it using the DroneTelemetryHUD actor.
UNREAL_PROJECT_DIR = Path(
    r"C:\Users\cmdp7\OneDrive\Documents\Unreal Projects\RiveleroCaseStudy"
)
TELEMETRY_FILE = UNREAL_PROJECT_DIR / "Saved" / "drone_telemetry.txt"

SCENE_CONFIG = "scene_basic_drone.jsonc"
DRONE_NAME = "Drone1"
SEGMENTATION_CAMERA = "DownCamera"

FLIGHT_ALTITUDE_M = 30.0
FLIGHT_VELOCITY_MPS = 5
TELEMETRY_INTERVAL_SEC = 0.25


# A plant counts as observed as soon 1 of its segmentation pixels are in frame.
MIN_VISIBLE_PIXELS_FOR_OBSERVATION = 1



MANUAL_HORIZONTAL_SPEED_MPS = 3.0
MANUAL_VERTICAL_SPEED_MPS = 2.0
MANUAL_COMMAND_DURATION_SEC = 0.20
MANUAL_STOP_DURATION_SEC = 0.05


# Simple waypoint handling.
# Each move has a finite controller timeout, then we do one brief ground-truth
# check before continuing to the next waypoint.
POSITION_TOLERANCE_M = 2.0
WAYPOINT_MOVE_TIMEOUT_SEC = 120.0
WAYPOINT_SETTLE_SEC = 0.25

# ---------------------------------------------------------------------------
# Display segmentation window
# ---------------------------------------------------------------------------


async def camera_debug_viewer(drone, stop_event):
    cv2.namedWindow("DownCamera - Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DownCamera - X-Ray", cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            images = drone.get_images(
                "DownCamera",
                [
                    ImageType.SEGMENTATION,
                    6,  # Rivelero X-ray channel
                ],
            )

            if images:
                segmentation_response = images[ImageType.SEGMENTATION]
                xray_response = images[6]

                segmentation_image = unpack_image(segmentation_response)
                xray_image = unpack_image(xray_response)

                cv2.imshow(
                    "DownCamera - Segmentation",
                    segmentation_image,
                )

                cv2.imshow(
                    "DownCamera - X-Ray",
                    xray_image,
                )

                # One waitKey() handles both OpenCV windows.
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

            await asyncio.sleep(0.05)

    finally:
        cv2.destroyWindow("DownCamera - Segmentation")
        cv2.destroyWindow("DownCamera - X-Ray")

# ---------------------------------------------------------------------------
# Load plant targets
# ---------------------------------------------------------------------------


def load_othonnas():
    """Load Othonna actor IDs and their segmentation IDs from flightpath.csv."""
    targets = []

    with OTHONNA_CSV.open(
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

                # Retained as metadata only. These coordinates are NOT used as
                # destinations during the sweep.
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
            })

    return targets


def load_sweep_path():
    """
    Load the lawnmower route from sweep_path.csv.

    Example:
        id,x_m,y_m
        A,...
        B,...
        C,...
        D,...

    The drone follows these rows consecutively. The rows are survey waypoints,
    not Othonna locations.
    """
    waypoints = []

    with SWEEP_PATH_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"id", "x_m", "y_m"}
        if reader.fieldnames is None:
            raise RuntimeError("sweep_path.csv has no header row.")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "sweep_path.csv is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            waypoint_id = row["id"].strip()

            waypoints.append({
                "id": waypoint_id,
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                "z": -FLIGHT_ALTITUDE_M,
            })

    if len(waypoints) < 2:
        raise RuntimeError(
            "sweep_path.csv must contain at least two waypoints."
        )

    return waypoints

def create_observation_status(targets):
    """Create persistent per-Othonna observation records for this survey."""
    return {
        target["id"]: {
            "is_observed": False,
            "max_visible_pixels": 0,
            "max_xray_pixels": 0,
            "max_visible_fraction": 0.0,
        }
        for target in targets
    }


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


def get_all_target_pixel_counts(drone, targets):
    """
    Return visible/X-ray pixel measurements for every Othonna in the same pair
    of camera frames.

    The returned dictionary is keyed by plant ID, e.g. "Othonna_017".
    """
    # Request both images together so every Othonna is measured from the same
    # camera position and instant.
    responses = drone.get_images(
        SEGMENTATION_CAMERA,
        [
            ImageType.SEGMENTATION,
            6,  # Rivelero X-ray channel
        ],
    )

    segmentation_image = unpack_image(
        responses[ImageType.SEGMENTATION]
    )
    xray_image = unpack_image(
        responses[6]
    )

    # Project AirSim's palette is RGB, but unpack_image gives us BGR here.
    segmentation_rgb = segmentation_image[:, :, :3][:, :, ::-1]
    xray_rgb = xray_image[:, :, :3][:, :, ::-1]

    measurements = {}

    for target in targets:
        # Each Othonna keeps its own segmentation ID/colour, so a single image
        # can tell us exactly which individual plants are visible.
        plant_colour = np.asarray(
            SEGMENTATION_PALLETE[target["segmentation_id"]],
            dtype=np.uint8,
        )

        segmentation_mask = np.all(
            segmentation_rgb == plant_colour,
            axis=2,
        )

        xray_mask = np.all(
            xray_rgb == plant_colour,
            axis=2,
        )

        visible_pixels = int(np.count_nonzero(segmentation_mask))
        xray_pixels = int(np.count_nonzero(xray_mask))

        if xray_pixels > 0:
            visible_fraction = visible_pixels / xray_pixels
        else:
            visible_fraction = 0.0

        measurements[target["id"]] = {
            "visible_pixels": visible_pixels,
            "xray_pixels": xray_pixels,
            "visible_fraction": visible_fraction,
        }

    return measurements


def update_observation_status(observation_status, measurements):
    """
    Update the persistent survey record from one camera frame.

    Once an Othonna becomes observed it remains observed for the rest of the
    survey. We also retain its best pixel/fraction measurements.
    """
    for plant_id, measurement in measurements.items():
        status = observation_status[plant_id]

        visible_pixels = measurement["visible_pixels"]
        xray_pixels = measurement["xray_pixels"]
        visible_fraction = measurement["visible_fraction"]

        if visible_pixels >= MIN_VISIBLE_PIXELS_FOR_OBSERVATION:
            if not status["is_observed"]:
                projectairsim_log().info(
                    "OBSERVED %s for the first time (%d visible pixels)",
                    plant_id,
                    visible_pixels,
                )

            status["is_observed"] = True

        status["max_visible_pixels"] = max(
            status["max_visible_pixels"],
            visible_pixels,
        )
        status["max_xray_pixels"] = max(
            status["max_xray_pixels"],
            xray_pixels,
        )
        status["max_visible_fraction"] = max(
            status["max_visible_fraction"],
            visible_fraction,
        )


def print_observation_summary(observation_status):
    """Print the final observed/not-observed state for every Othonna."""
    observed_count = sum(
        status["is_observed"]
        for status in observation_status.values()
    )
    total_count = len(observation_status)

    print("\n=== OTHONNA OBSERVATION SUMMARY ===")

    for plant_id, status in observation_status.items():
        state = "OBSERVED" if status["is_observed"] else "NOT OBSERVED"

        print(
            f"{plant_id}: {state} | "
            f"max visible pixels={status['max_visible_pixels']} | "
            f"max X-ray pixels={status['max_xray_pixels']} | "
            f"max visible fraction={status['max_visible_fraction']:.3f}"
        )

    if total_count > 0:
        recall = observed_count / total_count
    else:
        recall = 0.0

    print(
        f"Observed {observed_count}/{total_count} Othonnas "
        f"({recall:.1%} recall)"
    )


# ---------------------------------------------------------------------------
# Live telemetry
# ---------------------------------------------------------------------------


async def display_telemetry(drone, targets, observation_status):
    """
    Continuously inspect the current camera frame during the sweep.
    """
    while True:
        pose = drone.get_ground_truth_pose()
        position = pose["translation"]

        x = float(position["x"])
        y = float(position["y"])
        z = float(position["z"])

        measurements = get_all_target_pixel_counts(
            drone,
            targets,
        )

        update_observation_status(
            observation_status,
            measurements,
        )

        observed_count = sum(
            status["is_observed"]
            for status in observation_status.values()
        )

        # Only report Othonnas geometrically inside this camera frame.
        in_frame_measurements = {
            plant_id: measurement
            for plant_id, measurement in measurements.items()
            if measurement["xray_pixels"] > 0
        }

        othonna_lines = []

        for plant_id, measurement in in_frame_measurements.items():
            visible_pixels = measurement["visible_pixels"]
            xray_pixels = measurement["xray_pixels"]
            visible_fraction = measurement["visible_fraction"]
            occluded_fraction = 1.0 - visible_fraction

            othonna_lines.append(
                f"{plant_id}: "
                f"visible pixels={visible_pixels} | "
                f"X-ray pixels={xray_pixels} | "
                f"visible fraction={visible_fraction:.3f} | "
                f"occluded fraction={occluded_fraction:.3f}"
            )

        if not othonna_lines:
            othonna_lines.append("No Othonnas in current camera frame")

        message = (
            f"SURVEY SWEEP\n"
            f"X: {x:.2f} m\n"
            f"Y: {y:.2f} m\n"
            f"Z: {z:.2f} m\n"
            f"Othonnas observed so far: "
            f"{observed_count}/{len(observation_status)}\n"
            + "\n".join(othonna_lines)
        )

        # Terminal version is flattened onto one updating line.
        print(
            message.replace("\n", " | ").ljust(220),
            end="\r",
            flush=True,
        )

        # Unreal HUD gets the proper multi-line version.
        for attempt in range(3):
            try:
                TELEMETRY_FILE.write_text(
                    message,
                    encoding="utf-8",
                )
                break

            except PermissionError as e:
                if attempt < 2:
                    await asyncio.sleep(0.1)
                else:
                    print(f"\n[WARNING] Could not write telemetry file: {e}")

        await asyncio.sleep(TELEMETRY_INTERVAL_SEC)

# ---------------------------------------------------------------------------
#  Manual keyboard control
# ---------------------------------------------------------------------------

async def manual_keyboard_control(drone):
    """
    Control the drone from the Windows CMD/terminal window.

    Project AirSim uses NED coordinates:
      W/S -> north/south
      A/D -> west/east
      R/F -> up/down
      T   -> take off
      X   -> land
      Q   -> leave manual mode

    The movement keys send short velocity pulses. Holding a key also works
    through Windows key-repeat, while releasing it lets the drone settle.
    """

    print(
        "\n"
        "=== MANUAL DRONE CONTROL ===\n"
        "T       Take off\n"
        "W / S   North / South\n"
        "A / D   West / East\n"
        "R / F   Up / Down\n"
        "X       Land\n"
        "Q       Quit manual mode (lands first)\n"
        "\n"
        "Keep this CMD window focused while flying.\n"
    )

    velocity_by_key = {
        "w": (MANUAL_HORIZONTAL_SPEED_MPS, 0.0, 0.0),
        "s": (-MANUAL_HORIZONTAL_SPEED_MPS, 0.0, 0.0),
        "a": (0.0, -MANUAL_HORIZONTAL_SPEED_MPS, 0.0),
        "d": (0.0, MANUAL_HORIZONTAL_SPEED_MPS, 0.0),
        # NED uses positive Down, so Up is a negative down-velocity.
        "r": (0.0, 0.0, -MANUAL_VERTICAL_SPEED_MPS),
        "f": (0.0, 0.0, MANUAL_VERTICAL_SPEED_MPS),
    }

    while True:
        # msvcrt.kbhit() is non-blocking, so the asyncio camera viewer can keep
        # refreshing while we wait for a key.
        if not msvcrt.kbhit():
            await asyncio.sleep(0.02)
            continue

        key = msvcrt.getwch().lower()

        # Ignore the prefix bytes used by Windows for special keys such as
        # arrows/F-keys. We deliberately use ordinary letter keys here.
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            continue

        if key == "q":
            projectairsim_log().info(
                "Leaving manual control - landing first"
            )
            land_task = await drone.land_async()
            await land_task
            break

        if key == "t":
            projectairsim_log().info("Manual control: takeoff")
            takeoff_task = await drone.takeoff_async()
            await takeoff_task
            continue

        if key == "x":
            projectairsim_log().info("Manual control: landing")
            land_task = await drone.land_async()
            await land_task
            continue

        velocity = velocity_by_key.get(key)
        if velocity is None:
            continue

        v_north, v_east, v_down = velocity

        move_task = await drone.move_by_velocity_async(
            v_north=v_north,
            v_east=v_east,
            v_down=v_down,
            duration=MANUAL_COMMAND_DURATION_SEC,
        )
        await move_task

        # Explicitly command zero velocity after the pulse so a single tap does
        # not leave the previous velocity setpoint active.
        stop_task = await drone.move_by_velocity_async(
            v_north=0.0,
            v_east=0.0,
            v_down=0.0,
            duration=MANUAL_STOP_DURATION_SEC,
        )
        await stop_task


# ---------------------------------------------------------------------------
# Flight helpers
# ---------------------------------------------------------------------------


async def move_to_waypoint_precisely(drone, waypoint):
    """
    Fly once to a waypoint, briefly check the resulting ground-truth position,
    then continue.

    There is deliberately no correction loop here: sweep waypoints define the
    route, not precision hover targets.
    """
    projectairsim_log().info(
        "Moving to %s at %.1f m/s | waypoint NED=(%.3f, %.3f, %.3f)",
        waypoint["id"],
        FLIGHT_VELOCITY_MPS,
        waypoint["x"],
        waypoint["y"],
        waypoint["z"],
    )

    move_task = await drone.move_to_position_async(
        north=waypoint["x"],
        east=waypoint["y"],
        down=waypoint["z"],
        velocity=FLIGHT_VELOCITY_MPS,
        timeout_sec=WAYPOINT_MOVE_TIMEOUT_SEC,
    )
    await move_task

    # Give the vehicle a moment to settle before sampling its actual position.
    await asyncio.sleep(WAYPOINT_SETTLE_SEC)

    pose = drone.get_ground_truth_pose()
    position = pose["translation"]

    current_x = float(position["x"])
    current_y = float(position["y"])
    current_z = float(position["z"])

    dx = waypoint["x"] - current_x
    dy = waypoint["y"] - current_y
    dz = waypoint["z"] - current_z

    horizontal_error = (dx**2 + dy**2) ** 0.5
    vertical_error = abs(dz)

    if (
        horizontal_error > POSITION_TOLERANCE_M
        or vertical_error > POSITION_TOLERANCE_M
    ):
        projectairsim_log().warning(
            "%s reached with larger-than-expected error | "
            "actual=(%.3f, %.3f, %.3f) | "
            "horizontal error=%.2f m vertical error=%.2f m | continuing",
            waypoint["id"],
            current_x,
            current_y,
            current_z,
            horizontal_error,
            vertical_error,
        )
    else:
        projectairsim_log().info(
            "Reached %s | actual=(%.3f, %.3f, %.3f) | "
            "horizontal error=%.2f m vertical error=%.2f m",
            waypoint["id"],
            current_x,
            current_y,
            current_z,
            horizontal_error,
            vertical_error,
        )

    return current_x, current_y, current_z


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
        timeout_sec=WAYPOINT_MOVE_TIMEOUT_SEC,
    )
    await climb_task

    projectairsim_log().info(
        "Reached flight altitude %.1f m",
        FLIGHT_ALTITUDE_M,
    )


# ---------------------------------------------------------------------------
# Main flight
# ---------------------------------------------------------------------------

async def main(manual=False): 
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

        targets = load_othonnas()

        #persistent per-Othonna survey state, initially all False
        observation_status = create_observation_status(targets)
   

        projectairsim_log().info(
            "Loaded %d plant targets",
            len(targets),
        )

   
        # Manual control itself does not require a sweep path.
        # If targets do exist, keep configuring their segmentation IDs so the
        # existing debug camera remains useful in manual mode.
        if targets:
            configure_plant_segmentation(
                world,
                targets,
            )
        elif manual:
            projectairsim_log().warning(
                "No Othonnas found in flightpath.csv; manual control will "
                "continue without plant-specific segmentation IDs"
            )
        else:
            projectairsim_log().warning(
                "No Othonnas found in flightpath.csv"
            )
            return
    

        # Manual mode is only for debugging, so its camera viewer may start
        # immediately.
        if manual:
            projectairsim_log().info("Manual keyboard-control mode enabled")

            segmentation_stop = asyncio.Event()
            segmentation_task = asyncio.create_task(
                camera_debug_viewer(drone, segmentation_stop)
            )

            try:
                await manual_keyboard_control(drone)
            finally:
                segmentation_stop.set()
                await segmentation_task


                # Safety fallback: if manual mode exits because of an exception,
                # attempt to land before disarming rather than dropping the drone.
                try:
                    land_task = await drone.land_async()
                    await land_task
                except Exception as land_err:
                    projectairsim_log().warning(
                        "Could not complete cleanup landing: %s",
                        land_err,
                    )

                drone.disarm()
                drone.disable_api_control()


            return


        # Initial takeoff.
        projectairsim_log().info("Taking off")

        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        # First climb vertically above the takeoff point. Only once the drone
        # is clear of the trees do we allow any horizontal movement.
        await climb_to_flight_altitude(drone)

        sweep_waypoints = load_sweep_path()

        projectairsim_log().info(
            "Loaded %d sweep waypoints",
            len(sweep_waypoints),
        )

        # Reach the first survey waypoint before we start counting observations.
        # This prevents plants seen during the initial transit from contaminating
        # the survey result.
        first_waypoint = sweep_waypoints[0]

        projectairsim_log().info(
            "Moving to sweep start %s at X=%.3f, Y=%.3f",
            first_waypoint["id"],
            first_waypoint["x"],
            first_waypoint["y"],
        )

        await move_to_waypoint_precisely(
            drone,
            first_waypoint,
        )

        # The autonomous survey cameras start ONLY after the drone
        # has reached and settled at the first sweep waypoint.
        projectairsim_log().info(
            "Reached survey start %s - starting cameras",
            first_waypoint["id"],
        )

        segmentation_stop = asyncio.Event()
        segmentation_task = asyncio.create_task(
            camera_debug_viewer(drone, segmentation_stop)
        )

        # Observation/pixel counting also starts here, so nothing seen during
        # takeoff or transit to the start point can count towards the survey.
        telemetry_task = asyncio.create_task(
            display_telemetry(
                drone,
                targets,
                observation_status,
            )
        )


        try:
            # Give the observer one frame at the starting waypoint.
            await asyncio.sleep(TELEMETRY_INTERVAL_SEC)

            for waypoint in sweep_waypoints[1:]:
                projectairsim_log().info(
                    "Sweeping to waypoint %s at X=%.3f, Y=%.3f",
                    waypoint["id"],
                    waypoint["x"],
                    waypoint["y"],
                )

                await move_to_waypoint_precisely(
                    drone,
                    waypoint,
                )

        finally:
            # stop observation and camera display as soon as the
            # survey path ends. Landing is outside the survey.
            telemetry_task.cancel()
            try:
                await telemetry_task
            except asyncio.CancelledError:
                pass

            segmentation_stop.set()
            await segmentation_task

            projectairsim_log().info("Survey cameras stopped")
   
            print()

        # The survey is one continuous flight, so land only after the final
        # sweep waypoint has been reached.
        projectairsim_log().info("Sweep complete - landing")

        land_task = await drone.land_async()
        await land_task

        projectairsim_log().info("Survey sweep complete")


        print_observation_summary(observation_status)
        drone.disarm()
        drone.disable_api_control()

    except Exception as err:
        projectairsim_log().error(
            f"Exception occurred: {err}",
            exc_info=True,
        )

    finally:
        client.disconnect()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Rivelero Project AirSim drone mission."
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use keyboard control instead of the automatic flight path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(manual=args.manual))
