import argparse  
import asyncio
import csv
import msvcrt  # Windows CMD keyboard input
import ctypes   # Windows held-key state for smooth manual control
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
MANUAL_COMMAND_DURATION_SEC = 0.10
MANUAL_CONTROL_LOOP_SEC = 0.01


# Simple waypoint handling.
# Each move has a finite controller timeout, then we do one brief ground-truth
# check before continuing to the next waypoint.
POSITION_TOLERANCE_M = 5.0
WAYPOINT_SETTLE_SEC = 0.25


# Timeout now scales with the distance to the waypoint instead of using one
# short fixed value for every leg.
WAYPOINT_TIMEOUT_SAFETY_FACTOR = 1.5
WAYPOINT_TIMEOUT_BUFFER_SEC = 10.0
MIN_WAYPOINT_TIMEOUT_SEC = 15.0

# Log an error if the drone is still away from its waypoint but its measured
# ground-truth speed stays near zero for this long.
MOTION_CHECK_INTERVAL_SEC = 0.5
STALL_SPEED_THRESHOLD_MPS = 0.25
STALL_DURATION_SEC = 2.0
STALL_STARTUP_GRACE_SEC = 2.0


MAX_WAYPOINT_ATTEMPTS = 2


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

        actual_id = world.get_segmentation_id_by_name(
            target["id"],
            True,
        )       

        print(
            target["id"],
            "expected =", target["segmentation_id"],
            "actual =", actual_id,
            "colour =", SEGMENTATION_PALLETE[target["segmentation_id"]],
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
    Control the drone from the Windows keyboard.

    Project AirSim uses NED coordinates:
      W/S -> north/south
      A/D -> west/east
      R/F -> up/down
      T   -> take off
      X   -> land
      Q   -> leave manual mode

    Movement keys are polled from their actual held state with
    GetAsyncKeyState(), rather than relying on Windows terminal key-repeat.
    This avoids the old move/stop/move/stop pulsing behaviour.
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

    # Virtual-key codes for the movement keys. GetAsyncKeyState() returns a
    # value with the high bit set while that key is physically held down.
    movement_vk = {
        "w": ord("W"),
        "s": ord("S"),
        "a": ord("A"),
        "d": ord("D"),
        "r": ord("R"),
        "f": ord("F"),
    }

    def key_is_down(key):
        return bool(ctypes.windll.user32.GetAsyncKeyState(movement_vk[key]) & 0x8000)

    was_moving = False

    while True:
        # Handle one-shot commands through the terminal as before. Drain all
        # queued keypresses so movement-key repeats do not build up in msvcrt.
        while msvcrt.kbhit():
            key = msvcrt.getwch().lower()

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
                return

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

        # Read the current physical key state. Opposite keys cancel each other,
        # and diagonal movement is allowed naturally.
        north_input = int(key_is_down("w")) - int(key_is_down("s"))
        east_input = int(key_is_down("d")) - int(key_is_down("a"))
        down_input = int(key_is_down("f")) - int(key_is_down("r"))

        v_north = north_input * MANUAL_HORIZONTAL_SPEED_MPS
        v_east = east_input * MANUAL_HORIZONTAL_SPEED_MPS
        v_down = down_input * MANUAL_VERTICAL_SPEED_MPS

        moving = any((north_input, east_input, down_input))

        if moving:
            # Send consecutive short velocity commands while a key is held.
            # Crucially, there is NO zero-velocity command between them.
            move_task = await drone.move_by_velocity_async(
                v_north=v_north,
                v_east=v_east,
                v_down=v_down,
                duration=MANUAL_COMMAND_DURATION_SEC,
            )
            await move_task
            was_moving = True
            continue

        if was_moving:
            # A zero command is sent only once, when all movement keys have
            # actually been released.
            stop_task = await drone.move_by_velocity_async(
                v_north=0.0,
                v_east=0.0,
                v_down=0.0,
                duration=MANUAL_COMMAND_DURATION_SEC,
            )
            await stop_task
            was_moving = False

        await asyncio.sleep(MANUAL_CONTROL_LOOP_SEC)


# ---------------------------------------------------------------------------
# Flight helpers
# ---------------------------------------------------------------------------



async def monitor_waypoint_motion(drone, waypoint):
    """Return True if the drone effectively stops before reaching a waypoint."""
    loop = asyncio.get_running_loop()

    pose = drone.get_ground_truth_pose()
    position = pose["translation"]
    previous_x = float(position["x"])
    previous_y = float(position["y"])
    previous_z = float(position["z"])
    previous_time = loop.time()
    monitor_started = previous_time

    stall_started = None

    while True:
        await asyncio.sleep(MOTION_CHECK_INTERVAL_SEC)

        pose = drone.get_ground_truth_pose()
        position = pose["translation"]
        current_x = float(position["x"])
        current_y = float(position["y"])
        current_z = float(position["z"])
        current_time = loop.time()

        dt = max(current_time - previous_time, 1e-6)
        travelled = (
            (current_x - previous_x) ** 2
            + (current_y - previous_y) ** 2
            + (current_z - previous_z) ** 2
        ) ** 0.5
        measured_speed = travelled / dt

        dx = waypoint["x"] - current_x
        dy = waypoint["y"] - current_y
        dz = waypoint["z"] - current_z
        horizontal_error = (dx**2 + dy**2) ** 0.5
        vertical_error = abs(dz)

        at_waypoint = (
            horizontal_error <= POSITION_TOLERANCE_M
            and vertical_error <= POSITION_TOLERANCE_M
        )
        past_startup_grace = (
            current_time - monitor_started >= STALL_STARTUP_GRACE_SEC
        )

        if (
            past_startup_grace
            and not at_waypoint
            and measured_speed < STALL_SPEED_THRESHOLD_MPS
        ):
            if stall_started is None:
                stall_started = current_time
            elif current_time - stall_started >= STALL_DURATION_SEC:
                projectairsim_log().error(
                    "DRONE STALL detected while moving to %s | "
                    "measured speed=%.2f m/s | actual=(%.3f, %.3f, %.3f) | "
                    "horizontal error=%.2f m vertical error=%.2f m",
                    waypoint["id"],
                    measured_speed,
                    current_x,
                    current_y,
                    current_z,
                    horizontal_error,
                    vertical_error,
                )

                #report the stall to the waypoint mover immediately
                # so it can cancel this controller command and retry the SAME
                # waypoint instead of waiting for the long movement timeout.
                return True
      
        else:
            stall_started = None

        previous_x = current_x
        previous_y = current_y
        previous_z = current_z
        previous_time = current_time



async def move_to_waypoint_precisely(drone, waypoint):
    """
    Fly to a waypoint and verify the resulting ground-truth position.

    If the drone stalls, or the movement command ends outside the waypoint
    tolerance, retry the SAME waypoint once. A second failure aborts the mission.
    """

   
    last_failure = None

    for attempt in range(1, MAX_WAYPOINT_ATTEMPTS + 1):
        # Recalculate distance on every attempt because a retry starts from the
        # drone's new/current position.
        start_pose = drone.get_ground_truth_pose()
        start_position = start_pose["translation"]
        start_x = float(start_position["x"])
        start_y = float(start_position["y"])
        start_z = float(start_position["z"])

        start_dx = waypoint["x"] - start_x
        start_dy = waypoint["y"] - start_y
        start_dz = waypoint["z"] - start_z
        distance = (start_dx**2 + start_dy**2 + start_dz**2) ** 0.5

        expected_time = distance / FLIGHT_VELOCITY_MPS
        timeout_sec = max(
            MIN_WAYPOINT_TIMEOUT_SEC,
            expected_time * WAYPOINT_TIMEOUT_SAFETY_FACTOR
            + WAYPOINT_TIMEOUT_BUFFER_SEC,
        )

        if attempt > 1:
            projectairsim_log().warning(
                "Retrying SAME waypoint %s | attempt %d/%d",
                waypoint["id"],
                attempt,
                MAX_WAYPOINT_ATTEMPTS,
            )

        projectairsim_log().info(
            "Moving to %s at %.1f m/s | waypoint NED=(%.3f, %.3f, %.3f) | "
            "distance=%.1f m expected=%.1f s timeout=%.1f s | attempt=%d/%d",
            waypoint["id"],
            FLIGHT_VELOCITY_MPS,
            waypoint["x"],
            waypoint["y"],
            waypoint["z"],
            distance,
            expected_time,
            timeout_sec,
            attempt,
            MAX_WAYPOINT_ATTEMPTS,
        )

        controller_task = await drone.move_to_position_async(
            north=waypoint["x"],
            east=waypoint["y"],
            down=waypoint["z"],
            velocity=FLIGHT_VELOCITY_MPS,
            timeout_sec=timeout_sec,
        )

        # ensure_future accepts either the asyncio Task returned by Project
        # AirSim or another awaitable representing the controller request.
        move_task = asyncio.ensure_future(controller_task)
        motion_monitor = asyncio.create_task(
            monitor_waypoint_motion(drone, waypoint)
        )

        stall_detected = False

        try:
            done, _ = await asyncio.wait(
                {move_task, motion_monitor},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if motion_monitor in done:
                stall_detected = bool(motion_monitor.result())

            if stall_detected:
                # Stop the stuck Project AirSim controller command before
                # issuing the retry.
                cancelled = drone.cancel_last_task()
                projectairsim_log().error(
                    "Cancelled stalled movement to %s | "
                    "cancel_last_task returned %s",
                    waypoint["id"],
                    cancelled,
                )

                if not move_task.done():
                    move_task.cancel()
                    try:
                        await move_task
                    except asyncio.CancelledError:
                        pass

                last_failure = RuntimeError(
                    f"Drone stalled while moving to waypoint {waypoint['id']}"
                )

            else:
                # The movement command finished first. Propagate any controller
                # exception before checking the final ground-truth position.
                await move_task

        finally:
            if not motion_monitor.done():
                motion_monitor.cancel()
                try:
                    await motion_monitor
                except asyncio.CancelledError:
                    pass

        if stall_detected:
            if attempt < MAX_WAYPOINT_ATTEMPTS:
                await asyncio.sleep(WAYPOINT_SETTLE_SEC)
                continue

            projectairsim_log().error(
                "Waypoint %s stalled again on retry - aborting mission",
                waypoint["id"],
            )
            raise last_failure

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
            horizontal_error <= POSITION_TOLERANCE_M
            and vertical_error <= POSITION_TOLERANCE_M
        ):
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

        projectairsim_log().error(
            "%s movement ended before waypoint was reached | "
            "actual=(%.3f, %.3f, %.3f) | "
            "horizontal error=%.2f m vertical error=%.2f m | attempt=%d/%d",
            waypoint["id"],
            current_x,
            current_y,
            current_z,
            horizontal_error,
            vertical_error,
            attempt,
            MAX_WAYPOINT_ATTEMPTS,
        )

        last_failure = RuntimeError(
            f"Failed to reach waypoint {waypoint['id']}: "
            f"horizontal error={horizontal_error:.2f} m, "
            f"vertical error={vertical_error:.2f} m"
        )

        if attempt < MAX_WAYPOINT_ATTEMPTS:
            continue

        projectairsim_log().error(
            "Waypoint %s failed again on retry - aborting mission",
            waypoint["id"],
        )
        raise last_failure

    # Defensive fallback; the loop above always returns or raises.
    raise last_failure



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

    current_z = float(position["z"])
    climb_distance = abs((-FLIGHT_ALTITUDE_M) - current_z)
    climb_expected_time = climb_distance / FLIGHT_VELOCITY_MPS
    climb_timeout_sec = max(
        MIN_WAYPOINT_TIMEOUT_SEC,
        climb_expected_time * WAYPOINT_TIMEOUT_SAFETY_FACTOR
        + WAYPOINT_TIMEOUT_BUFFER_SEC,
    )

    climb_task = await drone.move_to_position_async(
        north=current_x,
        east=current_y,
        down=-FLIGHT_ALTITUDE_M,
        velocity=FLIGHT_VELOCITY_MPS,
        timeout_sec=climb_timeout_sec,
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

            # Run the same per-Othonna pixel telemetry used by the autonomous
            # survey while the drone is being flown manually.
            telemetry_task = asyncio.create_task(
                display_telemetry(
                    drone,
                    targets,
                    observation_status,
                )
            )

            try:
                await manual_keyboard_control(drone)
            finally:
                telemetry_task.cancel()
                try:
                    await telemetry_task
                except asyncio.CancelledError:
                    pass

                segmentation_stop.set()
                await segmentation_task

                print()
                print_observation_summary(observation_status)

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
