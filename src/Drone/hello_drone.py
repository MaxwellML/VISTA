import argparse  
import asyncio
import csv
import msvcrt  # Windows CMD keyboard input
import ctypes   # Windows held-key state for smooth manual control
from pathlib import Path

import numpy as np

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.drone import YawControlMode  # ADDED: continuous orientation correction
from projectairsim.image_utils import SEGMENTATION_PALLETE
from projectairsim.types import ImageType
from projectairsim.utils import (
    projectairsim_log,
    quaternion_to_rpy,  # ADDED: read actual drone yaw every control cycle
    unpack_image,
)

import cv2
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OTHONNA_CSV = Path(__file__).resolve().parent / "flightpath.csv"
SWEEP_PATH_CSV = Path(__file__).resolve().parent / "sweep_path.csv"
SURVEY_RESULTS_CSV = Path(__file__).resolve().parent / "survey_results.csv"

# Unreal reads this file and displays it using the DroneTelemetryHUD actor.
UNREAL_PROJECT_DIR = Path(
    r"C:\Users\cmdp7\OneDrive\Documents\Unreal Projects\RiveleroCaseStudy"
)
TELEMETRY_FILE = UNREAL_PROJECT_DIR / "Saved" / "drone_telemetry.txt"

SCENE_CONFIG = "scene_basic_drone.jsonc"
DRONE_NAME = "Drone1"
SEGMENTATION_CAMERA = "DownCamera"

FLIGHT_ALTITUDE_M = 30.0
FLIGHT_VELOCITY_MPS = 7
TELEMETRY_INTERVAL_SEC = 0.25

# ADDED: CONTINUOUS ORIENTATION CORRECTION
# Keep the drone at one absolute yaw throughout the autonomous survey.
# Project AirSim yaw is in radians in NED/world coordinates: 0 rad = North.
# The autonomous waypoint controller below now re-reads the ACTUAL yaw and
# reissues this absolute yaw target on every short control pulse.
SURVEY_YAW_DEG = 0.0
SURVEY_YAW_RAD = float(np.deg2rad(SURVEY_YAW_DEG))
YAW_CORRECTION_MARGIN_DEG = 1.0
YAW_CORRECTION_TIMEOUT_SEC = 10.0

# ADDED: CONTINUOUS ORIENTATION CORRECTION
# Recompute position + orientation control at 10 Hz instead of issuing one long
# waypoint command and relying on its initial yaw request for the entire leg.
AUTONOMOUS_CONTROL_INTERVAL_SEC = 0.10
YAW_WARNING_LOG_INTERVAL_SEC = 1.0



MIN_VISIBLE_FRACTION_FOR_OBSERVATION = 0.90

# An Othonna can only count as observed while the drone is within this absolute
# X/Y (planar Euclidean) distance of it. flightpath.csv currently provides only
# target X/Y coordinates, so no target Z value is invented here.
MAX_OBSERVATION_DISTANCE_M = 50.0



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


def update_observation_status(
    observation_status,
    measurements,
    targets_by_id,
    drone_x,
    drone_y,
):
    """
    Update the persistent survey record from one camera frame.

    Once an Othonna becomes observed it remains observed for the rest of the
    survey. We also retain its best pixel/fraction measurements.

    A camera observation is only successful when BOTH conditions are true:
      1. the visible fraction meets MIN_VISIBLE_FRACTION_FOR_OBSERVATION; and
      2. the drone is within MAX_OBSERVATION_DISTANCE_M of the target.

    """
    for plant_id, measurement in measurements.items():
        status = observation_status[plant_id]

        visible_pixels = measurement["visible_pixels"]
        xray_pixels = measurement["xray_pixels"]
        visible_fraction = measurement["visible_fraction"]


        target = targets_by_id[plant_id]
        dx = target["x"] - drone_x
        dy = target["y"] - drone_y
        distance_to_target_m = (dx**2 + dy**2) ** 0.5
        within_distance_gate = (
            distance_to_target_m <= MAX_OBSERVATION_DISTANCE_M
        )


        visibility_gate_passed = (
            xray_pixels > 0
            and visible_fraction >= MIN_VISIBLE_FRACTION_FOR_OBSERVATION
        )

        observation_successful = (
            within_distance_gate
            and visibility_gate_passed
        )

 
        measurement["visibility_gate_passed"] = visibility_gate_passed

        # Attach these frame-local values for telemetry/debugging below.
        measurement["distance_to_target_m"] = distance_to_target_m
        measurement["within_distance_gate"] = within_distance_gate

        if observation_successful:  #now hard-gated by distance.
            if not status["is_observed"]:
                projectairsim_log().info(
                    "OBSERVED %s for the first time "
                    "(visible fraction=%.3f >= %.3f, %d/%d pixels, "
                    "distance=%.2f m <= %.2f m)",
                    plant_id,
                    visible_fraction,
                    MIN_VISIBLE_FRACTION_FOR_OBSERVATION,
                    visible_pixels,
                    xray_pixels,
                    distance_to_target_m,
                    MAX_OBSERVATION_DISTANCE_M,
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



def export_survey_results(targets, observation_status):
    """
    Write one summary row per Othonna to survey_results.csv.

    Every target from flightpath.csv is included, even if it was never
    detected by either the normal segmentation camera or the X-ray camera.
    """
    fieldnames = [
        "id",
        "x_m",
        "y_m",
        "observed",
        "xray_detected",
        "max_visible_pixels",
        "max_xray_pixels",
        "max_visible_fraction",
    ]

    with SURVEY_RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for target in targets:
            status = observation_status[target["id"]]

            writer.writerow({
                "id": target["id"],
                "x_m": target["x"],
                "y_m": target["y"],
                "observed": status["is_observed"],
                "xray_detected": status["max_xray_pixels"] > 0,
                "max_visible_pixels": status["max_visible_pixels"],
                "max_xray_pixels": status["max_xray_pixels"],
                "max_visible_fraction": f"{status['max_visible_fraction']:.6f}",
            })

    projectairsim_log().info(
        "Exported per-Othonna survey results to %s",
        SURVEY_RESULTS_CSV,
    )


def calculate_survey_metrics(observation_status, survey_time_sec):
    """Return recall, survey time, and recall-per-second efficiency."""

    observed_count = sum(
        status["is_observed"]
        for status in observation_status.values()
    )
    total_count = len(observation_status)

    recall = (observed_count / total_count) if total_count > 0 else 0.0
    efficiency = (recall / survey_time_sec) if survey_time_sec > 0 else 0.0

    return observed_count, total_count, recall, survey_time_sec, efficiency



def print_observation_summary(observation_status, survey_time_sec):
    """Print per-Othonna status plus the final survey-level metrics."""
    (
        observed_count,
        total_count,
        recall,
        survey_time_sec,
        efficiency,
    ) = calculate_survey_metrics(
        observation_status,
        survey_time_sec,
    )

    print("\n=== OTHONNA OBSERVATION SUMMARY ===")

    for plant_id, status in observation_status.items():
        state = "OBSERVED" if status["is_observed"] else "NOT OBSERVED"

        print(
            f"{plant_id}: {state} | "
            f"max visible pixels={status['max_visible_pixels']} | "
            f"max X-ray pixels={status['max_xray_pixels']} | "
            f"max visible fraction={status['max_visible_fraction']:.3f}"
        )

    print(
        f"Observed {observed_count}/{total_count} Othonnas "
        f"({recall:.1%} recall)"
    )


    print("\n=== SURVEY METRICS ===")
    print(f"Total recall: {recall:.6f} ({recall:.2%})")
    print(f"Total survey time: {survey_time_sec:.3f} s")
    print(f"Efficiency (recall / survey time): {efficiency:.9f} recall/s")

    projectairsim_log().info(
        "SURVEY METRICS | recall=%.6f | survey_time=%.3f s | "
        "efficiency=%.9f recall/s",
        recall,
        survey_time_sec,
        efficiency,
    )




# ADDED: CONTINUOUS ORIENTATION CORRECTION
def wrap_angle_rad(angle):
    """Wrap an angle to [-pi, pi] so yaw error always takes the shortest path."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ADDED: CONTINUOUS ORIENTATION CORRECTION
def get_pose_yaw_rad(pose):
    """Extract yaw in radians from a Project AirSim ground-truth pose."""
    rotation = pose["rotation"]
    _, _, yaw = quaternion_to_rpy(
        float(rotation["w"]),
        float(rotation["x"]),
        float(rotation["y"]),
        float(rotation["z"]),
    )
    return float(yaw)


# ---------------------------------------------------------------------------
# Live telemetry
# ---------------------------------------------------------------------------


async def display_telemetry(drone, targets, observation_status):
    """
    Continuously inspect the current camera frame during the sweep.
    """

    targets_by_id = {target["id"]: target for target in targets} #lookup Othonna data for given ID.


    while True:
        pose = drone.get_ground_truth_pose()
        position = pose["translation"]

        x = float(position["x"])
        y = float(position["y"])
        z = float(position["z"])

        # ADDED: CONTINUOUS ORIENTATION CORRECTION
        # Measure the real ground-truth yaw every telemetry frame so the HUD shows
        # whether the controller is actually keeping the requested world heading.
        actual_yaw_rad = get_pose_yaw_rad(pose)
        yaw_error_rad = wrap_angle_rad(SURVEY_YAW_RAD - actual_yaw_rad)
        actual_yaw_deg = float(np.rad2deg(actual_yaw_rad))
        yaw_error_deg = float(np.rad2deg(yaw_error_rad))

        measurements = get_all_target_pixel_counts(
            drone,
            targets,
        )

        update_observation_status(
            observation_status,
            measurements,
            targets_by_id, 
            x,             
            y,  
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


            distance_to_target_m = measurement["distance_to_target_m"]
            distance_gate_text = (
                "PASS" if measurement["within_distance_gate"] else "FAIL"
            )

            visibility_gate_text = (
                "PASS" if measurement["visibility_gate_passed"] else "FAIL"
            )

            othonna_lines.append(
                f"{plant_id}: "
                f"visible pixels={visible_pixels} | "
                f"X-ray pixels={xray_pixels} | "
                f"visible fraction={visible_fraction:.3f} | "
                f"visibility gate={visibility_gate_text} "
                f"(>= {MIN_VISIBLE_FRACTION_FOR_OBSERVATION:.3f}) | "
                f"occluded fraction={occluded_fraction:.3f} | "
                f"distance={distance_to_target_m:.2f} m | "
                f"distance gate={distance_gate_text}"
            )

        if not othonna_lines:
            othonna_lines.append("No Othonnas in current camera frame")

        message = (
            f"SURVEY SWEEP\n"
            f"X: {x:.2f} m\n"
            f"Y: {y:.2f} m\n"
            f"Z: {z:.2f} m\n"
            # ADDED: CONTINUOUS ORIENTATION CORRECTION
            f"Yaw target: {SURVEY_YAW_DEG:.2f} deg\n"
            f"Yaw actual: {actual_yaw_deg:.2f} deg\n"
            f"Yaw error: {yaw_error_deg:+.2f} deg\n"
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



# ADDED: ORIENTATION CORRECTION
async def correct_survey_orientation(drone):
    """Rotate to the configured absolute survey yaw before observations begin."""
    projectairsim_log().info(
        "Correcting drone orientation to survey yaw %.1f deg",
        SURVEY_YAW_DEG,
    )

    rotate_task = await drone.rotate_to_yaw_async(
        yaw=SURVEY_YAW_RAD,
        timeout_sec=YAW_CORRECTION_TIMEOUT_SEC,
        margin=float(np.deg2rad(YAW_CORRECTION_MARGIN_DEG)),
    )
    await rotate_task




async def move_to_waypoint_precisely(drone, waypoint):
    """
    Fly to a waypoint using a short closed-loop control pulse every
    AUTONOMOUS_CONTROL_INTERVAL_SEC.

    """

    loop = asyncio.get_running_loop()
    last_failure = None

    for attempt in range(1, MAX_WAYPOINT_ATTEMPTS + 1):
        # Recalculate the timeout from the actual current position on each retry.
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
            "Moving to %s with %.2f s closed-loop control pulses at %.1f m/s | "
            "waypoint NED=(%.3f, %.3f, %.3f) | distance=%.1f m | "
            "target yaw=%.1f deg | attempt=%d/%d",
            waypoint["id"],
            AUTONOMOUS_CONTROL_INTERVAL_SEC,
            FLIGHT_VELOCITY_MPS,
            waypoint["x"],
            waypoint["y"],
            waypoint["z"],
            distance,
            SURVEY_YAW_DEG,
            attempt,
            MAX_WAYPOINT_ATTEMPTS,
        )

        attempt_started = loop.time()
        monitor_started = attempt_started
        stall_started = None
        last_yaw_warning = attempt_started - YAW_WARNING_LOG_INTERVAL_SEC

        previous_x = start_x
        previous_y = start_y
        previous_z = start_z
        previous_time = attempt_started

        failure_reason = None

        while True:

            # Re-read the REAL position and orientation before every command pulse.
            pose = drone.get_ground_truth_pose()
            position = pose["translation"]

            current_x = float(position["x"])
            current_y = float(position["y"])
            current_z = float(position["z"])
            current_yaw_rad = get_pose_yaw_rad(pose)

            current_time = loop.time()

            dx = waypoint["x"] - current_x
            dy = waypoint["y"] - current_y
            dz = waypoint["z"] - current_z

            horizontal_error = (dx**2 + dy**2) ** 0.5
            vertical_error = abs(dz)

            #orientation correction
            yaw_error_rad = wrap_angle_rad(SURVEY_YAW_RAD - current_yaw_rad)
            yaw_error_deg = float(np.rad2deg(yaw_error_rad))

            at_waypoint = (
                horizontal_error <= POSITION_TOLERANCE_M
                and vertical_error <= POSITION_TOLERANCE_M
            )

            if at_waypoint:

                stop_task = await drone.move_by_velocity_z_async(
                    v_north=0.0,
                    v_east=0.0,
                    z=waypoint["z"],
                    duration=AUTONOMOUS_CONTROL_INTERVAL_SEC,
                    yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
                    yaw_is_rate=False,
                    yaw=SURVEY_YAW_RAD,
                )#stay at point and continue to correct yaw.
                await stop_task

                final_pose = drone.get_ground_truth_pose()
                final_yaw_rad = get_pose_yaw_rad(final_pose)
                final_yaw_error_deg = float(
                    np.rad2deg(wrap_angle_rad(SURVEY_YAW_RAD - final_yaw_rad))
                )

                projectairsim_log().info(
                    "Reached %s | actual=(%.3f, %.3f, %.3f) | "
                    "horizontal error=%.2f m vertical error=%.2f m | "
                    "yaw error=%+.2f deg",
                    waypoint["id"],
                    current_x,
                    current_y,
                    current_z,
                    horizontal_error,
                    vertical_error,
                    final_yaw_error_deg,
                )
                return current_x, current_y, current_z

            elapsed = current_time - attempt_started
            if elapsed >= timeout_sec:
                failure_reason = (
                    f"Timed out moving to waypoint {waypoint['id']} after "
                    f"{elapsed:.2f} s"
                )
                projectairsim_log().error(failure_reason)
                break


            dt = max(current_time - previous_time, 1e-6)
            travelled = (
                (current_x - previous_x) ** 2
                + (current_y - previous_y) ** 2
                + (current_z - previous_z) ** 2
            ) ** 0.5
            measured_speed = travelled / dt

            past_startup_grace = (
                current_time - monitor_started >= STALL_STARTUP_GRACE_SEC
            )

            if (
                past_startup_grace
                and measured_speed < STALL_SPEED_THRESHOLD_MPS
            ):
                if stall_started is None:
                    stall_started = current_time
                elif current_time - stall_started >= STALL_DURATION_SEC:
                    failure_reason = (
                        f"Drone stalled while moving to waypoint {waypoint['id']}"
                    )
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
                    break
            else:
                stall_started = None

            # ADDED: CONTINUOUS ORIENTATION CORRECTION
            # Log sustained heading error at most once per second. The correction
            # itself is still reissued every control pulse, including when the
            # error is inside the configured margin.
            if (
                abs(yaw_error_deg) > YAW_CORRECTION_MARGIN_DEG
                and current_time - last_yaw_warning
                >= YAW_WARNING_LOG_INTERVAL_SEC
            ):
                projectairsim_log().warning(
                    "Yaw correction active | target=%.2f deg | "
                    "actual=%.2f deg | error=%+.2f deg",
                    SURVEY_YAW_DEG,
                    float(np.rad2deg(current_yaw_rad)),
                    yaw_error_deg,
                )
                last_yaw_warning = current_time


            if horizontal_error > 1e-9:
                commanded_speed = min(
                    FLIGHT_VELOCITY_MPS,
                    max(0.5, horizontal_error),
                )
                v_north = commanded_speed * dx / horizontal_error
                v_east = commanded_speed * dy / horizontal_error #alter x/y velocity components based on proximity.
            else:
                v_north = 0.0
                v_east = 0.0

            control_task = await drone.move_by_velocity_z_async(
                v_north=v_north,
                v_east=v_east,
                z=waypoint["z"],
                duration=AUTONOMOUS_CONTROL_INTERVAL_SEC,
                yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
                yaw_is_rate=False,
                yaw=SURVEY_YAW_RAD,
            )
            await control_task

            previous_x = current_x
            previous_y = current_y
            previous_z = current_z
            previous_time = current_time

        last_failure = RuntimeError(
            failure_reason
            or f"Failed to reach waypoint {waypoint['id']}"
        )

        if attempt < MAX_WAYPOINT_ATTEMPTS:
            await asyncio.sleep(WAYPOINT_SETTLE_SEC)
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

        yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
        yaw_is_rate=False,
        yaw=SURVEY_YAW_RAD,
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

            manual_survey_start_time = asyncio.get_running_loop().time()


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

                #finish timing before landing/cleanup ===
                manual_survey_time_sec = max(
                    0.0,
                    asyncio.get_running_loop().time() - manual_survey_start_time,
                )


                print()
                print_observation_summary(
                    observation_status,
                    manual_survey_time_sec,  
                )

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

        await correct_survey_orientation(drone)

        # The autonomous survey cameras start ONLY after the drone
        # has reached and settled at the first sweep waypoint.
        projectairsim_log().info(
            "Reached survey start %s - starting cameras",
            first_waypoint["id"],
        )


        # Start survey only after the first sweep waypoint is reached. Therefore
        # takeoff and transit to the survey start are excluded.
        survey_start_time = asyncio.get_running_loop().time()


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
            survey_end_time = asyncio.get_running_loop().time()
            survey_time_sec = max(
                0.0,
                survey_end_time - survey_start_time,
            )

            # Stop the observation task.
            telemetry_task.cancel()
            try:
                await telemetry_task
            except asyncio.CancelledError:
                pass
            except Exception as err:
                projectairsim_log().warning(
                    "Telemetry task ended with error during cleanup: %s",
                    err,
                )

            # Results no longer depend on the debug camera, so save them NOW.
            print()
            print_observation_summary(
                observation_status,
                survey_time_sec,
            )

            export_survey_results(
                targets,
                observation_status,
            )

            # The debug viewer is non-essential. Its failure must never prevent
            # experimental results from being saved.
            segmentation_stop.set()
            try:
                await segmentation_task
            except asyncio.CancelledError:
                pass
            except Exception as err:
                projectairsim_log().warning(
                    "Debug camera ended with error during cleanup: %s",
                    err,
                )

            projectairsim_log().info("Survey cameras stopped")

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
