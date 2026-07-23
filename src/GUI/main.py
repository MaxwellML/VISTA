"""Application entry point for the Rivelero GUI."""

import sys
from datetime import datetime, timezone
import argparse
from dateutil.relativedelta import relativedelta
from PySide6.QtWidgets import QApplication

from SOE.botanical_suitability_field import (
    build_botanical_suitability_field,
)
from SOE.obstacle_occlusion_field import (
    build_obstacle_occlusion_field,
)
from SOE.observability_potential_field import (
    build_observability_potential_field,
)
from SOE.visibility_field import build_visibility_field

from .main_window import MainWindow

from SOE.viewpoint import Viewpoint, build_viewpoint_opf

from admin.sysinfo.process_monitor import (
    start_process_monitor,
    stop_process_monitor,
)



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Rivelero.")

    parser.add_argument(
        "--sysinfo",
        action="store_true",
        help="Enable system-information monitoring.",
    )

    return parser.parse_args()

def main() -> int:
    """Create and run the Rivelero Qt application."""
    args = parse_arguments()
    app = QApplication(sys.argv)
    if args.sysinfo:
        start_process_monitor(interval_seconds=5.0)
        app.aboutToQuit.connect(stop_process_monitor)

    # These will eventually come from GUI controls or project data.
    target_geometry = None
    observer_geometry = None

    now = datetime.now(timezone.utc)

    time_to = now.isoformat(
        timespec="seconds",
    ).replace("+00:00", "Z")

    time_from = (
        now - relativedelta(months=1)
    ).isoformat(
        timespec="seconds",
    ).replace("+00:00", "Z")

    window = MainWindow(
        visibility_runner=build_visibility_field,
        botanical_runner=build_botanical_suitability_field,
        obstacle_runner=build_obstacle_occlusion_field,
        opf_runner=build_observability_potential_field,
        target_geometry=target_geometry,
        observer_geometry=observer_geometry,
        time_from=time_from,
        time_to=time_to,
    )

    window.show()
    return app.exec() #begin the event loop.


if __name__ == "__main__":
    raise SystemExit(main())