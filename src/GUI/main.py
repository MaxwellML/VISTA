# GUI/__main__.py

import sys

from PySide6.QtWidgets import QApplication
from datetime import datetime, timezone
from gui.main_window import MainWindow
from modules.botanical_suitability_field import (
    build_botanical_suitability_field,
)
from modules.obstacle_occlusion_field import (
    build_obstacle_occlusion_field,
)
from modules.visibility_field import build_visibility_field


def main() -> int:
    app = QApplication(sys.argv)

    # These must eventually come from GUI controls or loaded project data.
    target_geometry = load_target_geometry()
    observer_geometry = None
    
    time_to = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    time_from = (now - relativedelta(months=1)).isoformat(timespec="seconds").replace("+00:00", "Z")

    window = MainWindow(
        visibility_runner=build_visibility_field,
        botanical_runner=build_botanical_suitability_field,
        obstacle_runner=build_obstacle_occlusion_field,
        target_geometry=target_geometry,
        observer_geometry=observer_geometry,
        time_from=time_from,
        time_to=time_to,
    )

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())