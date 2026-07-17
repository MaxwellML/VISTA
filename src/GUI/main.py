# GUI/main.py

import sys
from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow

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


def main() -> int:
    app = QApplication(sys.argv)

    # These will eventually come from GUI controls or project data.
    target_geometry = None
    observer_geometry = None

    now = datetime.now(timezone.utc)

    time_to = now.isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")

    time_from = (
        now - relativedelta(months=1)
    ).isoformat(
        timespec="seconds"
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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())