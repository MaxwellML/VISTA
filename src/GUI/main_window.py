"""
Main PySide6 window for the first Rivelero GUI version.

Expected backend functions
--------------------------
The two imports below assume these functions already exist:

    modules.los.run_los(dem_path)
    modules.ndvi.run_ndvi(dem_path)

Each function may return any format accepted by
``gui.raster_view.coerce_raster_result``. Returning ``RasterResult`` is the
cleanest option.

There are deliberately no candidate-viewpoint controls in this version.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)



from .raster_view import (
    DynamicRasterView,
    RasterView,
    coerce_raster_result,
    write_raster_crop,
)
from .worker import FunctionWorker


@dataclass(frozen=True, slots=True)
class CroppedModuleOutput:
    """Backend result plus the georeferencing of its temporary DEM crop."""

    value: object
    transform: Any
    crs: Any


def run_module_on_visible_dem_crop(
    runner,
    *,
    source_dem_path: str | Path,
    visible_bounds: tuple[float, float, float, float],
    runner_kwargs: dict[str, Any],
) -> CroppedModuleOutput:
    """Crop the current DEM view, then run one backend on that crop."""

    with TemporaryDirectory(prefix="rivelero-dem-crop-") as directory:
        crop = write_raster_crop(
            source_path=source_dem_path,
            bounds=visible_bounds,
            output_path=Path(directory) / "visible_dem.tif",
        )

        arguments = dict(runner_kwargs)
        arguments["dem_path"] = crop.path
        value = runner(**arguments)

        return CroppedModuleOutput(
            value=value,
            transform=crop.transform,
            crs=crop.crs,
        )


class MainWindow(QMainWindow):
    """Rivelero window containing DEM, LoS and NDVI views."""

    def __init__(
        self,
        visibility_runner,
        botanical_runner,
        obstacle_runner,
        *,
        target_geometry: object | None = None,
        observer_geometry: object | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
    ) -> None:
        super().__init__()

        self.visibility_runner = visibility_runner
        self.botanical_runner = botanical_runner
        self.obstacle_runner = obstacle_runner

        self.target_geometry = target_geometry
        self.observer_geometry = observer_geometry

        self.time_from = time_from
        self.time_to = time_to

        self.dem_path: Path | None = None
        self.dem_transform: Any | None = None
        self.dem_crs: Any | None = None

        self.thread_pool = QThreadPool.globalInstance() #store reference to Qt's threadpool to stop heavy computation from blocking the interface.
        self._workers: set[FunctionWorker] = set() #store current workers.

        self.setWindowTitle("Rivelero")
        self.resize(1250, 780)
        self.setMinimumSize(900, 600)

        self._build_menu() #create a file menu.
        self._build_interface()
        self._apply_styles()
        self._update_controls()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open DEM…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_dem)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _build_interface(self) -> None:
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        outer_layout = QHBoxLayout(central_widget)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(10)

        self.left_panel = QFrame()
        self.left_panel.setObjectName("leftPanel")
        self.left_panel.setMinimumWidth(270)
        self.left_panel.setMaximumWidth(360)
        self.left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)

        title = QLabel("RIVELERO")
        title.setObjectName("applicationTitle")
        left_layout.addWidget(title)

        subtitle = QLabel("Spatial Observability Engine")
        subtitle.setObjectName("applicationSubtitle")
        subtitle.setWordWrap(True)
        left_layout.addWidget(subtitle)

        input_group = QGroupBox("Survey area")
        input_layout = QVBoxLayout(input_group)

        self.load_dem_button = QPushButton("Load DEM")
        self.load_dem_button.clicked.connect(self.choose_dem)
        input_layout.addWidget(self.load_dem_button)

        self.dem_path_label = QLabel("No DEM selected")
        self.dem_path_label.setObjectName("pathLabel")
        self.dem_path_label.setWordWrap(True)
        self.dem_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        input_layout.addWidget(self.dem_path_label)

        left_layout.addWidget(input_group)

        module_group = QGroupBox("Modules")
        module_layout = QVBoxLayout(module_group)


        # Geometric Visibility sub-box
        geometric_group = QGroupBox("Geometric Visibility")
        geometric_layout = QVBoxLayout(geometric_group)

        self.run_los_button = QPushButton("Line of sight")
        self.run_los_button.clicked.connect(self.run_los)

        self.run_obstacle_button = QPushButton("Obstacle occlusion")
        self.run_obstacle_button.clicked.connect(self.run_obstacle)

        geometric_layout.addWidget(self.run_los_button)
        geometric_layout.addWidget(self.run_obstacle_button)

        # Botanical Suitability sub-box
        botanical_group = QGroupBox("Botanical Suitability")
        botanical_layout = QVBoxLayout(botanical_group)

        self.run_ndvi_button = QPushButton("NDVI")
        self.run_ndvi_button.clicked.connect(self.run_botanical)

        botanical_layout.addWidget(self.run_ndvi_button)


        # Put both sub-boxes inside the outer Modules box
        module_layout.addWidget(geometric_group)
        module_layout.addWidget(botanical_group)

        left_layout.addWidget(module_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Load a DEM to begin.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        left_layout.addStretch(1)

        self.workspace = QFrame()
        self.workspace.setObjectName("workspace")

        workspace_layout = QVBoxLayout(self.workspace)
        workspace_layout.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.dem_view = DynamicRasterView("DEM preview will show here.")
        self.dem_view.area_selection_started.connect(
            self._on_area_selection_started
        )
        self.dem_view.area_selected.connect(
            self._receive_selected_area
        )
        self.dem_view.area_selection_cleared.connect(
            self._on_area_selection_cleared
        )
        self.dem_view.area_selection_error.connect(
            self._show_area_selection_error
        )

        self.los_view = RasterView("Run the LoS module to show a result.")
        self.ndvi_view = RasterView("Run the NDVI module to show a result.")
        self.obstacle_view = RasterView("Run the obstacle module to show a result.")

        self.tabs.addTab(self.dem_view, "DEM")

        workspace_layout.addWidget(self.tabs)

        outer_layout.addWidget(self.left_panel)
        outer_layout.addWidget(self.workspace, 1)

    def _on_area_selection_started(self) -> None:
        """Discard the old target before the replacement is drawn."""

        self.target_geometry = None
        self.tabs.setCurrentWidget(self.dem_view)
        self.status_label.setText(
            "Previous target cleared. Click points on the DEM to draw its "
            "replacement, then click the first point again to finish."
        )
        self._update_controls()

    def _on_area_selection_cleared(self) -> None:
        """Clear the stored target geometry as well as its visible patch."""

        self.target_geometry = None
        self.status_label.setText("Target region cleared.")
        self._update_controls()

    def _show_area_selection_error(self, message: str) -> None:
        """Show an error raised while activating polygon selection."""

        QMessageBox.warning(
            self,
            "Area selection unavailable",
            message,
        )

    def _receive_selected_area(
        self,
        vertices: list[tuple[float, float]],
    ) -> None:
        """Store the selected polygon as GeoJSON-like geometry."""

        self.target_geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [x, y]
                    for x, y in vertices
                ]
            ],
        }

        self.status_label.setText(
            f"Target region selected with {len(vertices) - 1} vertices."
        )

        self._update_controls()

    def choose_dem(self) -> None:
        """Ask the user for a local GeoTIFF and show its first band."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open DEM",
            "",
            "GeoTIFF files (*.tif *.tiff);;All files (*)",
        )

        if not filename:
            return

        path = Path(filename)

        self.dem_view.clear_selected_area()
        self.target_geometry = None

        try:
            self.dem_view.open_raster(
                path,
                title="DEM preview",
                colour_map="terrain",
                colourbar_label="Elevation",
            )
        except Exception as error:  # noqa: BLE001 - present file errors in GUI
            self._show_error(
                "DEM loading failed",
                str(error),
            )
            return

        self.dem_path = path
        self.dem_transform = self.dem_view.source_transform
        self.dem_crs = self.dem_view.source_crs
 

        self.dem_path_label.setText(str(path))
        self.dem_path_label.setToolTip(str(path))
        self.tabs.setCurrentWidget(self.dem_view)
        self.status_label.setText("DEM loaded. Choose a module to run.")
        self._update_controls()

    def run_los(self) -> None:
        """Build the terrain-visibility field."""

        if self.dem_path is None:
            self._show_missing_dem()
            return

        if self.target_geometry is None:
            QMessageBox.warning(
                self,
                "No target region",
                "Load or define a target region before running visibility.",
            )
            return

        self._start_module(
            name="Visibility",
            runner=self.visibility_runner,
            destination=self.los_view,
            runner_kwargs={
                "target_geometry": self.target_geometry,
                "observer_geometry": self.observer_geometry,
            },
            default_title="3D visibility field",
            default_colour_map="viridis",
            default_colourbar_label="Visibility field height",
            display_mode="surface",
            surface_colour_map="Wistia",
        )

    def run_botanical(self) -> None:
        """Build the botanical-suitability field."""

        if self.dem_path is None:
            self._show_missing_dem()
            return

        if not self.time_from or not self.time_to:
            QMessageBox.warning(
                self,
                "No imagery dates",
                "Select a Sentinel-2 start and end date.",
            )
            return

        self._start_module(
            name="Botanical suitability",
            runner=self.botanical_runner,
            destination=self.ndvi_view,
            runner_kwargs={
                "target_geometry": self.target_geometry,
                "time_from": self.time_from,
                "time_to": self.time_to,
            },
            default_title="Botanical suitability field",
            default_colour_map="YlGn",
            default_colourbar_label="Botanical suitability field height",
        )

    def run_obstacle(self) -> None:
        """Build the OpenStreetMap obstacle-occlusion field."""

        if self.dem_path is None:
            self._show_missing_dem()
            return

        self._start_module(
            name="Obstacle occlusion",
            runner=self.obstacle_runner,
            destination=self.obstacle_view,
            runner_kwargs={
                "field_geometry": self.observer_geometry,
            },
            default_title="Obstacle occlusion field",
            default_colour_map="magma",
            default_colourbar_label="Local obstacle coverage",
        )

    def _start_module(
        self,
        *,
        name: str,
        runner,
        destination: RasterView,
        runner_kwargs: dict[str, Any],
        default_title: str,
        default_colour_map: str,
        default_colourbar_label: str,
        display_mode: str = "raster",
        surface_colour_map: str | None = None,
    ) -> None:
        if self.dem_path is None:
            self._show_missing_dem()
            return

        try:
            visible_bounds = self.dem_view.current_visible_bounds()
        except RuntimeError as error:
            self._show_error("DEM crop unavailable", str(error))
            return

        worker = FunctionWorker(
            run_module_on_visible_dem_crop,
            runner,
            source_dem_path=self.dem_path,
            visible_bounds=visible_bounds,
            runner_kwargs=runner_kwargs,
        )
        self._workers.add(worker) #add worker to list of current workers.

        worker.signals.result.connect(
            lambda raw_result: self._receive_module_result(
                name=name,
                destination=destination,
                raw_result=raw_result,
                default_title=default_title,
                default_colour_map=default_colour_map,
                default_colourbar_label=default_colourbar_label,
                display_mode=display_mode,
                surface_colour_map=surface_colour_map,
            )
        ) #when a successful result is available, return it.
        worker.signals.error.connect(self._receive_worker_error) #if there is an error, report it.
        worker.signals.finished.connect(
            lambda: self._finish_worker(worker)
        ) #when a worker is finished, cleanup.

        self.status_label.setText(f"Cropping visible DEM and running {name}…")
        self.progress_bar.setRange(0, 0) #display "busy" bar.
        self._update_controls()
        self.thread_pool.start(worker)

    def _receive_module_result(
        self,
        *,
        name: str,
        destination: RasterView,
        raw_result: object,
        default_title: str,
        default_colour_map: str,
        default_colourbar_label: str,
        display_mode: str,
        surface_colour_map: str | None,
    ) -> None:
        try:
            fallback_transform = self.dem_transform
            fallback_crs = self.dem_crs

            if isinstance(raw_result, CroppedModuleOutput):
                fallback_transform = raw_result.transform
                fallback_crs = raw_result.crs
                raw_result = raw_result.value

            result = coerce_raster_result(
                raw_result,
                default_title=default_title,
                default_colour_map=default_colour_map,
                default_colourbar_label=default_colourbar_label,
                fallback_transform=fallback_transform,
                fallback_crs=fallback_crs,
            )

            if display_mode == "surface":
                destination.show_surface(
                    result,
                    colour_map=surface_colour_map,
                )
            else:
                destination.show_result(result)

            if self.tabs.indexOf(destination) == -1:
                self.tabs.addTab(destination, name)

        except Exception as error:  # noqa: BLE001 - display conversion failures
            self._show_error(
                f"{name} result could not be displayed",
                str(error),
            )
            self.status_label.setText(f"{name} finished, but display failed.")
            return

        self.tabs.setCurrentWidget(destination)
        if display_mode == "surface":
            self.status_label.setText(
                f"{name} complete. Drag the 3D field to rotate it."
            )
        else:
            self.status_label.setText(f"{name} complete.")

    def _receive_worker_error(
        self,
        message: str,
        traceback_text: str,
    ) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Module failed")
        box.setText(message or "The module failed.")
        box.setDetailedText(traceback_text)
        box.exec()

        self.status_label.setText("Module failed.")

    def _finish_worker(self, worker: FunctionWorker) -> None:
        self._workers.discard(worker)

        if not self._workers:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

        self._update_controls()

    def _update_controls(self) -> None:
        is_busy = bool(self._workers)
        has_dem = self.dem_path is not None
        has_target = self.target_geometry is not None
        has_dates = bool(self.time_from and self.time_to)

        self.load_dem_button.setEnabled(not is_busy)

        self.run_los_button.setEnabled(
            has_dem and has_target and not is_busy
        )

        self.run_ndvi_button.setEnabled(
            has_dem and has_dates and not is_busy
        )

        self.run_obstacle_button.setEnabled(
            has_dem and not is_busy
        )

        self.dem_view.set_area_selection_enabled(
            has_dem and not is_busy
        )

    def _show_missing_dem(self) -> None:
        QMessageBox.warning(
            self,
            "No DEM selected",
            "Load a DEM before running a module.",
        )

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow,
            QWidget#centralWidget {
                background-color: #111111;
                color: #f2f2f2;
                font-family: "Segoe UI";
                font-size: 14px;
            }

            QFrame#leftPanel,
            QFrame#workspace {
                background-color: #181818;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
            }

            QLabel#applicationTitle {
                font-size: 22px;
                font-weight: 700;
            }

            QLabel#applicationSubtitle,
            QLabel#pathLabel,
            QLabel#statusLabel {
                color: #bdbdbd;
            }

            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }

            QPushButton {
                background-color: #292929;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px 10px;
                text-align: left;
            }

            QPushButton:hover {
                background-color: #343434;
            }

            QPushButton:pressed {
                background-color: #202020;
            }

            QPushButton:disabled {
                color: #707070;
                background-color: #1d1d1d;
                border-color: #333333;
            }

            QTabWidget::pane {
                border: 1px solid #3d3d3d;
            }

            QTabBar::tab {
                background-color: #202020;
                border: 1px solid #3d3d3d;
                padding: 8px 18px;
            }

            QTabBar::tab:selected {
                background-color: #343434;
            }

            QProgressBar {
                background-color: #202020;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                min-height: 7px;
                max-height: 7px;
            }

            QProgressBar::chunk {
                background-color: #8e8e8e;
            }

            QMenuBar,
            QMenu {
                background-color: #181818;
                color: #f2f2f2;
            }

            QMenu::item:selected {
                background-color: #343434;
            }
            """
        )