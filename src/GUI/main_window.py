"""Main PySide6 window for the Rivelero GUI.

The user selects one or more component modules, then presses one submit button.
A single background worker:

1. crops the currently visible DEM once;
2. runs each selected component module in sequence;
3. passes the returned component results into the OPF runner; and
4. returns both the component results and the combined OPF to the GUI.

The component fields and the observability potential field are displayed in tabs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
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


ModuleRunner = Callable[..., object]
SelectedModules = dict[
    str,
    tuple[ModuleRunner, dict[str, Any]],
]


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """All results returned by one submitted Rivelero pipeline."""

    component_results: dict[str, object]
    opf_result: object
    transform: Any
    crs: Any


def run_selected_pipeline_on_visible_dem_crop(
    *,
    source_dem_path: str | Path,
    visible_bounds: tuple[float, float, float, float],
    selected_modules: SelectedModules,
    opf_runner: ModuleRunner,
) -> PipelineOutput:
    """Run the selected modules and OPF on one shared temporary DEM crop.

    This function is executed by one ``FunctionWorker``. Each component runner
    is called synchronously inside that worker, so the next runner starts only
    after the previous runner has returned. The GUI thread remains responsive.
    """

    if not selected_modules:
        raise ValueError("At least one component module must be selected.")

    with TemporaryDirectory(prefix="rivelero-dem-crop-") as directory:
        crop = write_raster_crop(
            source_path=source_dem_path,
            bounds=visible_bounds,
            output_path=Path(directory) / "visible_dem.tif",
        )

        component_results: dict[str, object] = {}

        for module_name, (runner, runner_kwargs) in selected_modules.items():
            arguments = dict(runner_kwargs)
            arguments["dem_path"] = crop.path
            component_results[module_name] = runner(**arguments)

        opf_result = opf_runner(
            visibility_result=component_results.get("visibility"),
            botanical_result=component_results.get("botanical"),
            occlusion_result=component_results.get("occlusion"),
        )

        return PipelineOutput(
            component_results=component_results,
            opf_result=opf_result,
            transform=crop.transform,
            crs=crop.crs,
        )



class MainWindow(QMainWindow):
    """Rivelero window containing the DEM and component-result views."""

    def __init__(
        self,
        visibility_runner: ModuleRunner,
        botanical_runner: ModuleRunner,
        obstacle_runner: ModuleRunner,
        opf_runner: ModuleRunner | None = None,
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
        self.opf_runner = opf_runner

        self.target_geometry = target_geometry
        self.observer_geometry = observer_geometry
        self.time_from = time_from
        self.time_to = time_to

        self.dem_path: Path | None = None
        self.dem_transform: Any | None = None
        self.dem_crs: Any | None = None

        # Exact, full-resolution backend results from the latest submission.
        self.visibility_result: object | None = None
        self.botanical_result: object | None = None
        self.occlusion_result: object | None = None
        self.opf_result: object | None = None

        self.thread_pool = QThreadPool.globalInstance()
        self._workers: set[FunctionWorker] = set()

        self.setWindowTitle("Rivelero")
        self.resize(1250, 780)
        self.setMinimumSize(900, 600)

        self._build_menu()
        self._build_interface()
        self._apply_styles()
        self._update_controls()

    def _build_menu(self) -> None:
        """Create the File menu."""

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
        """Construct the main window widgets and layouts."""

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

        geometric_group = QGroupBox("Geometric Visibility")
        geometric_layout = QVBoxLayout(geometric_group)

        self.los_checkbox = QCheckBox("Line of sight")
        self.obstacle_checkbox = QCheckBox("Obstacle occlusion")
        geometric_layout.addWidget(self.los_checkbox)
        geometric_layout.addWidget(self.obstacle_checkbox)

        botanical_group = QGroupBox("Botanical Suitability")
        botanical_layout = QVBoxLayout(botanical_group)

        self.ndvi_checkbox = QCheckBox("NDVI")
        botanical_layout.addWidget(self.ndvi_checkbox)

        self.los_checkbox.setChecked(True)
        self.ndvi_checkbox.setChecked(True)
        self.obstacle_checkbox.setChecked(True)

        module_layout.addWidget(geometric_group)
        module_layout.addWidget(botanical_group)

        self.run_selected_button = QPushButton("Run selected modules")
        self.run_selected_button.clicked.connect(self.run_selected_modules)
        module_layout.addWidget(self.run_selected_button)

        self.los_checkbox.toggled.connect(self._update_controls)
        self.ndvi_checkbox.toggled.connect(self._update_controls)
        self.obstacle_checkbox.toggled.connect(self._update_controls)

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
        self.dem_view.area_selected.connect(self._receive_selected_area)
        self.dem_view.area_selection_cleared.connect(
            self._on_area_selection_cleared
        )
        self.dem_view.area_selection_error.connect(
            self._show_area_selection_error
        )

        self.los_view = RasterView("Run the LoS module to show a result.")
        self.ndvi_view = RasterView("Run the NDVI module to show a result.")
        self.obstacle_view = RasterView(
            "Run the obstacle module to show a result."
        )
        self.opf_view = RasterView(
            "Run selected modules to show the observability potential field."
        )

        self.tabs.addTab(self.dem_view, "DEM")
        workspace_layout.addWidget(self.tabs)

        outer_layout.addWidget(self.left_panel)
        outer_layout.addWidget(self.workspace, 1)

    def _on_area_selection_started(self) -> None:
        """Discard the previous target before its replacement is drawn."""

        self.target_geometry = None
        self.tabs.setCurrentWidget(self.dem_view)
        self.status_label.setText(
            "Previous target cleared. Click points on the DEM to draw its "
            "replacement, then click the first point again to finish."
        )
        self._update_controls()

    def _on_area_selection_cleared(self) -> None:
        """Clear the stored target geometry and its visible patch."""

        self.target_geometry = None
        self.status_label.setText("Target region cleared.")
        self._update_controls()

    def _show_area_selection_error(self, message: str) -> None:
        """Display an error raised while activating polygon selection."""

        QMessageBox.warning(
            self,
            "Area selection unavailable",
            message,
        )

    def _receive_selected_area(
        self,
        vertices: list[tuple[float, float]],
    ) -> None:
        """Store the selected polygon as a GeoJSON-like geometry."""

        self.target_geometry = {
            "type": "Polygon",
            "coordinates": [
                [[x, y] for x, y in vertices]
            ],
        }

        self.status_label.setText(
            f"Target region selected with {len(vertices) - 1} vertices."
        )
        self._update_controls()

    def choose_dem(self) -> None:
        """Ask the user for a local GeoTIFF and display its first band."""

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
            self._show_error("DEM loading failed", str(error))
            return

        self.dem_path = path
        self.dem_transform = self.dem_view.source_transform
        self.dem_crs = self.dem_view.source_crs

        self.visibility_result = None
        self.botanical_result = None
        self.occlusion_result = None
        self.opf_result = None

        self.opf_view.show_message(
            "Run selected modules to show the observability potential field."
        )

        self.dem_path_label.setText(str(path))
        self.dem_path_label.setToolTip(str(path))
        self.tabs.setCurrentWidget(self.dem_view)
        self.status_label.setText("DEM loaded. Choose modules to run.")
        self._update_controls()

    def run_selected_modules(self) -> None:
        """Launch one worker for the selected components and the OPF."""

        if self.dem_path is None:
            self._show_missing_dem()
            return

        run_los = self.los_checkbox.isChecked()
        run_ndvi = self.ndvi_checkbox.isChecked()
        run_obstacle = self.obstacle_checkbox.isChecked()

        if not any((run_los, run_ndvi, run_obstacle)):
            QMessageBox.warning(
                self,
                "No modules selected",
                "Tick at least one module before submitting.",
            )
            return

        if self.opf_runner is None:
            QMessageBox.warning(
                self,
                "OPF module unavailable",
                "No observability-potential-field runner was supplied.",
            )
            return

        missing_requirements: list[str] = []

        if run_los and self.target_geometry is None:
            missing_requirements.append(
                "Line of sight requires a target region."
            )

        if run_ndvi and not (self.time_from and self.time_to):
            missing_requirements.append(
                "NDVI requires Sentinel-2 start and end dates."
            )

        if missing_requirements:
            QMessageBox.warning(
                self,
                "Missing inputs",
                "\n".join(missing_requirements),
            )
            return

        selected_modules: SelectedModules = {}
        selected_names: list[str] = []

        if run_los:
            selected_modules["visibility"] = (
                self.visibility_runner,
                {
                    "target_geometry": self.target_geometry,
                    "observer_geometry": self.observer_geometry,
                },
            )
            selected_names.append("LOS")
            self.visibility_result = None

        if run_ndvi:
            selected_modules["botanical"] = (
                self.botanical_runner,
                {
                    "target_geometry": self.target_geometry,
                    "time_from": self.time_from,
                    "time_to": self.time_to,
                },
            )
            selected_names.append("NDVI")
            self.botanical_result = None

        if run_obstacle:
            selected_modules["occlusion"] = (
                self.obstacle_runner,
                {
                    "field_geometry": self.observer_geometry,
                },
            )
            selected_names.append("obstacles")
            self.occlusion_result = None

        self.opf_result = None

        try:
            visible_bounds = self.dem_view.current_visible_bounds()
        except RuntimeError as error:
            self._show_error("DEM crop unavailable", str(error))
            return

        worker = FunctionWorker(
            run_selected_pipeline_on_visible_dem_crop,
            source_dem_path=self.dem_path,
            visible_bounds=visible_bounds,
            selected_modules=selected_modules,
            opf_runner=self.opf_runner,
        )
        self._workers.add(worker)

        worker.signals.result.connect(self._receive_pipeline_result)
        worker.signals.error.connect(self._receive_worker_error)
        worker.signals.finished.connect(
            lambda: self._finish_worker(worker)
        )

        self.status_label.setText(
            "Running " + ", ".join(selected_names) + " and building the OPF…"
        )
        self.progress_bar.setRange(0, 0)
        self._update_controls()
        self.thread_pool.start(worker)

    def _receive_pipeline_result(self, output: PipelineOutput) -> None:
        """Store and display all results from a completed pipeline."""

        results = output.component_results

        if "visibility" in results:
            self._receive_module_result(
                name="Visibility",
                destination=self.los_view,
                raw_result=results["visibility"],
                default_title="3D visibility field",
                default_colour_map="viridis",
                default_colourbar_label="Visibility field height",
                result_attribute="visibility_result",
                display_mode="surface",
                surface_colour_map="Wistia",
                fallback_transform=output.transform,
                fallback_crs=output.crs,
            )

        if "botanical" in results:
            self._receive_module_result(
                name="Botanical suitability",
                destination=self.ndvi_view,
                raw_result=results["botanical"],
                default_title="Botanical suitability field",
                default_colour_map="YlGn",
                default_colourbar_label=(
                    "Botanical suitability field height"
                ),
                result_attribute="botanical_result",
                display_mode="surface",
                surface_colour_map=None,
                fallback_transform=output.transform,
                fallback_crs=output.crs,
            )

        if "occlusion" in results:
            self._receive_module_result(
                name="Obstacle occlusion",
                destination=self.obstacle_view,
                raw_result=results["occlusion"],
                default_title="Obstacle occlusion field",
                default_colour_map="magma",
                default_colourbar_label="Local obstacle coverage",
                result_attribute="occlusion_result",
                display_mode="surface",
                surface_colour_map=None,
                fallback_transform=output.transform,
                fallback_crs=output.crs,
            )

        # Component display marks a previous OPF stale, so store/show the newly
        # calculated OPF only after every component has been processed.
        self._receive_opf_result(output.opf_result)

    def _receive_module_result(
        self,
        *,
        name: str,
        destination: RasterView,
        raw_result: object,
        default_title: str,
        default_colour_map: str,
        default_colourbar_label: str,
        result_attribute: str,
        display_mode: str,
        surface_colour_map: str | None,
        fallback_transform: Any | None = None,
        fallback_crs: Any | None = None,
    ) -> None:
        """Store one component result and display it in its tab."""

        try:
            if not hasattr(raw_result, "field"):
                raise TypeError(
                    f"{name} did not return a result containing a field."
                )

            setattr(self, result_attribute, raw_result)

            # Until the new pipeline OPF is stored, any previous OPF is stale.
            self.opf_result = None

            display_result = coerce_raster_result(
                raw_result,
                default_title=default_title,
                default_colour_map=default_colour_map,
                default_colourbar_label=default_colourbar_label,
                fallback_transform=(
                    fallback_transform
                    if fallback_transform is not None
                    else self.dem_transform
                ),
                fallback_crs=(
                    fallback_crs
                    if fallback_crs is not None
                    else self.dem_crs
                ),
            )

            if display_mode == "surface":
                destination.show_surface(
                    display_result,
                    colour_map=surface_colour_map,
                )
            else:
                destination.show_result(display_result)

            if self.tabs.indexOf(destination) == -1:
                self.tabs.addTab(destination, name)

        except Exception as error:  # noqa: BLE001 - display conversion failures
            self._show_error(
                f"{name} result could not be displayed",
                str(error),
            )
            self.status_label.setText(
                f"{name} finished, but display failed."
            )
            return

        self.tabs.setCurrentWidget(destination)
        self.status_label.setText(f"{name} complete.")

    def _receive_opf_result(self, raw_result: object) -> None:
        """Store the combined result and display it in the OPF tab."""

        try:
            if not hasattr(raw_result, "field"):
                raise TypeError(
                    "The OPF runner did not return a result containing a field."
                )

            self.opf_result = raw_result

            display_result = coerce_raster_result(
                raw_result,
                default_title="Observability potential field",
                default_colour_map="viridis",
                default_colourbar_label="Observability potential",
                fallback_transform=self.dem_transform,
                fallback_crs=self.dem_crs,
            )

            self.opf_view.show_surface(display_result)

            if self.tabs.indexOf(self.opf_view) == -1:
                self.tabs.addTab(self.opf_view, "OPF")

        except Exception as error:  # noqa: BLE001 - show GUI conversion errors
            self._show_error(
                "Observability potential field could not be displayed",
                str(error),
            )
            self.status_label.setText(
                "OPF finished, but its display failed."
            )
            return

        self.tabs.setCurrentWidget(self.opf_view)
        self.status_label.setText(
            "Observability potential field complete."
        )

    def _receive_worker_error(
        self,
        message: str,
        traceback_text: str,
    ) -> None:
        """Show an exception raised by the background pipeline worker."""

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Pipeline failed")
        box.setText(message or "The pipeline failed.")
        box.setDetailedText(traceback_text)
        box.exec()

        self.status_label.setText("Pipeline failed.")

    def _finish_worker(self, worker: FunctionWorker) -> None:
        """Release a completed worker and restore the idle controls."""

        self._workers.discard(worker)

        if not self._workers:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

        self._update_controls()

    def _update_controls(self, _checked: bool | None = None) -> None:
        """Enable or disable controls according to the current GUI state."""

        is_busy = bool(self._workers)
        has_dem = self.dem_path is not None
        any_selected = any(
            (
                self.los_checkbox.isChecked(),
                self.ndvi_checkbox.isChecked(),
                self.obstacle_checkbox.isChecked(),
            )
        )

        self.load_dem_button.setEnabled(not is_busy)

        checkboxes_enabled = has_dem and not is_busy
        self.los_checkbox.setEnabled(checkboxes_enabled)
        self.ndvi_checkbox.setEnabled(checkboxes_enabled)
        self.obstacle_checkbox.setEnabled(checkboxes_enabled)

        self.run_selected_button.setEnabled(
            has_dem and any_selected and not is_busy
        )

        self.dem_view.set_area_selection_enabled(
            has_dem and not is_busy
        )

    def _show_missing_dem(self) -> None:
        """Tell the user that a DEM must be loaded first."""

        QMessageBox.warning(
            self,
            "No DEM selected",
            "Load a DEM before running the pipeline.",
        )

    def _show_error(self, title: str, message: str) -> None:
        """Display a critical-error message box."""

        QMessageBox.critical(self, title, message)

    def _apply_styles(self) -> None:
        """Apply the application's dark Qt stylesheet."""

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

            QCheckBox {
                spacing: 8px;
                padding: 4px 2px;
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