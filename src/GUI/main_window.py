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

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

import rasterio


from pyproj import Transformer
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
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

from SOE.EXR_to_NDVI import load_ndvi_exr

from SOE.downsample_to_resolution import downsample_to_resolution


from SOE.redundancy_correction import correct_redundancy


from SOE.viewpoint import (
    ViewpointRegion,
    ViewpointOPFResult,
    build_viewpoint_opf,
)

from SOE.viewpoint_generator import (
    ViewpointGenerationResult,
    generate_viewpoints_from_opf,
)

UNREAL_LOCAL_COORDINATE_MODE = "unreal_local"



ModuleRunner = Callable[..., object]
SelectedModules = dict[
    str,
    tuple[ModuleRunner, dict[str, Any]],
]


def load_ndvi_exr_preview(
    *,
    path: str | Path,
    capture_x_cm: float,
    capture_y_cm: float,
    ortho_width_cm: float,
    encoded: bool,
) -> dict[str, Any]:
    """Load an EXR and return a mapping accepted by ``RasterView``."""
    result = load_ndvi_exr(
        path,
        capture_x_cm=capture_x_cm,
        capture_y_cm=capture_y_cm,
        ortho_width_cm=ortho_width_cm,
        encoded=encoded,
    )

    return {
        "data": result["ndvi"],
        "transform": result["transform"],
        "crs": result["crs"],
        "title": "NDVI EXR preview",
        "colour_map": "RdYlGn",
        "colourbar_label": "NDVI",
        "vmin": -1.0,
        "vmax": 1.0,
    }


@dataclass(frozen=True, slots=True)
class Viewpoint:
    """One original point or merged square in map/geographic coordinates."""

    identifier: int
    map_x: float
    map_y: float
    longitude: float
    latitude: float

    square_bounds: tuple[float, float, float, float] | None = None
    source_identifiers: tuple[int, ...] = ()

    @property
    def is_merged(self) -> bool:
        """Return whether this record represents a downsampled square."""
        return self.square_bounds is not None


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """All results returned by one submitted Rivelero pipeline."""

    component_results: dict[str, object]
    opf_result: object
    generated_viewpoints: list[ViewpointGenerationResult]
    transform: Any
    crs: Any


def run_selected_pipeline_on_visible_dem_crop(
    *,
    source_dem_path: str | Path,
    visible_bounds: tuple[float, float, float, float],
    selected_modules: SelectedModules,
    opf_runner: ModuleRunner,
    coordinate_mode: str | None = None,
    generate_viewpoints: bool = False,
    verification_radius_m: float = 30.0,
    suppression_radius_m: float = 50.0,
    max_generated_viewpoints: int = 10,
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
            coordinate_mode=coordinate_mode,
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

        generated_viewpoints: list[ViewpointGenerationResult] = []

        if generate_viewpoints:
            generated_viewpoints = generate_viewpoints_from_opf(
                opf_result,
                verification_radius_m=verification_radius_m,
                suppression_radius_m=suppression_radius_m,
                max_viewpoints=max_generated_viewpoints,
            )

        return PipelineOutput(
            component_results=component_results,
            opf_result=opf_result,
            generated_viewpoints=generated_viewpoints,
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
        fast_mode_resolution: float = 1000.0,
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
        self.ndvi_exr_path: Path | None = None  #CHANGED: Selected EXR source.

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

        # Slow mode retains exact input locations. Fast mode uses a derived
        # fixed-grid representation without mutating the original records.
        if fast_mode_resolution <= 0:
            raise ValueError("fast_mode_resolution must be greater than zero.")

        self.analysis_mode = "slow"
        self.fast_mode_resolution = float(fast_mode_resolution)
        self.original_viewpoints: list[Viewpoint] = []
        self.downsampled_viewpoints: list[Viewpoint] = []

        self._dem_to_wgs84: Transformer | None = None
        self._wgs84_to_dem: Transformer | None = None


        self._dem_is_unreal_local = False
        self._dem_tags: dict[str, str] = {}

        self.viewpoint_radius_m = 100.0
        self.viewpoint_results: dict[int, ViewpointOPFResult] = {}

        # Fast mode keeps the original cookie-cutter regions visible, but
        # records which regions fall outside the retained aggregate OPF score.
        self.redundancy_retention = 0.95
        self.discarded_viewpoint_identifiers: set[int] = set()

        self._build_menu()
        self._build_interface()
        self._apply_styles()
        self._update_controls()

    @property
    def active_viewpoints(self) -> list[Viewpoint]:
        """Return the viewpoint representation selected by the mode menu."""
        if self.analysis_mode == "fast":
            return self.downsampled_viewpoints

        return self.original_viewpoints


    def _stored_coordinates_from_map_point(
        self,
        map_x: float,
        map_y: float,
    ) -> tuple[float, float]:
        """Return values for the legacy longitude/latitude storage fields.

        For an Unreal-local DEM, the existing fields temporarily store X and
        Y respectively so the rest of the viewpoint pipeline can remain
        unchanged. The real analysis coordinates remain ``map_x``/``map_y``.
        """

        if self._dem_is_unreal_local:
            return map_x, map_y

        if self._dem_to_wgs84 is None:
            raise RuntimeError(
                "The DEM coordinate transformer has not been created."
            )

        return self._dem_to_wgs84.transform(map_x, map_y)

    def _viewpoint_table_coordinate_headers(self) -> tuple[str, str]:
        """Return the two coordinate-column labels for the active DEM."""

        if self._dem_is_unreal_local:
            # The table retains its existing Y-then-X value order.
            return "Y (m)", "X (m)"

        return "Latitude", "Longitude"

    def _format_map_position(self, map_x: float, map_y: float) -> str:
        """Format one viewpoint position for status messages."""

        if self._dem_is_unreal_local:
            return f"X={map_x:.3f} m, Y={map_y:.3f} m"

        longitude, latitude = self._stored_coordinates_from_map_point(
            map_x,
            map_y,
        )
        return f"{latitude:.6f}, {longitude:.6f}"


    def _build_menu(self) -> None:
        """Create the File and Mode menus."""

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

        mode_menu = self.menuBar().addMenu("&Mode")
        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.setExclusive(True)

        self.slow_mode_action = QAction("Slow mode", self)
        self.slow_mode_action.setCheckable(True)
        self.slow_mode_action.setChecked(True)
        self.slow_mode_action.triggered.connect(
            lambda checked: checked and self._set_analysis_mode("slow")
        )

        self.fast_mode_action = QAction("Fast mode", self)
        self.fast_mode_action.setCheckable(True)
        self.fast_mode_action.triggered.connect(
            lambda checked: checked and self._set_analysis_mode("fast")
        )

        self.mode_action_group.addAction(self.slow_mode_action)
        self.mode_action_group.addAction(self.fast_mode_action)
        mode_menu.addAction(self.slow_mode_action)
        mode_menu.addAction(self.fast_mode_action)


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

        viewpoints_group = QGroupBox("Viewpoints")
        viewpoints_layout = QVBoxLayout(viewpoints_group)

        self.import_viewpoints_button = QPushButton(
            "Import coordinates from CSV…"
        )
        self.import_viewpoints_button.clicked.connect(
            self.import_viewpoints_csv
        )
        viewpoints_layout.addWidget(self.import_viewpoints_button)

        self.add_viewpoints_button = QPushButton(
            "Add coordinates on map"
        )
        self.add_viewpoints_button.setCheckable(True)
        self.add_viewpoints_button.toggled.connect(
            self._set_viewpoint_mode_from_panel
        )
        viewpoints_layout.addWidget(self.add_viewpoints_button)

        self.viewpoint_count_label = QLabel("0 original viewpoints")
        self.viewpoint_count_label.setObjectName("pathLabel")
        viewpoints_layout.addWidget(self.viewpoint_count_label)

        self.mode_warning_label = QLabel(
            "Warning: original viewpoint information will be lost."
        )
        self.mode_warning_label.setObjectName("fastModeWarning")
        self.mode_warning_label.setWordWrap(True)
        self.mode_warning_label.setVisible(False)
        viewpoints_layout.addWidget(self.mode_warning_label)

        self.viewpoint_table = QTableWidget(0, 3)


        second_header, third_header = (
            self._viewpoint_table_coordinate_headers()
        )
        self.viewpoint_table.setHorizontalHeaderLabels(
            ["#", second_header, third_header]
        )

        self.viewpoint_table.verticalHeader().setVisible(False)
        self.viewpoint_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.viewpoint_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.viewpoint_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.viewpoint_table.itemSelectionChanged.connect(
            self._on_viewpoint_table_selection_changed
        )
        self.viewpoint_table.setMaximumHeight(160)

        header = self.viewpoint_table.horizontalHeader()
        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        header.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )
        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.Stretch,
        )

        viewpoints_layout.addWidget(self.viewpoint_table)
        left_layout.addWidget(viewpoints_group)

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

        self.sentinel_ndvi_radio = QRadioButton("Sentinel-2 API")
        self.exr_ndvi_radio = QRadioButton("Unreal NDVI EXR")
        self.sentinel_ndvi_radio.setChecked(True)
        botanical_layout.addWidget(self.sentinel_ndvi_radio)
        botanical_layout.addWidget(self.exr_ndvi_radio)

        self.load_ndvi_exr_button = QPushButton("Choose EXR…")
        self.load_ndvi_exr_button.clicked.connect(self.choose_ndvi_exr)
        botanical_layout.addWidget(self.load_ndvi_exr_button)

        self.ndvi_exr_path_label = QLabel("No EXR selected")
        self.ndvi_exr_path_label.setObjectName("pathLabel")
        self.ndvi_exr_path_label.setWordWrap(True)
        self.ndvi_exr_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        botanical_layout.addWidget(self.ndvi_exr_path_label)

        exr_metadata_form = QFormLayout()
        self.exr_capture_x_spin = self._make_exr_coordinate_spinbox()
        self.exr_capture_y_spin = self._make_exr_coordinate_spinbox()
        self.exr_ortho_width_spin = QDoubleSpinBox()
        self.exr_ortho_width_spin.setRange(0.01, 1_000_000_000.0)
        self.exr_ortho_width_spin.setDecimals(3)
        self.exr_ortho_width_spin.setValue(100_000.0)
        self.exr_ortho_width_spin.setSuffix(" cm")
        exr_metadata_form.addRow("Capture X:", self.exr_capture_x_spin)
        exr_metadata_form.addRow("Capture Y:", self.exr_capture_y_spin)
        exr_metadata_form.addRow(
            "Ortho width:",
            self.exr_ortho_width_spin,
        )
        botanical_layout.addLayout(exr_metadata_form)

        self.exr_encoded_checkbox = QCheckBox(
            "Encoded as (NDVI + 1) / 2"
        )
        self.exr_encoded_checkbox.setChecked(True)
        botanical_layout.addWidget(self.exr_encoded_checkbox)

        redundancy_group = QGroupBox("Redundancy correction")
        redundancy_layout = QVBoxLayout(redundancy_group)

        self.redundancy_checkbox = QCheckBox(
            "Apply redundancy correction"
        )
        self.redundancy_checkbox.setChecked(False)

        redundancy_layout.addWidget(self.redundancy_checkbox)

        self.los_checkbox.setChecked(True)
        self.ndvi_checkbox.setChecked(False)
        self.obstacle_checkbox.setChecked(False)

        module_layout.addWidget(geometric_group)
        module_layout.addWidget(botanical_group)
        module_layout.addWidget(redundancy_group)

        self.run_selected_button = QPushButton("Run selected modules")
        self.run_selected_button.clicked.connect(self.run_selected_modules)
        module_layout.addWidget(self.run_selected_button)

        self.los_checkbox.toggled.connect(self._update_controls)
        self.ndvi_checkbox.toggled.connect(self._update_controls)
        self.sentinel_ndvi_radio.toggled.connect(self._update_controls)
        self.exr_ndvi_radio.toggled.connect(self._update_controls)
        self.obstacle_checkbox.toggled.connect(self._update_controls)
        self.redundancy_checkbox.toggled.connect(self._update_controls)

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

        self.dem_view.viewpoint_clicked.connect(
            self._receive_viewpoint_click
        )
        self.dem_view.viewpoint_selection_toggled.connect(
            self._sync_viewpoint_mode
        )
        self.dem_view.viewpoints_clear_requested.connect(
            self._clear_viewpoints
        )
        self.dem_view.viewpoint_selection_error.connect(
            self._show_viewpoint_selection_error
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

    def _set_viewpoint_mode_from_panel(
        self,
        enabled: bool,
    ) -> None:
        """Mirror the left-panel button onto the Matplotlib tool."""

        if enabled:
            self.tabs.setCurrentWidget(self.dem_view)

        self.dem_view.set_viewpoint_selection_active(enabled)

    def _sync_viewpoint_mode(
        self,
        enabled: bool,
    ) -> None:
        """Keep the panel button synchronised with the toolbar."""

        self.add_viewpoints_button.blockSignals(True)
        self.add_viewpoints_button.setChecked(enabled)
        self.add_viewpoints_button.setText(
            "Finish adding coordinates"
            if enabled
            else "Add coordinates on map"
        )
        self.add_viewpoints_button.blockSignals(False)

        if enabled:
            self.status_label.setText(
                "Click the DEM to add viewpoints. "
                "Use pan or zoom to leave placement mode."
            )

    def _show_viewpoint_selection_error(
        self,
        message: str,
    ) -> None:
        """Display an error raised while activating placement."""

        QMessageBox.warning(
            self,
            "Viewpoint selection unavailable",
            message,
        )

    def _set_analysis_mode(self, mode: str) -> None:
        """Switch the table and DEM preview between exact and merged inputs."""
        if mode not in {"slow", "fast"}:
            raise ValueError(f"Unsupported analysis mode: {mode!r}")

        if mode == self.analysis_mode:
            return

        self.analysis_mode = mode

        # Discard decisions only apply to fast-mode merged regions.
        if mode != "fast":
            self.discarded_viewpoint_identifiers.clear()

        if mode == "fast":
            self._rebuild_downsampled_viewpoints()

        self.mode_warning_label.setVisible(mode == "fast")
        self._refresh_viewpoint_preview()

        if self.opf_result is not None:
            self._build_viewpoint_results()

        if mode == "fast":
            self.status_label.setText(
                "Fast mode selected. Warning: original viewpoint "
                "information will be lost."
            )
        else:
            self.status_label.setText(
                "Slow mode selected. Original viewpoints are shown."
            )

    def _rebuild_downsampled_viewpoints(self) -> None:
        """Derive merged square records from the retained original points."""
        self.downsampled_viewpoints.clear()

        # --- REDUNDANCY CORRECTOR EDIT ---
        # The cookie-cutter IDs are regenerated here, so any previous discard
        # decisions are stale until the OPF cutouts are scored again.
        self.discarded_viewpoint_identifiers.clear()
        # --- END REDUNDANCY CORRECTOR EDIT ---

        if not self.original_viewpoints:
            return

        #an Unreal-local DEM deliberately has no WGS84
        # transformer, but its map coordinates are already usable directly.
        if (
            not self._dem_is_unreal_local
            and self._dem_to_wgs84 is None
        ):
            return


        points = np.asarray(
            [
                (viewpoint.map_x, viewpoint.map_y)
                for viewpoint in self.original_viewpoints
            ],
            dtype=float,
        )

        origin_x = 0.0
        origin_y = 0.0

        if self.dem_transform is not None:
            origin_x = float(self.dem_transform.c)
            origin_y = float(self.dem_transform.f)

        merged_regions = downsample_to_resolution(
            points,
            self.fast_mode_resolution,
            origin_x=origin_x,
            origin_y=origin_y,
        )

        for identifier, region in enumerate(merged_regions, start=1):

            longitude, latitude = self._stored_coordinates_from_map_point(
                region.x,
                region.y,
            )
         
            source_identifiers = tuple(
                self.original_viewpoints[index].identifier
                for index in region.source_indices
            )

            self.downsampled_viewpoints.append(
                Viewpoint(
                    identifier=identifier,
                    map_x=region.x,
                    map_y=region.y,
                    longitude=longitude,
                    latitude=latitude,
                    square_bounds=region.bounds,
                    source_identifiers=source_identifiers,
                )
            )

    def _refresh_viewpoint_preview(self) -> None:
        """Redraw the table and DEM overlays for the active mode."""
        viewpoints = self.active_viewpoints

        self.viewpoint_table.blockSignals(True)

        try:
            self.viewpoint_table.setRowCount(0)
            self.dem_view.clear_viewpoint_markers()


            second_header, third_header = (
                self._viewpoint_table_coordinate_headers()
            )
            self.viewpoint_table.setHorizontalHeaderLabels(
                [
                    "Region" if self.analysis_mode == "fast" else "#",
                    second_header,
                    third_header,
                ]
            )
  
            for viewpoint in viewpoints:
                if viewpoint.square_bounds is None:
                    self.dem_view.add_viewpoint_marker(
                        viewpoint.map_x,
                        viewpoint.map_y,
                        viewpoint.identifier,
                    )
                    identifier_text = str(viewpoint.identifier)
                else:

                    # Keep every cookie-cutter square on the DEM. Discarded
                    # regions are visualised in red rather than removed.
                    self.dem_view.add_viewpoint_square(
                        bounds=viewpoint.square_bounds,
                        identifier=viewpoint.identifier,
                        source_count=len(viewpoint.source_identifiers),
                        discarded=(
                            viewpoint.identifier
                            in self.discarded_viewpoint_identifiers
                        ),
                    )

                    identifier_text = (
                        f"{viewpoint.identifier} "
                        f"({len(viewpoint.source_identifiers)})"
                    )

                row = self.viewpoint_table.rowCount()
                self.viewpoint_table.insertRow(row)

                values = (
                    identifier_text,
                    f"{viewpoint.latitude:.6f}",
                    f"{viewpoint.longitude:.6f}",
                )

                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        viewpoint.identifier,
                    )
                    self.viewpoint_table.setItem(row, column, item)
        finally:
            self.viewpoint_table.blockSignals(False)

        self.dem_view.highlight_viewpoint(None)
        self.opf_view.highlight_viewpoint_region(None)

        if self.analysis_mode == "fast":
            merged_count = len(self.downsampled_viewpoints)
            original_count = len(self.original_viewpoints)

            self.viewpoint_count_label.setText(
                f"{merged_count} merged viewpoint"
                f"{'s' if merged_count != 1 else ''} from "
                f"{original_count} original viewpoint"
                f"{'s' if original_count != 1 else ''}"
            )
        else:
            count = len(self.original_viewpoints)

            self.viewpoint_count_label.setText(
                f"{count} original viewpoint"
                f"{'s' if count != 1 else ''}"
            )

        self._update_controls()

    def _receive_viewpoint_click(
        self,
        point: tuple[float, float],
    ) -> None:
        """Convert a DEM click into a stored viewpoint."""

        if self.target_geometry is None:
            return

        map_x, map_y = point


        try:
            longitude, latitude = self._stored_coordinates_from_map_point(
                map_x,
                map_y,
            )
        except RuntimeError as error:
            self._show_error(
                "Coordinate conversion unavailable",
                str(error),
            )
            return


        self._store_viewpoint(
            map_x=map_x,
            map_y=map_y,
            longitude=longitude,
            latitude=latitude,
        )

    def _store_viewpoint(
        self,
        *,
        map_x: float,
        map_y: float,
        longitude: float,
        latitude: float,
    ) -> None:
        """Store one exact input viewpoint and refresh both representations."""

        identifier = max(
            (
                viewpoint.identifier
                for viewpoint in self.original_viewpoints
            ),
            default=0,
        ) + 1

        viewpoint = Viewpoint(
            identifier=identifier,
            map_x=map_x,
            map_y=map_y,
            longitude=longitude,
            latitude=latitude,
            source_identifiers=(identifier,),
        )

        self.original_viewpoints.append(viewpoint)
        self._rebuild_downsampled_viewpoints()
        self._refresh_viewpoint_preview()

        if self.opf_result is not None:
            self._build_viewpoint_results()

        if self.analysis_mode == "slow":
            row = len(self.original_viewpoints) - 1
            self.viewpoint_table.selectRow(row)
        else:
            for row, merged_viewpoint in enumerate(
                self.downsampled_viewpoints
            ):
                if identifier in merged_viewpoint.source_identifiers:
                    self.viewpoint_table.selectRow(row)
                    break

        #use X/Y wording for an Unreal-local DEM.
        self.status_label.setText(
            f"Original viewpoint {identifier} added at "
            f"{self._format_map_position(map_x, map_y)}."
        )


    def _on_viewpoint_table_selection_changed(self) -> None:
        """Highlight the map item linked to the selected table row."""

        selected_items = self.viewpoint_table.selectedItems()

        if not selected_items:
            self.dem_view.highlight_viewpoint(None)
            self.opf_view.highlight_viewpoint_region(None)
            return

        identifier = selected_items[0].data(
            Qt.ItemDataRole.UserRole
        )

        if not isinstance(identifier, int):
            self.dem_view.highlight_viewpoint(None)
            self.opf_view.highlight_viewpoint_region(None)
            return

        self.dem_view.highlight_viewpoint(identifier)
        self.opf_view.highlight_viewpoint_region(identifier)


        viewpoint = next(
            (
                candidate
                for candidate in self.active_viewpoints
                if candidate.identifier == identifier
            ),
            None,
        )

        if viewpoint is None:
            return

        location_text = self._format_map_position(
            viewpoint.map_x,
            viewpoint.map_y,
        )

        if viewpoint.is_merged:
            self.status_label.setText(
                f"Merged viewpoint {identifier} represents "
                f"{len(viewpoint.source_identifiers)} original "
                f"viewpoints; centre at {location_text}."
            )
        else:
            self.status_label.setText(
                f"Original viewpoint {identifier} selected at "
                f"{location_text}."
            )

    def _clear_viewpoints(self) -> None:
        """Clear both exact and downsampled viewpoint representations."""

        self.dem_view.set_viewpoint_selection_active(False)
        self.original_viewpoints.clear()
        self.downsampled_viewpoints.clear()


        self.discarded_viewpoint_identifiers.clear()

        self._refresh_viewpoint_preview()

        self.viewpoint_results.clear()
        self.opf_view.clear_viewpoint_regions()

    def import_viewpoints_csv(self) -> None:
        """Import geographic or Unreal-local coordinates from a CSV file."""

        if self.target_geometry is None:
            self._show_error(
                "CSV import unavailable",
                "Define the region of interest before importing viewpoints.",
            )
            return

        #Unreal-local CSVs contain X/Y and need no transformer.
        if (
            not self._dem_is_unreal_local
            and self._wgs84_to_dem is None
        ):
            self._show_error(
                "CSV import unavailable",
                "Load a georeferenced DEM first.",
            )
            return


        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import viewpoints",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not filename:
            return

        pending: list[tuple[float, float, float, float]] = []

        try:
            with open(
                filename,
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                reader = csv.DictReader(file)

                if reader.fieldnames is None:
                    raise ValueError(
                        "The CSV does not contain a header row."
                    )

                headings = {
                    heading.strip().lower(): heading
                    for heading in reader.fieldnames
                }


                if self._dem_is_unreal_local:
                    x_heading = (
                        headings.get("x")
                        or headings.get("map_x")
                        or headings.get("easting")
                    )
                    y_heading = (
                        headings.get("y")
                        or headings.get("map_y")
                        or headings.get("northing")
                    )

                    if x_heading is None or y_heading is None:
                        raise ValueError(
                            "An Unreal-local CSV must contain x and y columns."
                        )

                    for line_number, row in enumerate(reader, start=2):
                        try:
                            map_x = float(row[x_heading])
                            map_y = float(row[y_heading])
                        except (KeyError, TypeError, ValueError) as error:
                            raise ValueError(
                                "Invalid X/Y coordinates on CSV line "
                                f"{line_number}."
                            ) from error

                        # The legacy fields temporarily carry X and Y.
                        pending.append((map_x, map_y, map_x, map_y))

                else:
                    latitude_heading = (
                        headings.get("lat")
                        or headings.get("latitude")
                    )
                    longitude_heading = (
                        headings.get("lon")
                        or headings.get("longitude")
                    )

                    if (
                        latitude_heading is None
                        or longitude_heading is None
                    ):
                        raise ValueError(
                            "The CSV must contain lat and lon columns."
                        )

                    for line_number, row in enumerate(reader, start=2):
                        try:
                            latitude = float(row[latitude_heading])
                            longitude = float(row[longitude_heading])
                        except (KeyError, TypeError, ValueError) as error:
                            raise ValueError(
                                "Invalid coordinates on CSV line "
                                f"{line_number}."
                            ) from error

                        if not -90.0 <= latitude <= 90.0:
                            raise ValueError(
                                "Latitude outside -90…90 on "
                                f"line {line_number}."
                            )

                        if not -180.0 <= longitude <= 180.0:
                            raise ValueError(
                                "Longitude outside -180…180 on "
                                f"line {line_number}."
                            )

                        assert self._wgs84_to_dem is not None
                        map_x, map_y = self._wgs84_to_dem.transform(
                            longitude,
                            latitude,
                        )
                        pending.append(
                            (map_x, map_y, longitude, latitude)
                        )
  

        except (OSError, ValueError) as error:
            self._show_error("CSV import failed", str(error))
            return

        for map_x, map_y, longitude, latitude in pending:
            self._store_viewpoint(
                map_x=map_x,
                map_y=map_y,
                longitude=longitude,
                latitude=latitude,
            )

        self.tabs.setCurrentWidget(self.dem_view)
        self.status_label.setText(
            f"Imported {len(pending)} viewpoints."
        )


    def _build_viewpoint_results(self) -> None:
        """Extract one cropped OPF region for every active viewpoint."""

        if self.opf_result is None:
            self.viewpoint_results.clear()
            self.opf_view.clear_viewpoint_regions()
            self.discarded_viewpoint_identifiers.clear()

            return

        results: dict[int, ViewpointOPFResult] = {}

        for viewpoint in self.active_viewpoints:

            analysis_region = ViewpointRegion(
                identifier=f"Viewpoint {viewpoint.identifier}",
                x=viewpoint.map_x,
                y=viewpoint.map_y,
                radius_m=(
                    None
                    if viewpoint.square_bounds is not None
                    else self.viewpoint_radius_m
                ),
                bounds=viewpoint.square_bounds,
            )

            try:
                result = build_viewpoint_opf(
                    opf_result=self.opf_result,
                    viewpoint=analysis_region,
                )
            except ValueError:
                continue

            results[viewpoint.identifier] = result

        self.viewpoint_results = results


        self.discarded_viewpoint_identifiers.clear()

        if (self.analysis_mode == "fast" and self.redundancy_checkbox.isChecked() and results):
            correction = correct_redundancy(
                list(results.values()),
                retention=self.redundancy_retention,
            )

            discarded_labels = {
                viewpoint_score.identifier
                for viewpoint_score in correction.discarded
            }

            # The corrector stores the ViewpointRegion label (for example
            # "Viewpoint 3"). Match that label back to the GUI integer ID
            # without parsing the string.
            self.discarded_viewpoint_identifiers = {
                identifier
                for identifier, result in results.items()
                if result.viewpoint.identifier in discarded_labels
            }

            print("\nREDUNDANCY CORRECTION")
            print("---------------------")
            print(
                f"Retention target: {self.redundancy_retention:.1%}"
            )
            print(
                f"Retained: {len(correction.selected)} / "
                f"{len(correction.selected) + len(correction.discarded)}"
            )

            for viewpoint_score in correction.selected:
                print(
                    f"KEEP    {viewpoint_score.identifier} | "
                    f"score={viewpoint_score.score:.2f} | "
                    f"fraction={viewpoint_score.fraction:.2%} | "
                    f"cumulative="
                    f"{viewpoint_score.cumulative_fraction:.2%}"
                )

            for viewpoint_score in correction.discarded:
                print(
                    f"DISCARD {viewpoint_score.identifier} | "
                    f"score={viewpoint_score.score:.2f}"
                )

        # Redraw the DEM overlay now that discard decisions are known.
        self._refresh_viewpoint_preview()



        # Pass fast-mode discard decisions to the OPF renderer so rejected
        # viewpoint cutouts are drawn red there as well as on the DEM.
        self.opf_view.set_viewpoint_regions(
            results,
            discarded_identifiers=self.discarded_viewpoint_identifiers,
        )



    @staticmethod
    def _make_exr_coordinate_spinbox() -> QDoubleSpinBox:
        """Create a centimetre-valued Unreal capture-coordinate input."""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(-1_000_000_000.0, 1_000_000_000.0)
        spinbox.setDecimals(3)
        spinbox.setSuffix(" cm")
        return spinbox

    def choose_ndvi_exr(self) -> None:
        """Choose the Unreal EXR used by the botanical NDVI module."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Unreal NDVI EXR",
            "",
            "OpenEXR files (*.exr);;All files (*)",
        )

        if not filename:
            return

        self.ndvi_exr_path = Path(filename)
        self.ndvi_exr_path_label.setText(str(self.ndvi_exr_path))
        self.ndvi_exr_path_label.setToolTip(str(self.ndvi_exr_path))
        self.exr_ndvi_radio.setChecked(True)
        self.ndvi_checkbox.setChecked(True)

        self.ndvi_view.show_message("Loading NDVI EXR preview…")
        if self.tabs.indexOf(self.ndvi_view) == -1:
            self.tabs.addTab(self.ndvi_view, "NDVI")
        self.tabs.setCurrentWidget(self.ndvi_view)

        worker = FunctionWorker(
            load_ndvi_exr_preview,
            path=self.ndvi_exr_path,
            capture_x_cm=self.exr_capture_x_spin.value(),
            capture_y_cm=self.exr_capture_y_spin.value(),
            ortho_width_cm=self.exr_ortho_width_spin.value(),
            encoded=self.exr_encoded_checkbox.isChecked(),
        )
        self._workers.add(worker)
        worker.signals.result.connect(self._receive_ndvi_exr_preview)
        worker.signals.error.connect(self._receive_ndvi_exr_preview_error)
        worker.signals.finished.connect(
            lambda: self._finish_worker(worker)
        )

        self.status_label.setText(
            "Loading NDVI EXR preview…"
        )
        self.progress_bar.setRange(0, 0)
        self._update_controls()
        self.thread_pool.start(worker)

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
        self._clear_viewpoints()
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


        with rasterio.open(path) as source:
            self._dem_tags = source.tags()

        coordinate_mode = self._dem_tags.get(
            "coordinate_mode",
            "",
        ).strip().lower()

        self._dem_is_unreal_local = (
            coordinate_mode == UNREAL_LOCAL_COORDINATE_MODE
        )

        if self._dem_is_unreal_local:
            # The DEM coordinates already equal Unreal X/Y metres.
            self._dem_to_wgs84 = None
            self._wgs84_to_dem = None
        else:
            try:
                self._dem_to_wgs84 = Transformer.from_crs(
                    self.dem_crs,
                    "EPSG:4326",
                    always_xy=True,
                )
                self._wgs84_to_dem = Transformer.from_crs(
                    "EPSG:4326",
                    self.dem_crs,
                    always_xy=True,
                )
            except Exception as error:  # noqa: BLE001 - present CRS errors
                self._dem_to_wgs84 = None
                self._wgs84_to_dem = None
                self._show_error(
                    "Coordinate conversion unavailable",
                    str(error),
                )
                return

        # Refresh the empty table so its headings immediately show X/Y or
        # latitude/longitude for the newly loaded DEM.
        self._refresh_viewpoint_preview()


        self.visibility_result = None
        self.botanical_result = None
        self.occlusion_result = None
        self.opf_result = None

        self.viewpoint_results.clear()
        self.opf_view.clear_viewpoint_regions()

        self.opf_view.show_message(
            "Run selected modules to show the observability potential field."
        )

        self.dem_path_label.setText(str(path))
        self.dem_path_label.setToolTip(str(path))
        self.tabs.setCurrentWidget(self.dem_view)

        if self._dem_is_unreal_local:
            self.status_label.setText(
                "Unreal-local DEM loaded. X/Y coordinates are already in "
                "metres; define a region of interest to enable viewpoints."
            )
        else:
            self.status_label.setText(
                "DEM loaded. Define a region of interest to enable "
                "viewpoints."
            )

        self._update_controls()

    def run_selected_modules(self) -> None:
        """Launch one worker for the selected components and the OPF."""

        if self.dem_path is None:
            self._show_missing_dem()
            return

        run_los = self.los_checkbox.isChecked()
        run_ndvi = self.ndvi_checkbox.isChecked()
        run_obstacle = self.obstacle_checkbox.isChecked()
        use_ndvi_exr = self.exr_ndvi_radio.isChecked()

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

        if run_ndvi and use_ndvi_exr and self.ndvi_exr_path is None:
            missing_requirements.append(
                "NDVI is set to Unreal EXR, but no EXR file is selected."
            )

        if (
            run_ndvi
            and not use_ndvi_exr
            and not (self.time_from and self.time_to)
        ):
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
            botanical_kwargs: dict[str, Any] = {
                "target_geometry": self.target_geometry,
            }

            if use_ndvi_exr:
                botanical_kwargs.update(
                    {
                        "ndvi_exr_path": self.ndvi_exr_path,
                        "exr_capture_x_cm": (
                            self.exr_capture_x_spin.value()
                        ),
                        "exr_capture_y_cm": (
                            self.exr_capture_y_spin.value()
                        ),
                        "exr_ortho_width_cm": (
                            self.exr_ortho_width_spin.value()
                        ),
                        "exr_encoded": (
                            self.exr_encoded_checkbox.isChecked()
                        ),
                    }
                )
                selected_names.append("NDVI (EXR)")
            else:
                botanical_kwargs.update(
                    {
                        "time_from": self.time_from,
                        "time_to": self.time_to,
                    }
                )
                selected_names.append("NDVI (Sentinel-2)")

            selected_modules["botanical"] = (
                self.botanical_runner,
                botanical_kwargs,
            )
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
            coordinate_mode=self._dem_tags.get("coordinate_mode"),
            generate_viewpoints=not self.original_viewpoints,
            verification_radius_m=30.0,
            suppression_radius_m=50.0,
            max_generated_viewpoints=10,
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

    def _receive_ndvi_exr_preview(
        self,
        raw_result: dict[str, Any],
    ) -> None:
        """Display the decoded EXR immediately in the NDVI tab."""
        try:
            display_result = coerce_raster_result(
                raw_result,
                default_title="NDVI EXR preview",
                default_colour_map="RdYlGn",
                default_colourbar_label="NDVI",
            )
            self.ndvi_view.show_result(display_result)
        except Exception as error:  # noqa: BLE001 - report preview failures
            self._show_error("NDVI EXR preview failed", str(error))
            self.ndvi_view.show_message(
                "The selected NDVI EXR could not be displayed."
            )
            self.status_label.setText("NDVI EXR preview failed.")
            return

        if self.tabs.indexOf(self.ndvi_view) == -1:
            self.tabs.addTab(self.ndvi_view, "NDVI")
        self.tabs.setCurrentWidget(self.ndvi_view)
        self.status_label.setText(
            "NDVI EXR preview loaded. Run the selected modules to align it "
            "to the DEM and include it in the OPF."
        )

    def _receive_ndvi_exr_preview_error(
        self,
        message: str,
        traceback_text: str,
    ) -> None:
        """Show an exception raised while decoding an EXR preview."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("NDVI EXR preview failed")
        box.setText(message or "The selected EXR could not be loaded.")
        box.setDetailedText(traceback_text)
        box.exec()

        self.ndvi_view.show_message(
            "The selected NDVI EXR could not be displayed."
        )
        self.status_label.setText("NDVI EXR preview failed.")

    def _receive_pipeline_result(self, output: PipelineOutput) -> None:
        """Store and display all results from a completed pipeline."""

        results = output.component_results

        if output.generated_viewpoints:
            self._store_generated_viewpoints(
                output.generated_viewpoints
            )

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
                name="NDVI",
                destination=self.ndvi_view,
                raw_result=results["botanical"],
                default_title="NDVI",
                default_colour_map="RdYlGn",
                default_colourbar_label="NDVI",
                result_attribute="botanical_result",
                display_mode="raster",
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
        self._receive_opf_result(
            output.opf_result,
            fallback_transform=output.transform,
            fallback_crs=output.crs,
        )


    def _store_generated_viewpoints(
        self,
        generated: list[ViewpointGenerationResult],
    ) -> None:
        """Store OPF-generated viewpoints only when none were supplied."""

        # User-supplied viewpoints always take precedence. 
        # Generation is already disabled when a run starts with original viewpoints.
        if self.original_viewpoints:
            return

        next_identifier = max(
            (
                viewpoint.identifier
                for viewpoint in self.original_viewpoints
            ),
            default=0,
        ) + 1

        for result in generated:
            region = result.viewpoint

            longitude, latitude = (
                self._stored_coordinates_from_map_point(
                    region.x,
                    region.y,
                )
            )

            identifier = next_identifier
            next_identifier += 1

            self.original_viewpoints.append(
                Viewpoint(
                    identifier=identifier,
                    map_x=region.x,
                    map_y=region.y,
                    longitude=longitude,
                    latitude=latitude,
                    source_identifiers=(identifier,),
                )
            )

        self._rebuild_downsampled_viewpoints()
        self._refresh_viewpoint_preview()



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

    def _receive_opf_result(
        self,
        raw_result: object,
        *,
        fallback_transform: Any | None = None,
        fallback_crs: Any | None = None,
    ) -> None:
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

            self._build_viewpoint_results()
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
        has_roi = self.target_geometry is not None
        any_selected = any(
            (
                self.los_checkbox.isChecked(),
                self.ndvi_checkbox.isChecked(),
                self.obstacle_checkbox.isChecked(),
            )
        )

        self.load_dem_button.setEnabled(not is_busy)
        self.slow_mode_action.setEnabled(not is_busy)
        self.fast_mode_action.setEnabled(not is_busy)

        self.sentinel_ndvi_radio.setEnabled(not is_busy)
        self.exr_ndvi_radio.setEnabled(not is_busy)

        exr_controls_enabled = (
            self.exr_ndvi_radio.isChecked()
            and not is_busy
        )
        self.load_ndvi_exr_button.setEnabled(exr_controls_enabled)
        self.exr_capture_x_spin.setEnabled(exr_controls_enabled)
        self.exr_capture_y_spin.setEnabled(exr_controls_enabled)
        self.exr_ortho_width_spin.setEnabled(exr_controls_enabled)
        self.exr_encoded_checkbox.setEnabled(exr_controls_enabled)

        checkboxes_enabled = has_dem and not is_busy
        self.los_checkbox.setEnabled(checkboxes_enabled)
        self.ndvi_checkbox.setEnabled(checkboxes_enabled)
        self.obstacle_checkbox.setEnabled(checkboxes_enabled)

        viewpoint_entry_enabled = (
            has_dem
            and has_roi
            and not is_busy
        )

        self.import_viewpoints_button.setEnabled(
            viewpoint_entry_enabled
        )
        self.add_viewpoints_button.setEnabled(
            viewpoint_entry_enabled
        )
        self.dem_view.set_viewpoint_selection_enabled(
            viewpoint_entry_enabled
        )

        self.run_selected_button.setEnabled(
            has_dem and any_selected and not is_busy
        )

        self.dem_view.set_area_selection_enabled(
            has_dem and not is_busy
        )

        self.redundancy_checkbox.setEnabled(
            has_dem
            and self.analysis_mode == "fast"
            and not is_busy
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

            QLabel#fastModeWarning {
                color: #ffb74d;
                font-weight: 700;
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

            QTableWidget {
                background-color: #202020;
                alternate-background-color: #252525;
                border: 1px solid #3d3d3d;
                gridline-color: #3d3d3d;
                selection-background-color: #5a4b00;
                selection-color: #ffffff;
            }

            QHeaderView::section {
                background-color: #292929;
                border: 1px solid #3d3d3d;
                padding: 5px;
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
