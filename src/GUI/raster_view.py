#raster_view.py
"""Matplotlib-based raster display widget for PySide6."""

from __future__ import annotations

from math import ceil, floor

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from affine import Affine
from rasterio.enums import Resampling
from collections.abc import Callable

from matplotlib.backend_bases import MouseButton
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import Polygon as PolygonPatch

from matplotlib.patches import (
    Polygon as PolygonPatch,
    Rectangle as RectanglePatch,
)

from matplotlib.widgets import PolygonSelector

from PySide6.QtCore import QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction
from rasterio.windows import Window, crop, from_bounds

from .dem_downsampler import (
    copy_raster_metadata,
    downsample_raster_for_processing,
)
from .worker import FunctionWorker

import numpy as np
import rasterio
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget
from rasterio.plot import plotting_extent


RasterArray = np.ndarray | np.ma.MaskedArray


@dataclass(frozen=True, slots=True)
class RasterCropInfo:
    """Metadata for the processing crop written to disk."""

    path: Path
    transform: Affine
    crs: Any
    width: int
    height: int


def _clamped_window_from_bounds(
    source: Any,
    bounds: tuple[float, float, float, float],
) -> Window:
    """Convert map bounds to a complete-pixel window inside a raster."""

    left, bottom, right, top = bounds

    left = max(left, source.bounds.left)
    bottom = max(bottom, source.bounds.bottom)
    right = min(right, source.bounds.right)
    top = min(top, source.bounds.top)

    if left >= right or bottom >= top:
        raise ValueError("The visible area does not overlap the raster.")

    floating_window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=source.transform,
    )

    column_start = floor(floating_window.col_off)
    row_start = floor(floating_window.row_off)
    column_end = ceil(floating_window.col_off + floating_window.width)
    row_end = ceil(floating_window.row_off + floating_window.height)

    window = Window(
        col_off=column_start,
        row_off=row_start,
        width=column_end - column_start,
        height=row_end - row_start,
    )

    return crop(window, source.height, source.width)


def write_raster_crop(
    *,
    source_path: str | Path,
    bounds: tuple[float, float, float, float],
    output_path: str | Path,
    coordinate_mode: str | None = None,
) -> RasterCropInfo:
    """Write the visible DEM window as a processing GeoTIFF.

    The crop is copied block by block so creating it does not require loading
    the entire visible window into memory at once. The original raster is
    never modified. A finer-than-1-m temporary crop is reduced for processing
    after it is written, using maximum elevation per destination cell.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_path) as source:
        window = _clamped_window_from_bounds(source, bounds)
        width = max(1, int(window.width))
        height = max(1, int(window.height))
        transform = source.window_transform(window)

        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            transform=transform,
        )

        with rasterio.open(output_path, "w", **profile) as destination:
            # Copy one output block at a time to keep peak memory low.
            for _, destination_window in destination.block_windows(1):
                source_window = Window(
                    col_off=window.col_off + destination_window.col_off,
                    row_off=window.row_off + destination_window.row_off,
                    width=destination_window.width,
                    height=destination_window.height,
                )
                destination.write(
                    source.read(window=source_window),
                    window=destination_window,
                )

            copy_raster_metadata(source, destination)

    downsample_raster_for_processing(output_path)

    # Reopen the final processing raster because downsampling can change every
    # grid property consumed by the component modules and OPF.
    with rasterio.open(output_path) as processed:
        return RasterCropInfo(
            path=output_path,
            transform=processed.transform,
            crs=processed.crs,
            width=processed.width,
            height=processed.height,
        )


def read_raster_window(
    *,
    path: Path,
    bounds: tuple[float, float, float, float],
    target_width: int,
    target_height: int,
    title: str,
    colour_map: str,
    colourbar_label: str,
) -> RasterResult:
    """
    Read only the requested map extent from a raster.

    The returned array is limited to approximately the display resolution,
    avoiding unnecessarily large arrays.
    """

    with rasterio.open(path) as source:
        window = _clamped_window_from_bounds(source, bounds)

        native_width = max(1, int(window.width))
        native_height = max(1, int(window.height))

        # Reduce the window only when it contains more pixels than the
        # display can use.
        scale = max(
            native_width / target_width,
            native_height / target_height,
            1.0,
        )

        output_width = max(
            1,
            round(native_width / scale),
        )
        output_height = max(
            1,
            round(native_height / scale),
        )

        data = source.read(
            1,
            window=window,
            out_shape=(output_height, output_width),
            masked=True,
            resampling=Resampling.bilinear,
        ).astype(
            np.float32,
            copy=False,
        )

        data = np.ma.masked_invalid(data)
        data = np.ma.masked_where(
            np.abs(data) > 1e20,
            data,
        )

        # Transform for the native raster window.
        window_transform = source.window_transform(window)

        # Adjust it again because the window may have been resampled to
        # a smaller display array.
        display_transform = window_transform * Affine.scale(
            window.width / output_width,
            window.height / output_height,
        )

        return RasterResult(
            data=data,
            transform=display_transform,
            crs=source.crs,
            title=title,
            colour_map=colour_map,
            colourbar_label=colourbar_label,
        )

@dataclass(slots=True)
class RasterResult:
    """A common display format for LoS, NDVI and DEM rasters."""

    data: RasterArray
    transform: Any | None = None
    crs: Any | None = None
    title: str = "Raster result"
    colour_map: str = "viridis"
    colourbar_label: str = ""
    vmin: float | None = None
    vmax: float | None = None


def read_raster_result(
    path: str | Path,
    *,
    title: str,
    colour_map: str,
    colourbar_label: str,
    max_preview_dimension: int = 2000,
) -> RasterResult:
    """Read a downsampled raster preview for display."""

    raster_path = Path(path)

    with rasterio.open(raster_path) as source:
        scale = max(
            source.width / max_preview_dimension,
            source.height / max_preview_dimension,
            1.0,
        )

        preview_width = max(
            1,
            round(source.width / scale),
        )
        preview_height = max(
            1,
            round(source.height / scale),
        )

        data = source.read(
            1,
            out_shape=(preview_height, preview_width),
            masked=True,
            resampling=Resampling.bilinear,
        ).astype(
            np.float32,
            copy=False,
        )

        preview_transform = source.transform * Affine.scale(
            source.width / preview_width,
            source.height / preview_height,
        )

        data = np.ma.masked_invalid(data)
        data = np.ma.masked_where(
            np.abs(data) > 1e20,
            data,
        )

        return RasterResult(
            data=data,
            transform=preview_transform,
            crs=source.crs,
            title=title,
            colour_map=colour_map,
            colourbar_label=colourbar_label,
        )


def coerce_raster_result(
    value: object,
    *,
    default_title: str,
    default_colour_map: str,
    default_colourbar_label: str,
    fallback_transform: Any | None = None,
    fallback_crs: Any | None = None,
) -> RasterResult:
    """
    Convert common backend return formats into :class:`RasterResult`.

    Supported values:
    - ``RasterResult``
    - a NumPy array
    - a GeoTIFF path
    - ``(data, transform)``
    - ``(data, transform, crs)``
    - a mapping containing ``data`` and optional display fields

    Returning ``RasterResult`` directly from each module is still the clearest
    long-term interface.
    """
    if isinstance(value, RasterResult):
        return value
    
    display_converter = getattr(value, "as_display_mapping", None)

    if callable(display_converter):
        value = display_converter()

    if isinstance(value, (str, Path)):
        return read_raster_result(
            value,
            title=default_title,
            colour_map=default_colour_map,
            colourbar_label=default_colourbar_label,
        )

    if isinstance(value, np.ndarray) or np.ma.isMaskedArray(value):
        return RasterResult(
            data=value,
            transform=fallback_transform,
            crs=fallback_crs,
            title=default_title,
            colour_map=default_colour_map,
            colourbar_label=default_colourbar_label,
        )

    if isinstance(value, Mapping):
        if "data" not in value:
            raise TypeError("Raster result mapping must contain a 'data' value.")

        return RasterResult(
            data=value["data"],
            transform=value.get("transform", fallback_transform),
            crs=value.get("crs", fallback_crs),
            title=str(value.get("title", default_title)),
            colour_map=str(
                value.get(
                    "colour_map",
                    value.get("cmap", default_colour_map),
                )
            ),
            colourbar_label=str(
                value.get("colourbar_label", default_colourbar_label)
            ),
            vmin=value.get("vmin"),
            vmax=value.get("vmax"),
        )

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) not in (2, 3):
            raise TypeError(
                "Raster result tuple must be (data, transform) or "
                "(data, transform, crs)."
            )

        data = value[0]
        transform = value[1]
        crs = value[2] if len(value) == 3 else fallback_crs

        return RasterResult(
            data=data,
            transform=transform,
            crs=crs,
            title=default_title,
            colour_map=default_colour_map,
            colourbar_label=default_colourbar_label,
        )

    raise TypeError(
        "Module returned an unsupported result. Return RasterResult, a NumPy "
        "array, a raster path, a mapping, or a (data, transform) tuple."
    )


class RasterNavigationToolbar(NavigationToolbar):
    """Matplotlib toolbar with mutually exclusive custom map tools."""

    area_selection_toggled = Signal(bool)
    clear_area_requested = Signal()
    viewpoint_selection_toggled = Signal(bool)
    clear_viewpoints_requested = Signal()

    def __init__(
        self,
        canvas: FigureCanvas,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(canvas, parent)

        self.addSeparator()


        self.select_area_action = QAction("⬠", self)
        self.select_area_action.setCheckable(True)
        self.select_area_action.setToolTip(
            "Select a polygonal survey area"
        )
        self.select_area_action.setEnabled(False)
        self.select_area_action.toggled.connect(
            self._on_area_selection_toggled
        )
        self.addAction(self.select_area_action)

        self.clear_area_action = QAction("✕", self)
        self.clear_area_action.setToolTip(
            "Clear the selected survey area (Delete)"
        )
        self.clear_area_action.setShortcut("Delete")
        self.clear_area_action.setEnabled(False)
        self.clear_area_action.triggered.connect(
            self.clear_area_requested.emit
        )
        self.addAction(self.clear_area_action)

        self.addSeparator()

        self.add_viewpoint_action = QAction("⌖", self)
        self.add_viewpoint_action.setCheckable(True)
        self.add_viewpoint_action.setToolTip(
            "Add viewpoints by clicking on the DEM"
        )
        self.add_viewpoint_action.setEnabled(False)
        self.add_viewpoint_action.toggled.connect(
            self._on_viewpoint_selection_toggled
        )
        self.addAction(self.add_viewpoint_action)

        self.clear_viewpoints_action = QAction("×⌖", self)
        self.clear_viewpoints_action.setToolTip(
            "Clear all viewpoints"
        )
        self.clear_viewpoints_action.setEnabled(False)
        self.clear_viewpoints_action.triggered.connect(
            self.clear_viewpoints_requested.emit
        )
        self.addAction(self.clear_viewpoints_action)

    def _on_area_selection_toggled(self, enabled: bool) -> None:
        if enabled:
            self._turn_off_matplotlib_navigation()

            if self.add_viewpoint_action.isChecked():
                self.add_viewpoint_action.setChecked(False)

        self.area_selection_toggled.emit(enabled)

    def _on_viewpoint_selection_toggled(
        self,
        enabled: bool,
    ) -> None:
        if enabled:
            self._turn_off_matplotlib_navigation()

            if self.select_area_action.isChecked():
                self.select_area_action.setChecked(False)

        self.viewpoint_selection_toggled.emit(enabled)

    def _turn_off_matplotlib_navigation(self) -> None:
        """Turn off Pan or Zoom without triggering the overrides."""

        mode_name = getattr(self.mode, "name", "")
        mode_text = f"{mode_name} {self.mode}".lower()

        if "pan" in mode_text:
            super().pan()
        elif "zoom" in mode_text:
            super().zoom()

    def _turn_off_custom_tools(self) -> None:
        """Turn off Rivelero's custom map-interaction tools."""

        if self.select_area_action.isChecked():
            self.select_area_action.setChecked(False)

        if self.add_viewpoint_action.isChecked():
            self.add_viewpoint_action.setChecked(False)

    def pan(self, *args: Any) -> None:
        """Disable custom tools before toggling Matplotlib pan."""

        self._turn_off_custom_tools()
        super().pan(*args)

    def zoom(self, *args: Any) -> None:
        """Disable custom tools before toggling Matplotlib zoom."""

        self._turn_off_custom_tools()
        super().zoom(*args)

    def set_area_selection_enabled(self, enabled: bool) -> None:
        """Enable area selection only while it is available."""

        if not enabled and self.select_area_action.isChecked():
            self.select_area_action.setChecked(False)

        self.select_area_action.setEnabled(enabled)

    def set_area_clear_enabled(self, enabled: bool) -> None:
        """Enable clearing only when an area exists or is being drawn."""

        self.clear_area_action.setEnabled(enabled)

    def set_viewpoint_selection_enabled(
        self,
        enabled: bool,
    ) -> None:
        """Enable or disable the viewpoint-placement action."""

        if not enabled and self.add_viewpoint_action.isChecked():
            self.add_viewpoint_action.setChecked(False)

        self.add_viewpoint_action.setEnabled(enabled)

    def set_viewpoint_clear_enabled(
        self,
        enabled: bool,
    ) -> None:
        """Enable or disable the clear-viewpoints action."""

        self.clear_viewpoints_action.setEnabled(enabled)


class RasterView(QWidget):
    """A reusable Matplotlib raster viewer with toolbar and wheel zoom."""

    area_selection_started = Signal()
    area_selected = Signal(object)
    area_selection_cleared = Signal()
    area_selection_error = Signal(str)

    viewpoint_clicked = Signal(object)
    viewpoint_selection_toggled = Signal(bool)
    viewpoints_clear_requested = Signal()
    viewpoint_selection_error = Signal(str)

    def __init__(
        self,
        empty_message: str = "No raster loaded",
        parent: QWidget | None = None,
        *,
        allow_area_selection: bool = False,
        allow_viewpoint_selection: bool = False,
    ) -> None:
        super().__init__(parent)

        self._area_selector: PolygonSelector | None = None
        self._area_patch: PolygonPatch | None = None
        self._roi_path: MatplotlibPath | None = None
        self._viewpoint_artists: dict[int, tuple[Any, Any]] = {}

        self._viewpoint_square_artists: dict[int, tuple[Any, Any]] = {}


        # IDs of fast-mode cookie-cutter squares rejected by the post-OPF
        # redundancy corrector. The squares remain visible and are drawn red.
        self._discarded_viewpoint_square_identifiers: set[int] = set()


        self._selected_viewpoint_identifier: int | None = None


        self._viewpoint_region_results: dict[int, object] = {}
        self._viewpoint_region_artists: dict[int, Any] = {}
        self._selected_viewpoint_region_identifier: int | None = None

        self._discarded_viewpoint_region_identifiers: set[int] = set()


        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = RasterNavigationToolbar(self.canvas, self)
        self.toolbar.select_area_action.setVisible(allow_area_selection)
        self.toolbar.clear_area_action.setVisible(allow_area_selection)

        if allow_area_selection:
            self.toolbar.area_selection_toggled.connect(
                self._toggle_area_selection
            )
            self.toolbar.clear_area_requested.connect(
                self.clear_selected_area
            )

        self.toolbar.add_viewpoint_action.setVisible(
            allow_viewpoint_selection
        )
        self.toolbar.clear_viewpoints_action.setVisible(
            allow_viewpoint_selection
        )

        if allow_viewpoint_selection:
            self.toolbar.viewpoint_selection_toggled.connect(
                self._toggle_viewpoint_selection
            )
            self.toolbar.clear_viewpoints_requested.connect(
                self.viewpoints_clear_requested.emit
            )

        self._axes = None
        self._colourbar = None


        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect(
            "button_press_event",
            self._on_viewpoint_click,
        )
        self.show_message(empty_message)

    def show_message(self, message: str) -> None:
        """Replace the plot with a centred status message."""
        self.figure.clear()
        self._colourbar = None
        self._axes = self.figure.add_subplot(111)

        self._axes.text(
            0.5,
            0.5,
            message,
            horizontalalignment="center",
            verticalalignment="center",
            transform=self._axes.transAxes,
        )
        self._axes.set_axis_off()
        self.canvas.draw_idle()

    def show_result(self, result: RasterResult) -> None:
        """Draw a raster and an optional colour bar."""
        data = np.ma.masked_invalid(result.data)

        if data.ndim != 2:
            raise ValueError(
                f"Raster display expects a 2D array, received shape {data.shape}."
            )

        self.figure.clear()
        self._colourbar = None
        self._axes = self.figure.add_subplot(111)

        image_arguments: dict[str, object] = {
            "origin": "upper",
            "cmap": result.colour_map,
            "vmin": result.vmin,
            "vmax": result.vmax,
        }

        if result.transform is not None:
            image_arguments["extent"] = plotting_extent(data, result.transform)

        image = self._axes.imshow(data, **image_arguments)
        self._axes.set_title(result.title)
        self._axes.set_aspect("equal")

        if result.transform is not None:
            self._axes.set_xlabel("Easting / X")
            self._axes.set_ylabel("Northing / Y")
        else:
            self._axes.set_xlabel("Column")
            self._axes.set_ylabel("Row")

        self._colourbar = self.figure.colorbar(image, ax=self._axes)

        if result.colourbar_label:
            self._colourbar.set_label(result.colourbar_label)

        self.figure.tight_layout()

        self._redraw_viewpoint_regions()

        self.canvas.draw_idle()

    def show_surface(
        self,
        result: RasterResult,
        *,
        colour_map: str | None = None,
        max_samples_per_axis: int = 240,
    ) -> None:
        """Draw either the complete field or its viewpoint regions."""

        data = np.ma.masked_invalid(result.data)

        if data.ndim != 2:
            raise ValueError(
                f"3D display expects a 2D array, received shape {data.shape}."
            )

        height, width = data.shape
        row_step = max(1, ceil(height / max_samples_per_axis))
        column_step = max(1, ceil(width / max_samples_per_axis))

        rows = np.arange(
            0,
            height,
            row_step,
            dtype=np.float64,
        )
        columns = np.arange(
            0,
            width,
            column_step,
            dtype=np.float64,
        )
        column_grid, row_grid = np.meshgrid(columns, rows)

        sampled = data[::row_step, ::column_step]
        z = np.asarray(
            sampled.filled(np.nan),
            dtype=np.float32,
        )

        if not np.isfinite(z).any():
            raise ValueError(
                "The field contains no finite values to display."
            )

        if result.transform is not None:
            pixel_columns = column_grid + 0.5
            pixel_rows = row_grid + 0.5

            x = (
                result.transform.a * pixel_columns
                + result.transform.b * pixel_rows
                + result.transform.c
            )
            y = (
                result.transform.d * pixel_columns
                + result.transform.e * pixel_rows
                + result.transform.f
            )
        else:
            x = column_grid
            y = row_grid

        z_min = result.vmin

        if z_min is None:
            z_min = float(np.nanmin(z))

        z_max = result.vmax

        if z_max is None:
            z_max = float(np.nanmax(z))

        if z_min == z_max:
            z_max = z_min + 1.0

        self._remove_viewpoint_region_artists()
        self.figure.clear()
        self._colourbar = None
        self._axes = self.figure.add_subplot(
            111,
            projection="3d",
        )

        self._axes.set_title(result.title)
        self._axes.set_xlabel(
            "Easting / X"
            if result.transform is not None
            else "Column"
        )
        self._axes.set_ylabel(
            "Northing / Y"
            if result.transform is not None
            else "Row"
        )
        self._axes.set_zlabel(
            result.colourbar_label or "Field height"
        )
        self._axes.set_zlim(z_min, z_max)
        self._axes.view_init(elev=28, azim=-58)

        try:
            self._axes.set_box_aspect((1.45, 1.0, 0.55))
        except AttributeError:
            pass

        if self._viewpoint_region_results:
            self._redraw_viewpoint_regions()
        else:
            surface = self._axes.plot_surface(
                x,
                y,
                z,
                cmap=colour_map or result.colour_map,
                vmin=z_min,
                vmax=z_max,
                rstride=1,
                cstride=1,
                linewidth=0.18,
                edgecolor="#d39b00",
                antialiased=True,
                shade=True,
            )

            self._colourbar = self.figure.colorbar(
                surface,
                ax=self._axes,
                shrink=0.68,
                pad=0.10,
            )

            if result.colourbar_label:
                self._colourbar.set_label(
                    result.colourbar_label
                )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    
    def set_viewpoint_regions(
        self,
        results: Mapping[int, object],
        *,
        discarded_identifiers: set[int] | None = None,
    ) -> None:
        """Store the cropped OPF regions and draw them as overlays."""

        self._viewpoint_region_results = dict(results)

        # Copy the IDs so this view owns its current rendering state rather
        # than depending on a mutable set held by MainWindow.
        self._discarded_viewpoint_region_identifiers = set(
            discarded_identifiers or ()
        )


        if (
            self._selected_viewpoint_region_identifier
            not in self._viewpoint_region_results
        ):
            self._selected_viewpoint_region_identifier = None

        self._redraw_viewpoint_regions()


    def highlight_viewpoint_region(
        self,
        identifier: int | None,
    ) -> None:
        """Highlight the OPF region linked to a selected table row."""

        self._selected_viewpoint_region_identifier = identifier
        self._redraw_viewpoint_regions()


    def clear_viewpoint_regions(self) -> None:
        """Remove all stored and visible viewpoint-region overlays."""

        self._viewpoint_region_results.clear()
        self._selected_viewpoint_region_identifier = None


        self._discarded_viewpoint_region_identifiers.clear()

        self._remove_viewpoint_region_artists()
        self.canvas.draw_idle()


    def _remove_viewpoint_region_artists(self) -> None:
        """Remove currently attached viewpoint-region surfaces."""

        artists = tuple(self._viewpoint_region_artists.values())
        self._viewpoint_region_artists.clear()

        for artist in artists:
            if artist.axes is None:
                continue

            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass


    def _redraw_viewpoint_regions(self) -> None:
        """Draw all regions, emphasizing the table-selected one."""

        self._remove_viewpoint_region_artists()

        if self._axes is None:
            return

        if getattr(self._axes, "name", "") != "3d":
            return

        if not self._viewpoint_region_results:
            self.canvas.draw_idle()
            return

        z_bottom, z_top = self._axes.get_zlim()
        z_span = max(abs(z_top - z_bottom), 1.0) #slight offset to each layer to avoid z-fighting.

        for identifier, result in self._viewpoint_region_results.items():
            data = np.ma.masked_invalid(result.field)

            if data.ndim != 2:
                continue

            height, width = data.shape
            row_step = max(1, ceil(height / 90))
            column_step = max(1, ceil(width / 90))

            rows = (
                np.arange(0, height, row_step, dtype=np.float64)
                + 0.5
            )
            columns = (
                np.arange(0, width, column_step, dtype=np.float64)
                + 0.5
            )
            column_grid, row_grid = np.meshgrid(columns, rows)

            transform = result.transform
            x = (
                transform.a * column_grid
                + transform.b * row_grid
                + transform.c
            )
            y = (
                transform.d * column_grid
                + transform.e * row_grid
                + transform.f
            )

            sampled = data[::row_step, ::column_step]
            z = np.asarray(
                sampled.filled(np.nan),
                dtype=np.float32,
            )

            if not np.isfinite(z).any():
                continue

            selected = (
                identifier
                == self._selected_viewpoint_region_identifier
            )


            discarded = (
                identifier
                in self._discarded_viewpoint_region_identifiers
            )

            z = z + z_span * (0.012 if selected else 0.006)

            display_mapping = result.as_display_mapping()


            # Retained regions keep their OPF colour map. Discarded regions
            # use a solid red surface, while table selection still adds the
            # existing yellow outline.
            surface_arguments: dict[str, object] = {
                "alpha": 1.0,
                "linewidth": 0.7 if selected else 0.0,
                "edgecolor": "#ffd54f" if selected else "none",
                "antialiased": True,
                "shade": True,
            }

            if discarded:
                surface_arguments["color"] = "#d32f2f"
            else:
                surface_arguments.update(
                    cmap=display_mapping["colour_map"],
                    vmin=display_mapping["vmin"],
                    vmax=display_mapping["vmax"],
                )

            surface = self._axes.plot_surface(
                x,
                y,
                z,
                **surface_arguments,
            )

            self._viewpoint_region_artists[identifier] = surface

        self.canvas.draw_idle()


    def _on_scroll(self, event: Any) -> None:
        """Zoom around the mouse cursor when no toolbar tool is active."""
        if self._axes is None or event.inaxes is not self._axes:
            return

        if getattr(self._axes, "name", "") == "3d":
            return

        if event.xdata is None or event.ydata is None:
            return

        if self.toolbar.mode:
            return

        if event.button == "up":
            scale_factor = 0.8
        elif event.button == "down":
            scale_factor = 1.25
        else:
            return

        x_left, x_right = self._axes.get_xlim()
        y_bottom, y_top = self._axes.get_ylim()

        x = float(event.xdata)
        y = float(event.ydata)

        self._axes.set_xlim(
            x - (x - x_left) * scale_factor,
            x + (x_right - x) * scale_factor,
        )
        self._axes.set_ylim(
            y - (y - y_bottom) * scale_factor,
            y + (y_top - y) * scale_factor,
        )

        self.canvas.draw_idle()

    def _toggle_viewpoint_selection(
        self,
        enabled: bool,
    ) -> None:
        """Validate and report viewpoint-placement mode changes."""

        if enabled and (
            self._axes is None
            or not self._axes.images
        ):
            self.toolbar.add_viewpoint_action.blockSignals(True)
            self.toolbar.add_viewpoint_action.setChecked(False)
            self.toolbar.add_viewpoint_action.blockSignals(False)

            self.viewpoint_selection_toggled.emit(False)
            self.viewpoint_selection_error.emit(
                "A raster must be displayed before adding viewpoints."
            )
            return

        self.viewpoint_selection_toggled.emit(enabled)

    def _on_viewpoint_click(self, event: Any) -> None:
        """Emit one DEM-coordinate point inside the active ROI."""

        if not self.toolbar.add_viewpoint_action.isChecked():
            return

        if event.button != MouseButton.LEFT:
            return

        if event.inaxes is not self._axes:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        if self._roi_path is None:
            self.viewpoint_selection_error.emit(
                "Select a region of interest before adding viewpoints."
            )
            return

        if not self._roi_path.contains_point((x, y), radius=1e-9):
            self.viewpoint_selection_error.emit(
                "Viewpoints may only be added inside the selected ROI."
            )
            return

        self.viewpoint_clicked.emit((x, y))

    def set_viewpoint_selection_active(
        self,
        active: bool,
    ) -> None:
        """Turn viewpoint placement on or off."""

        self.toolbar.add_viewpoint_action.setChecked(active)

    def set_viewpoint_selection_enabled(
        self,
        enabled: bool,
    ) -> None:
        """Enable placement only when GUI prerequisites exist."""

        self.toolbar.set_viewpoint_selection_enabled(enabled)
        self.toolbar.set_viewpoint_clear_enabled(
            enabled
            and bool(
                self._viewpoint_artists
                or self._viewpoint_square_artists
            )
        )

    def add_viewpoint_marker(
        self,
        x: float,
        y: float,
        identifier: int,
    ) -> None:
        """Draw one numbered marker without clearing the raster."""

        if self._axes is None or not self._axes.images:
            raise RuntimeError(
                "A raster must be displayed before drawing viewpoints."
            )

        marker = self._axes.scatter(
            [x],
            [y],
            s=55,
            marker="o",
            facecolors="#111111",
            edgecolors="#ffffff",
            linewidths=1.6,
            zorder=8,
        )
        label = self._axes.annotate(
            str(identifier),
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            color="#ffffff",
            fontsize=9,
            fontweight="bold",
            zorder=9,
        )

        self._viewpoint_artists[identifier] = (marker, label)
        self.toolbar.set_viewpoint_clear_enabled(True)
        self.canvas.draw_idle()


    def add_viewpoint_square(
        self,
        *,
        bounds: tuple[float, float, float, float],
        identifier: int,
        source_count: int,
        # --- REDUNDANCY CORRECTOR EDIT ---
        discarded: bool = False,
        # --- END REDUNDANCY CORRECTOR EDIT ---
    ) -> None:
        """Draw one fixed-grid square representing merged viewpoints."""
        if self._axes is None or not self._axes.images:
            raise RuntimeError(
                "A raster must be displayed before drawing viewpoints."
            )

        xmin, ymin, xmax, ymax = bounds

        # --- REDUNDANCY CORRECTOR EDIT ---
        if discarded:
            self._discarded_viewpoint_square_identifiers.add(identifier)
        else:
            self._discarded_viewpoint_square_identifiers.discard(identifier)

        face_colour = "#c6282844" if discarded else "#5f7d8a33"
        edge_colour = "#ff5252" if discarded else "#ffffff"
        label_colour = "#ff8a80" if discarded else "#ffffff"
        # --- END REDUNDANCY CORRECTOR EDIT ---

        square = RectanglePatch(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            # --- REDUNDANCY CORRECTOR EDIT ---
            facecolor=face_colour,
            edgecolor=edge_colour,
            # --- END REDUNDANCY CORRECTOR EDIT ---
            linewidth=1.8,
            zorder=8,
        )
        self._axes.add_patch(square)

        label = self._axes.annotate(
            f"{identifier} × {source_count}",
            ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0),
            ha="center",
            va="center",
            # --- REDUNDANCY CORRECTOR EDIT ---
            color=label_colour,
            # --- END REDUNDANCY CORRECTOR EDIT ---
            fontsize=9,
            fontweight="bold",
            zorder=9,
        )

        self._viewpoint_square_artists[identifier] = (square, label)
        self.toolbar.set_viewpoint_clear_enabled(True)
        self.canvas.draw_idle()
   
    def highlight_viewpoint(
        self,
        identifier: int | None,
    ) -> None:
        """Highlight the marker linked to a selected table row."""

        self._selected_viewpoint_identifier = identifier

        for viewpoint_identifier, (marker, label) in (
            self._viewpoint_artists.items()
        ):
            selected = viewpoint_identifier == identifier
            marker.set_sizes([95 if selected else 55])
            marker.set_linewidths([3.0 if selected else 1.6])
            marker.set_edgecolors(
                ["#ffd54f" if selected else "#ffffff"]
            )
            label.set_color(
                "#ffd54f" if selected else "#ffffff"
            )
            label.set_fontsize(10 if selected else 9)

        for viewpoint_identifier, (square, label) in (
            self._viewpoint_square_artists.items()
        ):
            selected = viewpoint_identifier == identifier


            # Selection still uses yellow. When selection ends, restore the
            # square to either normal styling or discarded-red styling.
            discarded = (
                viewpoint_identifier
                in self._discarded_viewpoint_square_identifiers
            )

            normal_edge_colour = (
                "#ff5252" if discarded else "#ffffff"
            )
            normal_face_colour = (
                "#c6282844" if discarded else "#5f7d8a33"
            )
            normal_label_colour = (
                "#ff8a80" if discarded else "#ffffff"
            )

            square.set_linewidth(3.0 if selected else 1.8)
            square.set_edgecolor(
                "#ffd54f" if selected else normal_edge_colour
            )
            square.set_facecolor(
                "#ffd54f44" if selected else normal_face_colour
            )

            label.set_color(
                "#ffd54f" if selected else normal_label_colour
            )
            label.set_fontsize(10 if selected else 9)
  
        self.canvas.draw_idle()
        
    def clear_viewpoint_markers(self) -> None:
        """Remove every visible point marker and merged square."""
        artist_groups = (
            *self._viewpoint_artists.values(),
            *self._viewpoint_square_artists.values(),
        )

        for geometry, label in artist_groups:
            for artist in (geometry, label):
                try:
                    artist.remove()
                except ValueError:
                    pass

        self._viewpoint_artists.clear()
        self._viewpoint_square_artists.clear()


        self._discarded_viewpoint_square_identifiers.clear()
   

        self._selected_viewpoint_identifier = None
        self.toolbar.set_viewpoint_clear_enabled(False)
        self.canvas.draw_idle()

class DynamicRasterView(RasterView):
    """
    Raster viewer that reloads the visible area when the user zooms or pans.
    """

    raster_load_failed = Signal(str)

    def __init__(
        self,
        empty_message: str = "No raster loaded",
        parent: QWidget | None = None,
        *,
        max_display_dimension: int = 2500,
        oversampling: float = 1.5,
    ) -> None:
        super().__init__(
            empty_message,
            parent,
            allow_area_selection=True,
            allow_viewpoint_selection=True,
        )

        self._raster_path: Path | None = None
        self._source_transform: Any | None = None
        self._source_crs: Any | None = None
        self._source_bounds: tuple[
            float,
            float,
            float,
            float,
        ] | None = None

        self._title = "Raster"
        self._colour_map = "viridis"
        self._colourbar_label = ""

        self._image = None

        self._max_display_dimension = max_display_dimension
        self._oversampling = oversampling

        self._thread_pool = QThreadPool.globalInstance()
        self._window_workers: set[FunctionWorker] = set()

        # Each new zoom request gets a higher number. Results from older
        # requests can then be ignored.
        self._latest_request_id = 0

        # Prevent programmatic axis changes from starting another read.
        self._suppress_limit_events = False

        # Wait until the user has paused zooming or panning.
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(180)
        self._reload_timer.timeout.connect(
            self._request_visible_window
        )

    @property
    def source_transform(self) -> Any | None:
        return self._source_transform

    @property
    def source_crs(self) -> Any | None:
        return self._source_crs

    def current_visible_bounds(
        self,
    ) -> tuple[float, float, float, float]:
        """Return the map bounds currently visible inside the DEM axes."""

        if self._axes is None or self._source_bounds is None:
            raise RuntimeError("A DEM must be displayed before reading its view.")

        x_start, x_end = self._axes.get_xlim()
        y_start, y_end = self._axes.get_ylim()

        source_left, source_bottom, source_right, source_top = (
            self._source_bounds
        )

        left = max(min(x_start, x_end), source_left)
        bottom = max(min(y_start, y_end), source_bottom)
        right = min(max(x_start, x_end), source_right)
        top = min(max(y_start, y_end), source_top)

        if left >= right or bottom >= top:
            raise RuntimeError("The current view does not overlap the DEM.")

        return left, bottom, right, top

    def open_raster(
        self,
        path: str | Path,
        *,
        title: str,
        colour_map: str,
        colourbar_label: str,
    ) -> None:
        """Open a raster and display a reduced full-extent preview."""

        raster_path = Path(path)

        with rasterio.open(raster_path) as source:
            if source.count < 1:
                raise ValueError("The raster contains no bands.")

            if source.crs is None:
                raise ValueError(
                    "The raster has no coordinate reference system."
                )

            self._raster_path = raster_path
            self._source_transform = source.transform
            self._source_crs = source.crs
            self._source_bounds = (
                source.bounds.left,
                source.bounds.bottom,
                source.bounds.right,
                source.bounds.top,
            )

        self._title = title
        self._colour_map = colour_map
        self._colourbar_label = colourbar_label

        target_width, target_height = self._target_display_size()

        initial_result = read_raster_window(
            path=raster_path,
            bounds=self._source_bounds,
            target_width=target_width,
            target_height=target_height,
            title=title,
            colour_map=colour_map,
            colourbar_label=colourbar_label,
        )

        self._draw_initial_raster(initial_result)

    def _target_display_size(self) -> tuple[int, int]:
        """
        Return a sensible raster-array size for the current canvas.

        Slight oversampling keeps the image sharp without reading millions
        of pixels that cannot appear on screen.
        """

        device_ratio = self.canvas.devicePixelRatioF()

        width = int(
            self.canvas.width()
            * device_ratio
            * self._oversampling
        )
        height = int(
            self.canvas.height()
            * device_ratio
            * self._oversampling
        )

        width = min(
            self._max_display_dimension,
            max(256, width),
        )
        height = min(
            self._max_display_dimension,
            max(256, height),
        )

        return width, height

    def _draw_initial_raster(
        self,
        result: RasterResult,
    ) -> None:
        """Create the initial Matplotlib image."""

        data = np.ma.masked_invalid(result.data)

        self.figure.clear()
        self._colourbar = None
        self._axes = self.figure.add_subplot(111)

        extent = plotting_extent(
            data,
            result.transform,
        )

        self._image = self._axes.imshow(
            data,
            origin="upper",
            extent=extent,
            cmap=result.colour_map,
            vmin=result.vmin,
            vmax=result.vmax,
        )

        self._axes.set_title(result.title)
        self._axes.set_xlabel("Easting / X")
        self._axes.set_ylabel("Northing / Y")
        self._axes.set_aspect("equal")

        self._colourbar = self.figure.colorbar(
            self._image,
            ax=self._axes,
        )

        if result.colourbar_label:
            self._colourbar.set_label(
                result.colourbar_label
            )

        # These run whenever toolbar zoom, toolbar pan or wheel zoom changes
        # the visible coordinates.
        self._axes.callbacks.connect(
            "xlim_changed",
            self._schedule_window_reload,
        )
        self._axes.callbacks.connect(
            "ylim_changed",
            self._schedule_window_reload,
        )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _schedule_window_reload(
        self,
        _axes: Any,
    ) -> None:
        """Schedule a detailed read after the view stops moving."""

        if self._suppress_limit_events:
            return

        if self._raster_path is None:
            return

        # Calling start again restarts the single-shot timer. Therefore
        # continuous panning does not launch hundreds of reads.
        self._reload_timer.start()

    def _request_visible_window(self) -> None:
        """Run a windowed Rasterio read in the thread pool."""

        if (
            self._raster_path is None
            or self._axes is None
            or self._source_bounds is None
        ):
            return

        visible_bounds = self.current_visible_bounds()

        target_width, target_height = self._target_display_size()

        self._latest_request_id += 1
        request_id = self._latest_request_id

        worker = FunctionWorker(
            read_raster_window,
            path=self._raster_path,
            bounds=visible_bounds,
            target_width=target_width,
            target_height=target_height,
            title=self._title,
            colour_map=self._colour_map,
            colourbar_label=self._colourbar_label,
        )

        self._window_workers.add(worker)

        worker.signals.result.connect(
            lambda result: self._apply_window_result(
                request_id,
                result,
            )
        )

        worker.signals.error.connect(
            lambda message, traceback_text: (
                self._handle_window_error(
                    request_id,
                    message,
                )
            )
        )

        worker.signals.finished.connect(
            lambda: self._window_workers.discard(worker)
        )

        self._thread_pool.start(worker)

    def _apply_window_result(
        self,
        request_id: int,
        result: RasterResult,
    ) -> None:
        """Replace the image with a more detailed visible window."""

        # The user may have zoomed again while this read was running.
        if request_id != self._latest_request_id:
            return

        if self._axes is None or self._image is None:
            return

        current_x_limits = self._axes.get_xlim()
        current_y_limits = self._axes.get_ylim()

        self._suppress_limit_events = True

        try:
            data = np.ma.masked_invalid(result.data)
            extent = plotting_extent(
                data,
                result.transform,
            )

            # Update the existing Matplotlib image. Do not clear the axes,
            # because doing so would also remove a selected-area polygon.
            self._image.set_data(data)
            self._image.set_extent(extent)

            # set_extent can alter the axes limits, so restore the user's
            # exact visible position.
            self._axes.set_xlim(current_x_limits)
            self._axes.set_ylim(current_y_limits)

            self.canvas.draw_idle()
        finally:
            self._suppress_limit_events = False

    def _handle_window_error(
        self,
        request_id: int,
        message: str,
    ) -> None:
        """Report an error only if it belongs to the newest request."""

        if request_id != self._latest_request_id:
            return

        self.raster_load_failed.emit(message)

    def set_area_selection_enabled(self, enabled: bool) -> None:
        """Enable or disable both area-selection toolbar actions."""

        self.toolbar.set_area_selection_enabled(enabled)
        self.toolbar.set_area_clear_enabled(
            enabled
            and (
                self._area_selector is not None
                or self._area_patch is not None
            )
        )

    def _toggle_area_selection(self, enabled: bool) -> None:
        """Start or stop polygon selection when the toolbar action changes."""

        if not enabled:
            self.cancel_area_selection()
            return

        try:
            # A new polygon always replaces the previous one immediately.
            self._remove_area_patch()
            self.select_area()
        except RuntimeError as error:
            # Undo the checked state without recursively calling this slot.
            self.toolbar.select_area_action.blockSignals(True)
            self.toolbar.select_area_action.setChecked(False)
            self.toolbar.select_area_action.blockSignals(False)
            self.area_selection_error.emit(str(error))

    def select_area(self) -> None:
        """Allow the user to draw one replacement polygon on the raster."""

        if self._axes is None or not self._axes.images:
            raise RuntimeError(
                "A raster must be displayed before selecting an area."
            )

        self.cancel_area_selection()
        self.toolbar.set_area_clear_enabled(True)

        self._area_selector = PolygonSelector(
            self._axes,
            self._complete_area_selection,
            useblit=True,
        )

        self.area_selection_started.emit()
        self.canvas.draw_idle()

    def _complete_area_selection(
        self,
        vertices: list[tuple[float, float]],
    ) -> None:
        """Finish the polygon and report its map coordinates."""

        points = [
            (float(x), float(y))
            for x, y in vertices
        ]

        if len(points) < 3:
            return

        # A GeoJSON polygon ring must finish at its starting point.
        if points[0] != points[-1]:
            points.append(points[0])

        self._remove_area_patch()

        self._roi_path = MatplotlibPath(points)

        self._area_patch = PolygonPatch(
            points,
            closed=True,
            fill=True,
            facecolor="#f05337",
            edgecolor="#111111",
            alpha=0.24,
            linewidth=2,
            zorder=5,
        )
        self._axes.add_patch(self._area_patch)

        self.cancel_area_selection()
        self.toolbar.select_area_action.setChecked(False)
        self.toolbar.set_area_clear_enabled(True)

        self.canvas.draw_idle()
        self.area_selected.emit(points)

    def cancel_area_selection(self) -> None:
        """Stop and remove the currently active polygon selector."""

        if self._area_selector is None:
            return

        selector = self._area_selector

        selector.set_active(False)
        selector.set_visible(False)
        selector.disconnect_events()

        self._area_selector = None
        self.canvas.draw_idle()

    def _remove_area_patch(self) -> bool:
        """Remove the completed polygon patch without changing GUI state."""

        if self._area_patch is None:
            return False

        try:
            self._area_patch.remove()
        except ValueError:
            # The axes may already have been cleared while loading another DEM.
            pass

        self._area_patch = None
        self._roi_path = None
        return True

    def clear_selected_area(self) -> None:
        """Cancel selection and clear the active survey polygon."""

        had_selection = (
            self._area_selector is not None
            or self._area_patch is not None
        )

        self.cancel_area_selection()
        self.toolbar.select_area_action.setChecked(False)
        self._remove_area_patch()
        self.toolbar.set_area_clear_enabled(False)

        if had_selection:
            self.canvas.draw_idle()
            self.area_selection_cleared.emit()
