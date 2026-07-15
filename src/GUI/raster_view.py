"""Matplotlib-based raster display widget for PySide6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
) -> RasterResult:
    """Read the first band of a raster file into a display result."""
    raster_path = Path(path)

    with rasterio.open(raster_path) as source:
        data = source.read(1).astype(np.float64)

        if source.nodata is not None:
            data = np.ma.masked_equal(data, source.nodata)

        data = np.ma.masked_invalid(data)
        data = np.ma.masked_where(np.abs(data) > 1e20, data)

        return RasterResult(
            data=data,
            transform=source.transform,
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


class RasterView(QWidget):
    """A reusable Matplotlib raster viewer with toolbar and wheel zoom."""

    def __init__(
        self,
        empty_message: str = "No raster loaded",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self._axes = None
        self._colourbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
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
        self.canvas.draw_idle()

    def _on_scroll(self, event: Any) -> None:
        """Zoom around the mouse cursor when no toolbar tool is active."""
        if self._axes is None or event.inaxes is not self._axes:
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
