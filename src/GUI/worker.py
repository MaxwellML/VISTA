"""Reusable background worker for long-running Rivelero modules. Basically, all the QRunnable boilerplate."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """Signals emitted by :class:`FunctionWorker`."""

    result = Signal(object)
    error = Signal(str, str)
    finished = Signal()


class FunctionWorker(QRunnable):
    """
    Run an ordinary Python callable in Qt's global thread pool.

    The callable must not update Qt widgets directly. Its return value is
    emitted through ``signals.result`` and handled by the main GUI thread.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """Execute the callable and report its outcome through Qt signals."""
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception as error:  # noqa: BLE001 - worker must report all failures
            self.signals.error.emit(str(error), traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
