"""Continuous system resource monitoring for Rivelero."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread

import psutil


BYTES_PER_MIB = 1024 * 1024
MONITOR_INTERVAL_SECONDS = 5.0

_stop_event = Event()
_monitor_thread: Thread | None = None


def get_current_process() -> psutil.Process:
    """Return the currently running Rivelero process."""

    return psutil.Process()


def get_memory_usage_mib(
    process: psutil.Process | None = None,
) -> float:
    """Return the process's current RAM usage in MiB."""

    if process is None:
        process = get_current_process()

    memory_bytes = process.memory_info().rss

    return memory_bytes / BYTES_PER_MIB


def create_info_file() -> Path:
    """Create a new monitoring file for this Rivelero execution."""

    info_directory = Path(__file__).parent / "info"
    info_directory.mkdir(parents=True, exist_ok=True)

    execution_time = datetime.now(timezone.utc)
    timestamp = execution_time.strftime(
        "%Y-%m-%d_%H-%M-%S_%f_UTC"
    )

    file_path = (
        info_directory
        / f"rivelero_sysinfo_{timestamp}.txt"
    )

    with file_path.open(
        mode="x",
        encoding="utf-8",
    ) as info_file:
        info_file.write("Rivelero system information\n")
        info_file.write(
            f"Execution started: "
            f"{execution_time.isoformat()}\n"
        )
        info_file.write(f"Process ID: {psutil.Process().pid}\n")
        info_file.write("-" * 72 + "\n")

    return file_path


def format_resource_reading(
    process: psutil.Process,
) -> str:
    """Create one timestamped CPU and RAM reading."""

    current_time = datetime.now(timezone.utc)
    cpu_percent = process.cpu_percent(interval=None)
    memory_mib = get_memory_usage_mib(process)

    return (
        f"{current_time.isoformat()} | "
        f"CPU: {cpu_percent:.2f}% | "
        f"RAM: {memory_mib:.2f} MiB"
    )


def monitor_process(
    info_file_path: Path,
    interval_seconds: float,
) -> None:
    """Print and save process information repeatedly."""

    process = get_current_process()

    process.cpu_percent(interval=None)

    while not _stop_event.wait(interval_seconds):
        reading = format_resource_reading(process)

        print(reading)

        with info_file_path.open(
            mode="a",
            encoding="utf-8",
        ) as info_file:
            info_file.write(reading + "\n")


def start_process_monitor(
    interval_seconds: float = MONITOR_INTERVAL_SECONDS,
) -> Path:
    """Start continuous Rivelero process monitoring."""

    global _monitor_thread

    if _monitor_thread is not None:
        raise RuntimeError(
            "The process monitor is already running."
        )

    if interval_seconds <= 0:
        raise ValueError(
            "The monitoring interval must be positive."
        )

    info_file_path = create_info_file()

    _stop_event.clear()

    _monitor_thread = Thread(
        target=monitor_process,
        args=(info_file_path, interval_seconds),
        name="rivelero-process-monitor",
        daemon=True,
    )
    _monitor_thread.start()

    print(f"System monitor started: {info_file_path}")

    return info_file_path


def stop_process_monitor() -> None:
    """Stop continuous Rivelero process monitoring."""

    global _monitor_thread

    if _monitor_thread is None:
        return

    _stop_event.set()
    _monitor_thread.join()
    _monitor_thread = None

    print("System monitor stopped.")
