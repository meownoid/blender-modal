"""Terminal output: colored human results by default, verbose logs on stderr, JSON on demand."""

from __future__ import annotations

import json
import sys
from decimal import Decimal, InvalidOperation
from typing import Any

from rich.console import Console, RenderableType
from rich.status import Status

_stdout = Console()
_stderr = Console(stderr=True)

_verbose = False
_status: Status | None = None


def configure(*, verbose: bool) -> None:
    global _verbose
    stop_status()
    _verbose = verbose


def log(command: str, message: str) -> None:
    global _status
    if _verbose:
        stop_status()
        _stderr.print(f"[dim]{command}:[/] {message}")
        return
    if not _stderr.is_terminal:
        return
    status = _status
    if status is None:
        status = Status(message, console=_stderr)
        status.start()
        _status = status
    else:
        status.update(message)


def stop_status() -> None:
    global _status
    if _status is not None:
        _status.stop()
        _status = None


def emit_json(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            print(json.dumps(item, default=str, sort_keys=True))
    else:
        print(json.dumps(value, default=str, indent=2, sort_keys=True))
    sys.stdout.flush()


def emit_human(renderable: RenderableType) -> None:
    stop_status()
    _stdout.print(renderable)
    _stdout.file.flush()


def short_id(identifier: str) -> str:
    """Abbreviate a scene or result ID for human-readable labels only."""
    return identifier[:12]


def cost(value: str) -> str:
    """Format monetary amounts while preserving billing status labels."""
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return value
    return f"${amount:.2f}" if amount.is_finite() else value


def size_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def status_style(status: str) -> str:
    return {
        "completed": "green",
        "running": "cyan",
        "queued": "yellow",
        "failed": "red",
        "cancelled": "yellow",
    }.get(status, "white")
