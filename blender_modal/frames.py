"""Frame selection and inexpensive PNG validation shared by CLI and workers."""

from __future__ import annotations

import struct
from pathlib import Path


def parse_frames(expression: str) -> tuple[int, ...]:
    frames: set[int] = set()
    for part in (item.strip() for item in expression.split(",")):
        if not part:
            raise ValueError("Frame selection contains an empty item")
        values = part.split(":")
        try:
            if len(values) == 1:
                frames.add(int(values[0]))
                continue
            if len(values) not in {2, 3}:
                raise ValueError
            start, end = int(values[0]), int(values[1])
            step = int(values[2]) if len(values) == 3 else 1
        except ValueError as exc:
            raise ValueError(f"Invalid frame selection item: {part}") from exc
        if step <= 0 or end < start:
            raise ValueError(f"Invalid inclusive frame range: {part}")
        frames.update(range(start, end + 1, step))
    if not frames:
        raise ValueError("At least one frame is required")
    return tuple(sorted(frames))


def png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as source:
            header = source.read(24)
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return (width, height) if width > 0 and height > 0 else None
