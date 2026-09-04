"""Executed inside Blender to configure and render a BlendRender job."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import bpy

PREFIX = "BR "
SAMPLE_PATTERN = re.compile(r"Sample\s+(\d+)\s*/\s*(\d+)")
REMAINING_PATTERN = re.compile(r"Remaining:\s+(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")


def emit(event_type: str, **payload: object) -> None:
    print(PREFIX + json.dumps({"type": event_type, **payload}), flush=True)


def config_path() -> Path:
    if "--" not in sys.argv:
        raise RuntimeError("Missing BlendRender render configuration")
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if len(arguments) != 1:
        raise RuntimeError("Expected exactly one render configuration path")
    return Path(arguments[0]).resolve()


def bundled_flip_material_library() -> Path | None:
    module = os.getenv("FLIP_FLUIDS_ADDON", "").strip()
    if not module:
        return None
    try:
        material_library = importlib.import_module(f"{module}.materials.material_library")
    except ModuleNotFoundError:
        return None
    library_file = getattr(material_library, "__file__", None)
    if not library_file:
        return None
    return (Path(library_file).resolve().parent / "material_library").resolve()


def unavailable_external_assets(project_root: Path) -> list[str]:
    unavailable: list[str] = []
    material_library = bundled_flip_material_library()

    def validate(kind: str, filepath: str) -> None:
        if not filepath.startswith("//"):
            unavailable.append(f"{kind} uses an absolute path: {filepath}")
            return
        path = Path(bpy.path.abspath(filepath)).resolve()
        if not path.is_relative_to(project_root):
            unavailable.append(f"{kind} is outside the uploaded project: {filepath}")
        elif not path.is_file():
            unavailable.append(f"missing {kind}: {filepath}")

    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file and image.filepath:
            validate("image", image.filepath)
    for library in bpy.data.libraries:
        if library.filepath:
            library_path = Path(bpy.path.abspath(library.filepath)).resolve()
            if material_library is not None and library_path.is_relative_to(material_library):
                continue
            validate("linked library", library.filepath)
    for sound in bpy.data.sounds:
        if not sound.packed_file and sound.filepath:
            validate("sound", sound.filepath)
    return sorted(set(unavailable))


def enable_backend(backend: str) -> list[str]:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = backend
    preferences.get_devices()
    enabled: list[str] = []
    for device in preferences.devices:
        device.use = device.type == backend
        if device.use:
            enabled.append(device.name)
    if not enabled:
        raise RuntimeError(f"No {backend} render device is available")
    return enabled


def emit_render_progress(frame: int, stats: str) -> None:
    payload: dict[str, object] = {"frame": frame}
    if sample_match := SAMPLE_PATTERN.search(stats):
        payload["sample_current"] = int(sample_match.group(1))
        payload["sample_total"] = int(sample_match.group(2))
    if remaining_match := REMAINING_PATTERN.search(stats):
        hours = int(remaining_match.group(1) or 0)
        minutes = int(remaining_match.group(2))
        seconds = float(remaining_match.group(3))
        payload["remaining_seconds"] = hours * 3600 + minutes * 60 + seconds
    if len(payload) > 1:
        emit("frame_progress", **payload)


def run() -> None:
    config = json.loads(config_path().read_text(encoding="utf-8"))
    backend = str(config["backend"])
    frames = [int(frame) for frame in config["frames"]]
    output_dir = Path(config["output_dir"]).resolve()
    project_root = Path(str(config.get("project_root", Path(bpy.data.filepath).parent))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError("The active scene does not have an active camera")
    unavailable = unavailable_external_assets(project_root)
    if unavailable:
        sample = ", ".join(unavailable[:5])
        suffix = "" if len(unavailable) <= 5 else f" and {len(unavailable) - 5} more"
        raise RuntimeError(f"Project contains unavailable unpacked assets: {sample}{suffix}")

    cpu_render = backend == "CPU"
    devices = ["CPU"] if cpu_render else enable_backend(backend)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU" if cpu_render else "GPU"
    if (samples := config.get("samples")) is not None:
        scene.cycles.samples = int(samples)
    if (tile_size := config.get("tile_size")) is not None:
        scene.cycles.tile_size = int(tile_size)
    if (resolution_x := config.get("resolution_x")) is not None:
        scene.render.resolution_x = int(resolution_x)
    if (resolution_y := config.get("resolution_y")) is not None:
        scene.render.resolution_y = int(resolution_y)
    if (resolution_percentage := config.get("resolution_percentage")) is not None:
        scene.render.resolution_percentage = int(resolution_percentage)
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_file_extension = True
    emit(
        "job_started",
        backend=backend,
        devices=devices,
        samples=scene.cycles.samples,
        frame_count=len(frames),
    )

    for frame in frames:
        started = time.monotonic()
        scene.frame_set(frame)
        scene.render.filepath = str(output_dir / f"frame_{frame:06d}.png")
        emit("frame_started", frame=frame)

        def render_stats_handler(stats: str, _: object, *, rendered_frame: int = frame) -> None:
            emit_render_progress(rendered_frame, stats)

        bpy.app.handlers.render_stats.append(render_stats_handler)
        try:
            bpy.ops.render.render(write_still=True)
        finally:
            bpy.app.handlers.render_stats.remove(render_stats_handler)
        emit(
            "frame_completed",
            frame=frame,
            seconds=time.monotonic() - started,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        )
    if report_path := config.get("diagnostic_report"):
        Path(str(report_path)).write_text(
            json.dumps(
                {
                    "camera": scene.camera.name,
                    "resolution_x": scene.render.resolution_x,
                    "resolution_y": scene.render.resolution_y,
                    "resolution_percentage": scene.render.resolution_percentage,
                    "samples": scene.cycles.samples,
                    "tile_size": scene.cycles.tile_size,
                    "denoising": scene.cycles.use_denoising,
                    "view_transform": scene.view_settings.look,
                    "compositor_nodes": scene.use_nodes,
                    "engine": scene.render.engine,
                    "file_format": scene.render.image_settings.file_format,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    emit("job_completed", rendered_frames=frames)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        emit("error", message=str(exc))
        traceback.print_exc()
        raise
