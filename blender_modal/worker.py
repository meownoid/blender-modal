"""The only remote workload: GPU Cycles rendering against a mounted Volume."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import modal

from .catalog import DEFAULT_VOLUME, worker_path
from .frames import png_dimensions
from .models import RenderSpec, utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOLUME_MOUNT = "/data"
app = modal.App("blender-modal")
volume = modal.Volume.from_name(DEFAULT_VOLUME, create_if_missing=True, version=2)
render_image = modal.Image.from_dockerfile(PROJECT_ROOT / "Dockerfile")


@app.function(
    image=render_image,
    volumes={VOLUME_MOUNT: volume},
    min_containers=0,
    timeout=12 * 60 * 60,
    ephemeral_disk=512 * 1024,
    single_use_containers=True,
)
def render_shard(
    volume_name: str,
    job_id: str,
    result_id: str,
    worker_index: int,
    frames: list[int],
    spec_payload: dict[str, Any],
) -> dict[str, Any]:
    """Render a disjoint frame shard and atomically publish every verified PNG."""
    mounted_volume = modal.Volume.from_name(volume_name, version=2)
    mounted_volume.reload()
    spec = RenderSpec.from_dict(spec_payload)
    status_path = Path(VOLUME_MOUNT) / worker_path(job_id, worker_index)
    _write_json(status_path, {"status": "running", "frames": frames, "started_at": utc_now()})
    mounted_volume.commit()
    completed: list[int] = []
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        scene = _read_json(Path(VOLUME_MOUNT) / f"scenes/{spec.scene_id}/manifest.json")
        with tempfile.TemporaryDirectory(prefix=f"blender-modal-{job_id[:8]}-") as scratch_name:
            scratch = Path(scratch_name)
            project_root = scratch / "project"
            _materialize_scene(scene, project_root)
            entrypoint = project_root / str(scene["entrypoint"])
            output_dir = scratch / "outputs"
            output_dir.mkdir()
            config = {
                "backend": spec.backend,
                "frames": frames,
                "output_dir": str(output_dir),
                "project_root": str(project_root),
            }
            for key in (
                "samples",
                "tile_size",
                "resolution_x",
                "resolution_y",
                "resolution_percentage",
            ):
                value = getattr(spec, key)
                if value is not None:
                    config[key] = value
            config_path = scratch / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            process = subprocess.Popen(
                _blender_command(entrypoint, config_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                event = _render_event(line)
                if event and event.get("type") == "frame_completed":
                    frame = int(event["frame"])
                    source = output_dir / f"frame_{frame:06d}.png"
                    if png_dimensions(source) is None:
                        raise RuntimeError(f"Blender reported an invalid PNG for frame {frame}")
                    _publish_frame(
                        mounted_volume,
                        result_id,
                        frame,
                        source,
                        {
                            "status": "completed",
                            "frame": frame,
                            "job_id": job_id,
                            "worker_index": worker_index,
                            "backend": spec.backend,
                            "render_seconds": float(event.get("seconds", 0)),
                            "completed_at": str(event.get("completed_at") or utc_now()),
                            "sha256": _sha256(source),
                        },
                    )
                    completed.append(frame)
                    _write_json(
                        status_path,
                        {
                            "status": "running",
                            "frames": frames,
                            "completed_frames": completed,
                            "updated_at": utc_now(),
                        },
                    )
                    mounted_volume.commit()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Blender exited with status {return_code}")
        status = {
            "status": "completed",
            "frames": frames,
            "completed_frames": completed,
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(status_path, status)
        mounted_volume.commit()
        return status
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
        status = {
            "status": "failed",
            "frames": frames,
            "completed_frames": completed,
            "error": str(exc)[:1000],
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(status_path, status)
        mounted_volume.commit()
        raise


def _materialize_scene(scene: dict[str, Any], project_root: Path) -> None:
    for item in scene["files"]:
        relative = Path(str(item["path"]))
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(VOLUME_MOUNT) / "blobs" / "sha256" / str(item["sha256"])
        shutil.copyfile(source, destination)
        if _sha256(destination) != item["sha256"]:
            raise RuntimeError(f"Scene content hash mismatch: {relative}")


def _blender_command(scene: Path, config: Path) -> list[str]:
    command = [
        "/opt/blender/blender",
        "--background",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
    ]
    if os.getenv("FLIP_FLUIDS_ADDON"):
        command.extend(["--python", "/app/renderer/blendrender_enable_flip_fluids.py"])
    return [
        *command,
        str(scene),
        "--python",
        "/app/renderer/blendrender_render.py",
        "--",
        str(config),
    ]


def _publish_frame(
    mounted_volume: modal.Volume,
    result_id: str,
    frame: int,
    source: Path,
    metadata: dict[str, Any],
) -> None:
    root = Path(VOLUME_MOUNT) / f"results/{result_id}/frames/{frame}"
    destination = root / "frame.png"
    marker = root / "metadata.json"
    if marker.is_file() and destination.is_file():
        return
    staging = Path(VOLUME_MOUNT) / f"staging/{result_id}-{frame}-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, staging / "frame.png")
    _write_json(staging / "metadata.json", metadata)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staging, root)
    except FileExistsError:
        shutil.rmtree(staging, ignore_errors=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object at {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _render_event(line: str) -> dict[str, Any] | None:
    if not line.startswith("BR "):
        return None
    try:
        value = json.loads(line[3:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
