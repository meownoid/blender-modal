"""The immutable Volume catalog used by local CLI commands and GPU workers."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from threading import Event, Thread
from time import monotonic
from typing import Any

import modal

from .models import JobManifest, RenderSpec, SceneFile, SceneManifest, canonical_json, utc_now

DEFAULT_VOLUME = "blender-modal-v2"
ROOTS = ("blobs", "scenes", "results", "jobs", "staging")
_MISSING_PATH_ERRORS = (FileNotFoundError, modal.exception.NotFoundError)
ProgressReporter = Callable[[str], None]


class CatalogError(RuntimeError):
    """The command could not safely read or modify the render catalog."""


def blob_path(sha256: str) -> str:
    return f"blobs/sha256/{sha256}"


def scene_path(scene_id: str) -> str:
    return f"scenes/{scene_id}/manifest.json"


def result_manifest_path(result_id: str) -> str:
    return f"results/{result_id}/manifest.json"


def frame_path(result_id: str, frame: int, name: str) -> str:
    return f"results/{result_id}/frames/{frame}/{name}"


def job_path(job_id: str) -> str:
    return f"jobs/{job_id}/manifest.json"


def worker_path(job_id: str, worker_index: int) -> str:
    return f"jobs/{job_id}/workers/{worker_index}.json"


class Catalog:
    """A small wrapper around the public local Modal Volume API."""

    def __init__(self, volume_name: str = DEFAULT_VOLUME, environment: str | None = None):
        self.volume_name = volume_name
        self.environment = environment
        self.volume = modal.Volume.from_name(
            volume_name,
            environment_name=environment,
            create_if_missing=True,
            version=2,
        )

    def read_bytes(self, path: str) -> bytes:
        try:
            return b"".join(self.volume.read_file(path))
        except _MISSING_PATH_ERRORS as exc:
            raise CatalogError(f"Missing catalog item: {path}") from exc

    def read_json(self, path: str) -> dict[str, Any]:
        try:
            value = json.loads(self.read_bytes(path))
        except json.JSONDecodeError as exc:
            raise CatalogError(f"Invalid JSON in catalog item: {path}") from exc
        if not isinstance(value, dict):
            raise CatalogError(f"Catalog item is not an object: {path}")
        return value

    def exists(self, path: str) -> bool:
        try:
            entries = self.volume.listdir(path)
        except _MISSING_PATH_ERRORS:
            return False
        return bool(entries)

    def write_json(self, path: str, value: dict[str, Any], *, overwrite: bool = True) -> None:
        data = canonical_json(value).encode()
        with self.volume.batch_upload(force=overwrite) as upload:
            upload.put_file(io.BytesIO(data), path)

    def scene(self, scene_id: str) -> SceneManifest:
        return SceneManifest.from_dict(self.read_json(scene_path(scene_id)))

    def job(self, job_id: str) -> JobManifest:
        return JobManifest.from_dict(self.read_json(job_path(job_id)))

    def write_job(self, job: JobManifest) -> None:
        self.write_json(job_path(job.id), job.to_dict())

    def list_scenes(self) -> list[SceneManifest]:
        manifests: list[SceneManifest] = []
        for entry in self._files("scenes"):
            if entry.path.count("/") == 2 and entry.path.endswith("/manifest.json"):
                manifests.append(SceneManifest.from_dict(self.read_json(entry.path)))
        return sorted(manifests, key=lambda item: item.created_at, reverse=True)

    def list_results(self, scene_id: str | None = None) -> list[dict[str, Any]]:
        job_ids: dict[str, list[str]] = {}
        for job in self.list_jobs():
            job_ids.setdefault(job.result_id, []).append(job.id)
        values: list[dict[str, Any]] = []
        for entry in self._files("results"):
            if entry.path.count("/") == 2 and entry.path.endswith("/manifest.json"):
                value = self.read_json(entry.path)
                if scene_id is None or value.get("spec", {}).get("scene_id") == scene_id:
                    value["id"] = PurePosixPath(entry.path).parent.name
                    value["frames"] = self.completed_frames(value["id"])
                    value["jobs"] = job_ids.get(value["id"], [])
                    values.append(value)
        return sorted(values, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def list_jobs(self) -> list[JobManifest]:
        jobs: list[JobManifest] = []
        for entry in self._files("jobs"):
            if entry.path.count("/") == 2 and entry.path.endswith("/manifest.json"):
                jobs.append(JobManifest.from_dict(self.read_json(entry.path)))
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def completed_frames(self, result_id: str) -> tuple[int, ...]:
        frames: set[int] = set()
        for entry in self._files(f"results/{result_id}/frames"):
            if entry.path.endswith("/metadata.json"):
                try:
                    frame = int(PurePosixPath(entry.path).parent.name)
                    metadata = self.read_json(entry.path)
                except (CatalogError, ValueError):
                    continue
                if metadata.get("status") == "completed" and self.exists(
                    frame_path(result_id, frame, "frame.png")
                ):
                    frames.add(frame)
        return tuple(sorted(frames))

    def create_result_set(self, spec: RenderSpec) -> str:
        result_id = spec.id
        path = result_manifest_path(result_id)
        if not self.exists(path):
            self.write_json(
                path,
                {"schema_version": 1, "created_at": utc_now(), "spec": spec.to_dict()},
                overwrite=False,
            )
        return result_id

    def remove_scene(self, scene_id: str, *, dry_run: bool) -> list[str]:
        path = f"scenes/{scene_id}"
        self.scene(scene_id)
        return self._remove(path, dry_run=dry_run)

    def remove_results(
        self, result_id: str, frames: Iterable[int] | None, *, dry_run: bool
    ) -> list[str]:
        if frames is None:
            return self._remove(f"results/{result_id}", dry_run=dry_run)
        removed: list[str] = []
        for frame in frames:
            removed.extend(self._remove(f"results/{result_id}/frames/{frame}", dry_run=dry_run))
        return removed

    def cleanup(
        self,
        *,
        dry_run: bool,
        force: bool = False,
        progress: ProgressReporter | None = None,
    ) -> list[str]:
        """Remove unreferenced content, retaining fresh partial uploads unless forced."""
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        _report(progress, "Loading scene manifests")
        with _progress_heartbeat(progress, "Still loading scene manifests"):
            referenced = {file.sha256 for scene in self.list_scenes() for file in scene.files}
        remove: list[str] = []
        _report(progress, "Scanning content-addressed blobs")
        with _progress_heartbeat(progress, "Still scanning content-addressed blobs"):
            for entry in self._files("blobs/sha256"):
                if PurePosixPath(entry.path).name not in referenced and (
                    force or _older(entry.mtime, cutoff)
                ):
                    remove.append(entry.path)
        _report(progress, "Scanning staging files")
        with _progress_heartbeat(progress, "Still scanning staging files"):
            for entry in self._files("staging"):
                if force or _older(entry.mtime, cutoff):
                    remove.append(entry.path)
        if not dry_run:
            _report(progress, f"Removing {len(remove)} path(s)")
            for path in remove:
                with suppress(*_MISSING_PATH_ERRORS):
                    self.volume.remove_file(path)
        return remove

    def _remove(self, path: str, *, dry_run: bool) -> list[str]:
        entries = [entry.path for entry in self._files(path)]
        if not dry_run:
            try:
                self.volume.remove_file(path, recursive=True)
            except _MISSING_PATH_ERRORS:
                return []
        return entries

    def _files(self, path: str) -> list[Any]:
        try:
            return [
                entry
                for entry in self.volume.iterdir(path, recursive=True)
                if entry.type.name == "FILE"
            ]
        except _MISSING_PATH_ERRORS:
            return []


def build_scene(
    root: Path,
    blend: Path,
    includes: Iterable[Path],
    name: str | None,
    *,
    progress: ProgressReporter | None = None,
) -> tuple[SceneManifest, dict[str, Path]]:
    root = root.resolve()
    if not root.is_dir():
        raise CatalogError(f"Project root is not a directory: {root}")
    blend = _inside(root, blend)
    if blend.suffix.lower() != ".blend" or not blend.is_file():
        raise CatalogError("--blend must name a .blend file inside ROOT")
    selected = list(includes)
    _report(progress, "Scanning project files")
    with _progress_heartbeat(progress, "Still scanning project files"):
        sources = _project_files(root) if not selected else _included_files(root, selected)
    sources.add(blend)
    mapped: dict[str, Path] = {}
    files: list[SceneFile] = []
    total = len(sources)
    with _progress_heartbeat(progress, "Still hashing project files"):
        for index, source in enumerate(sorted(sources), start=1):
            relative = source.relative_to(root).as_posix()
            if _should_report_step(index, total):
                _report(progress, f"Hashing {index}/{total}: {relative}")
            sha256, size = sha256_file(source)
            mapped[relative] = source
            files.append(SceneFile(relative, sha256, size))
    return SceneManifest.create(blend.relative_to(root).as_posix(), files, name), mapped


def upload_scene(
    catalog: Catalog,
    manifest: SceneManifest,
    sources: dict[str, Path],
    *,
    progress: ProgressReporter | None = None,
) -> bool:
    """Upload content-addressed files and publish a complete immutable scene manifest."""
    _report(progress, f"Checking whether scene {manifest.id} already exists")
    with _progress_heartbeat(progress, "Still checking for an existing scene"):
        try:
            existing = catalog.scene(manifest.id)
        except CatalogError:
            existing = None
    if existing is not None:
        _report(progress, "Scene already exists; skipping upload")
        return False

    total_files = len(manifest.files)
    _report(progress, f"Verifying {total_files} file(s) are unchanged")
    with _progress_heartbeat(progress, "Still verifying source files"):
        for index, item in enumerate(manifest.files, start=1):
            if _should_report_step(index, total_files):
                _report(progress, f"Verifying {index}/{total_files}: {item.path}")
            current_hash, current_size = sha256_file(sources[item.path])
            if (current_hash, current_size) != (item.sha256, item.size):
                raise CatalogError(f"File changed before upload: {item.path}")

    _report(progress, f"Checking {total_files} blob(s) already in the Volume")
    pending: list[SceneFile] = []
    uploaded_hashes: set[str] = set()
    with _progress_heartbeat(progress, "Still checking existing blobs"):
        for index, item in enumerate(manifest.files, start=1):
            if _should_report_step(index, total_files):
                _report(progress, f"Checking blob {index}/{total_files}: {item.path}")
            if item.sha256 not in uploaded_hashes and not catalog.exists(blob_path(item.sha256)):
                pending.append(item)
            uploaded_hashes.add(item.sha256)

    if pending:
        upload_size = sum(item.size for item in pending)
        _report(
            progress,
            f"Uploading {len(pending)} new blob(s) ({_format_size(upload_size)})",
        )
        with (
            _progress_heartbeat(progress, "Still uploading blobs"),
            catalog.volume.batch_upload() as upload,
        ):
            for index, item in enumerate(pending, start=1):
                if _should_report_step(index, len(pending)):
                    _report(progress, f"Queueing blob {index}/{len(pending)}: {item.path}")
                upload.put_file(sources[item.path], blob_path(item.sha256))
            _report(progress, "Committing blob upload")
        _report(progress, "Blob upload complete")
    else:
        _report(progress, "All blobs are already in the Volume")

    # Build a ready-to-mount tree using server-side copies. Placeholders make every
    # destination directory exist on both Volume implementations before copying.
    placeholder_paths = sorted(
        {
            str(PurePosixPath(f"scenes/{manifest.id}/source/{item.path}").parent)
            for item in manifest.files
        }
    )
    _report(progress, f"Creating {len(placeholder_paths)} scene directory marker(s)")
    with (
        _progress_heartbeat(progress, "Still creating scene directory markers"),
        catalog.volume.batch_upload() as upload,
    ):
        for directory in placeholder_paths:
            upload.put_file(io.BytesIO(b""), f"{directory}/.blender-modal-dir")
    _report(progress, f"Materializing {total_files} scene file(s)")
    with _progress_heartbeat(progress, "Still materializing scene files"):
        for index, item in enumerate(manifest.files, start=1):
            if _should_report_step(index, total_files):
                _report(progress, f"Materializing {index}/{total_files}: {item.path}")
            destination = f"scenes/{manifest.id}/source/{item.path}"
            catalog.volume.copy_files([blob_path(item.sha256)], destination)
    for directory in placeholder_paths:
        with suppress(*_MISSING_PATH_ERRORS):
            catalog.volume.remove_file(f"{directory}/.blender-modal-dir")
    _report(progress, "Publishing scene manifest")
    with _progress_heartbeat(progress, "Still publishing scene manifest"):
        catalog.write_json(scene_path(manifest.id), manifest.to_dict(), overwrite=False)
    _report(progress, "Scene upload complete")
    return True


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _should_report_step(index: int, total: int) -> bool:
    return total <= 10 or index == 1 or index == total or index % 25 == 0


def _format_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


@contextmanager
def _progress_heartbeat(
    progress: ProgressReporter | None, status: str, interval_seconds: float = 30
) -> Iterator[None]:
    if progress is None:
        yield
        return

    started_at = monotonic()
    completed = Event()

    def report_heartbeat() -> None:
        while not completed.wait(interval_seconds):
            elapsed_seconds = int(monotonic() - started_at)
            _report(progress, f"{status} ({elapsed_seconds}s elapsed)")

    heartbeat = Thread(target=report_heartbeat, daemon=True)
    heartbeat.start()
    try:
        yield
    finally:
        completed.set()
        heartbeat.join()


def sha256_file(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise CatalogError(f"Only regular files are supported: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise CatalogError(f"File changed while hashing: {path}")
    return digest.hexdigest(), after.st_size


def _project_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directories:
            path = current / name
            if path.is_symlink():
                raise CatalogError(f"Symlinks are not supported: {path}")
        for filename in filenames:
            path = current / filename
            if path.is_symlink():
                raise CatalogError(f"Symlinks are not supported: {path}")
            files.add(path)
    return files


def _included_files(root: Path, includes: Iterable[Path]) -> set[Path]:
    files: set[Path] = set()
    for include in includes:
        path = _inside(root, include)
        if path.is_symlink():
            raise CatalogError(f"Symlinks are not supported: {path}")
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(_project_files(path))
        else:
            raise CatalogError(f"Included path does not exist: {path}")
    return files


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise CatalogError(f"Path is outside project root: {path}")
    return resolved


def _older(timestamp: int, cutoff: datetime) -> bool:
    return datetime.fromtimestamp(timestamp, UTC) < cutoff
