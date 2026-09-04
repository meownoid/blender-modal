"""Command-line interface for immutable Modal Blender render workspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import modal

from .catalog import Catalog, CatalogError, frame_path, result_manifest_path
from .frames import parse_frames
from .models import JobManifest, RenderSpec, utc_now


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _commands()[args.command](args)
    except (CatalogError, ValueError, modal.exception.Error) as exc:
        parser.error(str(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blender-modal")
    parser.add_argument("--volume", default="blender-modal-v2", help="Modal Volume name")
    parser.add_argument("--environment", help="Modal environment name")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="machine-readable output"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("upload", help="upload an immutable project tree")
    upload.add_argument(
        "root",
        nargs="?",
        type=Path,
        help="project root, or a .blend file when --blend is omitted",
    )
    upload.add_argument("--blend", type=Path, help="entrypoint .blend file")
    upload.add_argument("--include", action="append", type=Path, default=[])
    upload.add_argument("--name")

    listed = commands.add_parser("list", help="list catalog resources")
    list_commands = listed.add_subparsers(dest="list_command", required=True)
    list_commands.add_parser("scenes")
    results = list_commands.add_parser("results")
    results.add_argument("--scene")

    render = commands.add_parser("render", help="render missing frames on Modal GPUs")
    render.add_argument("scene")
    render.add_argument("--frames", required=True)
    render.add_argument("--gpu", default="T4")
    render.add_argument("--gpus-per-instance", type=_positive, default=1)
    render.add_argument("--instances", type=_positive, default=1)
    render.add_argument("--backend", choices=("OPTIX", "CUDA"), default="OPTIX")
    render.add_argument("--samples", type=_positive)
    render.add_argument("--tile-size", type=_positive)
    render.add_argument("--resolution-x", type=_positive)
    render.add_argument("--resolution-y", type=_positive)
    render.add_argument("--resolution-percentage", type=_positive)
    render.add_argument("--detach", action="store_true")

    download = commands.add_parser("download", help="download completed PNGs")
    download.add_argument("results")
    download.add_argument("--output", required=True, type=Path)
    download.add_argument("--frames")
    download.add_argument("--overwrite", action="store_true")

    remove = commands.add_parser("remove", help="delete selected catalog resources")
    remove_commands = remove.add_subparsers(dest="remove_command", required=True)
    scene = remove_commands.add_parser("scene")
    scene.add_argument("scene")
    scene.add_argument("--dry-run", action="store_true")
    result = remove_commands.add_parser("results")
    result.add_argument("results")
    result.add_argument("--frames")
    result.add_argument("--dry-run", action="store_true")

    cleanup = commands.add_parser("cleanup", help="remove abandoned staging and unreferenced blobs")
    cleanup.add_argument("--dry-run", action="store_true")
    cleanup.add_argument(
        "--force",
        action="store_true",
        help="also remove fresh unreferenced blobs from interrupted uploads",
    )

    info = commands.add_parser("info", help="show job state and Modal billing")
    info.add_argument("job", nargs="?")
    info.add_argument("--watch", action="store_true")

    cancel = commands.add_parser("cancel", help="cancel a submitted render")
    cancel.add_argument("job")
    return parser


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    return {
        "upload": _upload,
        "list": _list,
        "render": _render,
        "download": _download,
        "remove": _remove,
        "cleanup": _cleanup,
        "info": _info,
        "cancel": _cancel,
    }


def _catalog(args: argparse.Namespace) -> Catalog:
    return Catalog(args.volume, args.environment)


def _upload(args: argparse.Namespace) -> None:
    from .catalog import build_scene, upload_scene

    progress = _command_progress("upload")
    root, blend = _upload_root_and_blend(args)
    includes = _upload_includes(args, root, blend)
    progress(f"Preparing upload from {root}")
    manifest, files = build_scene(root, blend, includes, args.name, progress=progress)
    progress(f"Prepared {len(files)} file(s) for scene {manifest.id}")
    uploaded = upload_scene(_catalog(args), manifest, files, progress=progress)
    _emit(args, {"scene": manifest.to_dict(), "uploaded": uploaded})


def _upload_root_and_blend(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.blend is None:
        if args.root is None:
            raise ValueError("upload requires a .blend file or ROOT with --blend")
        blend = args.root.resolve()
        return blend.parent, blend

    if args.root is None:
        blend = args.blend.resolve()
        return blend.parent, blend

    root = args.root.resolve()
    blend = args.blend if args.blend.is_absolute() else root / args.blend
    return root, blend


def _upload_includes(args: argparse.Namespace, root: Path, blend: Path) -> list[Path]:
    includes = [path if path.is_absolute() else root / path for path in args.include]
    if args.root is None or args.blend is None:
        includes.append(blend)
    return includes


def _list(args: argparse.Namespace) -> None:
    _log("list", f"Loading {args.list_command}")
    catalog = _catalog(args)
    values: Any = (
        catalog.list_scenes() if args.list_command == "scenes" else catalog.list_results(args.scene)
    )
    if args.list_command == "scenes":
        values = [
            {
                "id": scene.id,
                "name": scene.name,
                "entrypoint": scene.entrypoint,
                "files": len(scene.files),
                "size_bytes": sum(file.size for file in scene.files),
                "created_at": scene.created_at,
            }
            for scene in values
        ]
    _log("list", f"Found {len(values)} {args.list_command}")
    _emit(args, values)


def _render(args: argparse.Namespace) -> None:
    _log("render", f"Loading scene {args.scene}")
    catalog = _catalog(args)
    catalog.scene(args.scene)
    _validate_render_overrides(args)
    requested = parse_frames(args.frames)
    _log("render", "Calculating renderer fingerprint")
    spec = RenderSpec(
        scene_id=args.scene,
        backend=args.backend,
        samples=args.samples,
        tile_size=args.tile_size,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        resolution_percentage=args.resolution_percentage,
        renderer_fingerprint=_renderer_fingerprint(),
    )
    _log("render", "Preparing result set")
    result_id = catalog.create_result_set(spec)
    _log("render", "Checking completed frames")
    existing = set(catalog.completed_frames(result_id))
    missing = tuple(frame for frame in requested if frame not in existing)
    if not missing:
        _log("render", f"All {len(requested)} requested frame(s) are already complete")
        _emit(
            args,
            {
                "job": None,
                "results": result_id,
                "cached_frames": list(requested),
                "submitted_frames": [],
            },
        )
        return
    _log("render", f"{len(existing)} cached frame(s), {len(missing)} frame(s) to render")
    _log("render", "Checking for an active matching render")
    _ensure_no_active_render(catalog, result_id)
    job = JobManifest(
        id=str(uuid.uuid4()),
        result_id=result_id,
        spec=spec,
        requested_frames=missing,
        gpu=args.gpu,
        gpus_per_instance=args.gpus_per_instance,
        instances=args.instances,
        created_at=utc_now(),
        status="running",
    )
    _log("render", f"Creating render job {job.id}")
    catalog.write_job(job)
    from .worker import app, render_shard

    shards = _split(missing, min(args.instances, len(missing)))
    _log("render", f"Submitting {len(shards)} render shard(s)")
    calls: list[modal.FunctionCall[Any]] = []
    with app.run(
        name=f"blender-modal-{job.id}",
        detach=args.detach,
        environment_name=args.environment,
    ):
        function = render_shard.with_options(
            gpu=_gpu_spec(args.gpu, args.gpus_per_instance),
            max_containers=len(shards),
            volumes={
                "/data": modal.Volume.from_name(
                    args.volume,
                    environment_name=args.environment,
                    version=2,
                )
            },
        )
        for index, shard in enumerate(shards):
            calls.append(
                function.spawn(args.volume, job.id, result_id, index, list(shard), spec.to_dict())
            )
        job = replace(
            job,
            app_id=getattr(app, "app_id", None),
            call_ids=tuple(call.object_id for call in calls),
        )
        catalog.write_job(job)
        _log("render", f"Submitted {len(calls)} render shard(s)")
        _emit(
            args,
            {
                "job": job.to_dict(),
                "results": result_id,
                "cached_frames": sorted(existing),
                "submitted_frames": list(missing),
            },
        )
        if not args.detach:
            failures: list[str] = []
            for index, call in enumerate(calls, start=1):
                try:
                    _log("render", f"Waiting for shard {index}/{len(calls)} ({call.object_id})")
                    call.get()
                except (
                    Exception
                ) as exc:  # Modal preserves worker failure details in the job record.
                    failures.append(str(exc))
            _log("render", "Refreshing final job status")
            _refresh_job(catalog, job.id)
            if failures:
                raise CatalogError("One or more render shards failed; use info JOB for details")
            _log("render", "Render completed")
        else:
            _log("render", "Render detached; use info to monitor the job")


def _download(args: argparse.Namespace) -> None:
    _log("download", f"Loading results {args.results}")
    catalog = _catalog(args)
    catalog.read_json(result_manifest_path(args.results))
    frames = parse_frames(args.frames) if args.frames else catalog.completed_frames(args.results)
    _log("download", f"Preparing {len(frames)} frame(s) in {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    downloaded: list[int] = []
    skipped: list[int] = []
    for index, frame in enumerate(frames, start=1):
        if _should_log_step(index, len(frames)):
            _log("download", f"Processing frame {index}/{len(frames)} ({frame})")
        metadata = catalog.read_json(frame_path(args.results, frame, "metadata.json"))
        target = args.output / f"frame_{frame:06d}.png"
        if target.is_file() and _file_hash(target) == metadata.get("sha256"):
            skipped.append(frame)
            continue
        if target.exists() and not args.overwrite:
            raise CatalogError(f"Refusing to overwrite {target}; pass --overwrite")
        temporary = target.with_name(f".{target.name}.partial")
        with temporary.open("wb") as output:
            for chunk in catalog.volume.read_file(frame_path(args.results, frame, "frame.png")):
                output.write(chunk)
        if _file_hash(temporary) != metadata.get("sha256"):
            temporary.unlink(missing_ok=True)
            raise CatalogError(f"Downloaded checksum does not match for frame {frame}")
        temporary.replace(target)
        downloaded.append(frame)
    _log("download", f"Finished: {len(downloaded)} downloaded, {len(skipped)} already present")
    _emit(
        args, {"results": args.results, "downloaded_frames": downloaded, "skipped_frames": skipped}
    )


def _remove(args: argparse.Namespace) -> None:
    catalog = _catalog(args)
    if args.remove_command == "scene":
        _log("remove", f"Checking scene {args.scene} is not rendering")
        _ensure_no_active_scene(catalog, args.scene)
        _log("remove", f"{'Previewing' if args.dry_run else 'Removing'} scene {args.scene}")
        removed = catalog.remove_scene(args.scene, dry_run=args.dry_run)
    else:
        _log("remove", f"Checking results {args.results} are not rendering")
        _ensure_no_active_render(catalog, args.results)
        frames = parse_frames(args.frames) if args.frames else None
        _log("remove", f"{'Previewing' if args.dry_run else 'Removing'} results {args.results}")
        removed = catalog.remove_results(args.results, frames, dry_run=args.dry_run)
    _log("remove", f"Selected {len(removed)} path(s)")
    _emit(args, {"dry_run": args.dry_run, "removed": removed})


def _cleanup(args: argparse.Namespace) -> None:
    progress = _command_progress("cleanup")
    catalog = _catalog(args)
    progress("Checking for active renders")
    if any(_refresh_job(catalog, job.id).status == "running" for job in catalog.list_jobs()):
        raise CatalogError("cleanup is unavailable while renders are active")
    if args.force:
        progress("Force mode will include fresh unreferenced upload blobs")
    progress("Scanning removable catalog content")
    removed = catalog.cleanup(dry_run=args.dry_run, force=args.force, progress=progress)
    progress(f"Selected {len(removed)} path(s)")
    _emit(
        args,
        {
            "dry_run": args.dry_run,
            "removed": removed,
        },
    )


def _info(args: argparse.Namespace) -> None:
    catalog = _catalog(args)
    if args.job:
        _log("info", f"Loading job {args.job}")
        job = _refresh_job(catalog, args.job)
        _log("info", "Loading worker status and billing")
        payload: dict[str, Any] = {
            "job": job.to_dict(),
            "workers": _worker_statuses(catalog, job),
            "billing": _job_billing(catalog, job),
        }
        if args.watch:
            _log("info", "Watching for job updates")
            _watch(catalog, job.id, args, payload)
            return
        _emit(args, payload)
        return
    _log("info", "Loading jobs and workspace billing")
    rates, summary, report_error = _billing()
    _emit(
        args,
        {
            "jobs": [job.to_dict() for job in catalog.list_jobs()],
            "billing": {
                "rates": rates,
                "summary": summary,
                "reported_costs": "pending",
                "error": report_error,
            },
        },
    )


def _cancel(args: argparse.Namespace) -> None:
    _log("cancel", f"Loading job {args.job}")
    catalog = _catalog(args)
    job = _refresh_job(catalog, args.job)
    if job.status != "running":
        raise CatalogError(f"Job {job.id} is not active")
    _log("cancel", "Recording cancellation")
    catalog.write_job(replace(job, cancelled=True, status="cancelled", completed_at=utc_now()))
    errors: list[str] = []
    for index, call_id in enumerate(job.call_ids, start=1):
        try:
            _log("cancel", f"Cancelling shard {index}/{len(job.call_ids)}")
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        except Exception as exc:
            errors.append(str(exc))
    _log("cancel", f"Cancellation complete with {len(errors)} error(s)")
    _emit(args, {"job": job.id, "cancelled": True, "errors": errors})


def _refresh_job(catalog: Catalog, job_id: str) -> JobManifest:
    job = catalog.job(job_id)
    if job.status != "running":
        return job
    statuses = _worker_statuses(catalog, job)
    expected_workers = min(job.instances, len(job.requested_frames))
    if len(statuses) < expected_workers:
        return job
    states = {str(status.get("status")) for status in statuses}
    completed_frames = {
        int(frame) for status in statuses for frame in status.get("completed_frames", [])
    }
    if states <= {"completed"} and set(job.requested_frames) <= completed_frames:
        updated = replace(job, status="completed", completed_at=utc_now())
    elif states <= {"completed"} or "failed" in states:
        updated = replace(job, status="failed", completed_at=utc_now())
    else:
        return job
    catalog.write_job(updated)
    return updated


def _worker_statuses(catalog: Catalog, job: JobManifest) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for index in range(min(job.instances, len(job.requested_frames))):
        try:
            statuses.append(catalog.read_json(f"jobs/{job.id}/workers/{index}.json"))
        except CatalogError:
            continue
    return statuses


def _ensure_no_active_render(catalog: Catalog, result_id: str) -> None:
    for job in catalog.list_jobs():
        refreshed = _refresh_job(catalog, job.id)
        if refreshed.result_id == result_id and refreshed.status == "running":
            raise CatalogError(f"A matching render is already active: {refreshed.id}")


def _ensure_no_active_scene(catalog: Catalog, scene_id: str) -> None:
    for job in catalog.list_jobs():
        refreshed = _refresh_job(catalog, job.id)
        if refreshed.spec.scene_id == scene_id and refreshed.status == "running":
            raise CatalogError(f"Scene has an active render: {refreshed.id}")


def _watch(
    catalog: Catalog, job_id: str, args: argparse.Namespace, payload: dict[str, Any]
) -> None:
    import time

    while payload["job"]["status"] == "running":
        _emit(args, payload)
        time.sleep(2)
        job = _refresh_job(catalog, job_id)
        payload = {"job": job.to_dict(), "workers": _worker_statuses(catalog, job)}
    _emit(args, payload)


def _billing() -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    try:
        workspace = modal.Workspace.from_context()
        rates = {key: str(value) for key, value in workspace.billing.rates().items()}
        summary = workspace.billing.summary()
        return rates, {key: str(value) for key, value in asdict(summary).items()}, None
    except Exception as exc:
        return None, None, f"Billing unavailable or delayed: {exc}"


def _renderer_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "Dockerfile",
        "renderer/blendrender_render.py",
        "renderer/blendrender_enable_flip_fluids.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    addon_root = root / "third_party" / "flip-fluids"
    for path in sorted(
        path for path in addon_root.rglob("*") if path.is_file() and ".git" not in path.parts
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _job_billing(catalog: Catalog, job: JobManifest) -> dict[str, Any]:
    worker_seconds = sum(
        float(status.get("elapsed_seconds", 0)) for status in _worker_statuses(catalog, job)
    )
    estimate = {
        "status": "pending-rate",
        "gpu": job.gpu,
        "gpus_per_instance": job.gpus_per_instance,
        "gpu_seconds": worker_seconds * job.gpus_per_instance,
    }
    if not job.app_id:
        return {"reported_cost": "pending", "estimate": estimate}
    try:
        workspace = modal.Workspace.from_context()
        start = datetime.fromisoformat(job.created_at) - timedelta(hours=1)
        report = workspace.billing.report(start=start, resolution="h", tag_names=["*"])
        items = [item for item in report if item.object_id == job.app_id]
        if not items:
            return {"reported_cost": "pending", "estimate": estimate}
        resources = {name for item in items for name in item.cost_by_resource}
        return {
            "reported_cost": str(sum(item.cost for item in items)),
            "cost_by_resource": {
                name: str(sum(item.cost_by_resource.get(name, 0) for item in items))
                for name in resources
            },
            "estimate": estimate,
        }
    except Exception as exc:
        return {"reported_cost": "unavailable", "error": str(exc), "estimate": estimate}


def _split(frames: tuple[int, ...], count: int) -> list[tuple[int, ...]]:
    return [frames[index::count] for index in range(count)]


def _gpu_spec(gpu: str, count: int) -> str:
    return gpu if count == 1 else f"{gpu}:{count}"


def _validate_render_overrides(args: argparse.Namespace) -> None:
    if (args.resolution_x is None) != (args.resolution_y is None):
        raise CatalogError("--resolution-x and --resolution-y must be supplied together")
    if args.resolution_percentage is not None and args.resolution_percentage > 100:
        raise CatalogError("--resolution-percentage must be at most 100")


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _command_progress(command: str) -> Callable[[str], None]:
    def progress(message: str) -> None:
        _log(command, message)

    return progress


def _log(command: str, message: str) -> None:
    print(f"{command}: {message}", file=sys.stderr, flush=True)


def _should_log_step(index: int, total: int) -> bool:
    return total <= 10 or index == 1 or index == total or index % 25 == 0


def _emit(args: argparse.Namespace, value: Any) -> None:
    if args.as_json:
        print(json.dumps(value, default=str, indent=2, sort_keys=True))
        return
    if isinstance(value, list):
        for item in value:
            print(json.dumps(item, default=str, sort_keys=True))
    else:
        print(json.dumps(value, default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
