"""Command-line interface for immutable Modal Blender render workspaces."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import modal
from modal.types import BillingReportItem
from rich.console import Group
from rich.text import Text

from . import output
from .catalog import Catalog, CatalogError, frame_path, result_manifest_path
from .frames import parse_frames
from .models import JobManifest, RenderSpec, SceneManifest, utc_now
from .resources import image_context


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    output.configure(verbose=args.verbose)
    try:
        args.handler(args)
    except (CatalogError, ValueError, modal.exception.Error) as exc:
        output.stop_status()
        parser.error(str(exc))
    finally:
        output.stop_status()


def _add_global_arguments(parser: argparse.ArgumentParser) -> None:
    # Nested parsers must preserve options already supplied at an earlier level.
    parser.add_argument(
        "--volume", default=argparse.SUPPRESS, help="Modal Volume name (default: blender-modal-v2)"
    )
    parser.add_argument("--environment", default=argparse.SUPPRESS, help="Modal environment name")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        default=argparse.SUPPRESS,
        help="machine-readable output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="log progress details to standard error",
    )


def _command_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help: str,
    handler: Callable[[argparse.Namespace], None] | None = None,
) -> argparse.ArgumentParser:
    parser = commands.add_parser(name, help=help, description=help)
    _add_global_arguments(parser)
    if handler is not None:
        parser.set_defaults(handler=handler)
    return parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blender-modal")
    _add_global_arguments(parser)
    parser.set_defaults(volume="blender-modal-v2", environment=None, as_json=False, verbose=False)
    commands = parser.add_subparsers(dest="command", required=True)
    scenes = _command_parser(commands, "scene", "manage uploaded scenes")
    scene_commands = scenes.add_subparsers(dest="action", required=True)
    jobs = _command_parser(commands, "job", "manage render jobs")
    job_commands = jobs.add_subparsers(dest="action", required=True)
    results = _command_parser(commands, "result", "manage render result sets")
    result_commands = results.add_subparsers(dest="action", required=True)

    upload = _command_parser(scene_commands, "upload", "upload an immutable project tree", _upload)
    upload.add_argument(
        "root",
        nargs="?",
        type=Path,
        help="project root, or a .blend file when --blend is omitted",
    )
    upload.add_argument("--blend", type=Path, help="entrypoint .blend file")
    upload.add_argument(
        "--include",
        action="append",
        type=Path,
        default=[],
        help="extra file or directory to include; relative to the project root "
        "(repeatable)",
    )
    upload.add_argument("--name", help="human-readable scene name")

    _command_parser(scene_commands, "list", "list uploaded scenes", _scene_list)
    listed_results = _command_parser(
        result_commands, "list", "list render result sets", _result_list
    )
    listed_results.add_argument(
        "--scene", help="only show results for this scene ID or unique prefix"
    )

    render = _command_parser(
        scene_commands, "render", "render missing frames on Modal GPUs", _render
    )
    render.add_argument("scene", help="scene ID or unique prefix to render")
    render.add_argument(
        "--frames", required=True, help="frame selection, e.g. 1:120 or 1:7:3,2"
    )
    render.add_argument("--gpu", default="L4", help="Modal GPU type (default: %(default)s)")
    render.add_argument(
        "--gpus-per-instance",
        type=_positive,
        default=1,
        help="GPUs per worker container (default: %(default)s)",
    )
    render.add_argument(
        "--instances",
        type=_positive,
        default=1,
        help="maximum number of parallel workers (default: %(default)s)",
    )
    render.add_argument(
        "--backend",
        choices=("OPTIX", "CUDA"),
        default="OPTIX",
        help="Cycles device backend (default: %(default)s)",
    )
    render.add_argument("--samples", type=_positive, help="override scene render samples")
    render.add_argument("--tile-size", type=_positive, help="override render tile size")
    render.add_argument(
        "--resolution-x", type=_positive, help="override output width (requires --resolution-y)"
    )
    render.add_argument(
        "--resolution-y", type=_positive, help="override output height (requires --resolution-x)"
    )
    render.add_argument(
        "--resolution-percentage", type=_positive, help="scale the render resolution (1-100)"
    )
    render.add_argument(
        "--detach", action="store_true", help="submit and return immediately without waiting"
    )

    download = _command_parser(result_commands, "download", "download completed PNGs", _download)
    download.add_argument("results", help="results ID or unique prefix to download")
    download.add_argument(
        "--output", required=True, type=Path, help="destination directory for PNG frames"
    )
    download.add_argument(
        "--frames", help="frame selection (default: all completed frames)"
    )
    download.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing local files instead of failing",
    )

    scene = _command_parser(scene_commands, "remove", "delete an uploaded scene", _scene_remove)
    scene.add_argument("scene", help="scene ID or unique prefix to delete")
    scene.add_argument(
        "--dry-run", action="store_true", help="show what would be removed without deleting"
    )
    result = _command_parser(result_commands, "remove", "delete rendered frames", _result_remove)
    result.add_argument("results", help="results ID or unique prefix to delete")
    result.add_argument(
        "--frames", help="only remove these frames (default: the whole result set)"
    )
    result.add_argument(
        "--dry-run", action="store_true", help="show what would be removed without deleting"
    )

    cleanup = _command_parser(
        commands, "cleanup", "remove abandoned staging and unreferenced blobs", _cleanup
    )
    cleanup.add_argument(
        "--dry-run", action="store_true", help="show what would be removed without deleting"
    )
    cleanup.add_argument(
        "--force",
        action="store_true",
        help="also remove fresh unreferenced blobs from interrupted uploads",
    )

    _command_parser(job_commands, "list", "list render jobs", _job_list)
    logs = _command_parser(job_commands, "logs", "show render worker logs", _job_logs)
    logs.description = "Show render worker logs as text (--json is unsupported)."
    logs.add_argument("job", help="job ID whose logs to display")
    log_mode = logs.add_mutually_exclusive_group()
    log_mode.add_argument(
        "--tail", type=_log_tail, help="number of recent entries (1-20000; default: 100)"
    )
    log_mode.add_argument(
        "-f", "--follow", action="store_true", help="stream logs until the Modal app stops"
    )
    info = _command_parser(job_commands, "info", "show job state and Modal billing", _info)
    info.add_argument("job", help="job ID to inspect")
    info.add_argument(
        "--watch", action="store_true", help="reprint job state until it finishes"
    )

    cancel = _command_parser(job_commands, "cancel", "cancel a submitted render", _cancel)
    cancel.add_argument("job", help="job ID to cancel")
    _command_parser(commands, "billing", "show workspace billing", _workspace_billing)
    return parser


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
    _emit(
        args,
        {"scene": manifest.to_dict(), "uploaded": uploaded},
        _upload_human(manifest, uploaded),
    )


def _upload_root_and_blend(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.blend is None:
        if args.root is None:
            raise ValueError("scene upload requires a .blend file or ROOT with --blend")
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


def _scene_list(args: argparse.Namespace) -> None:
    _log("scene list", "Loading scenes")
    values = [
        {
            "id": scene.id,
            "name": scene.name,
            "entrypoint": scene.entrypoint,
            "files": len(scene.files),
            "size_bytes": sum(file.size for file in scene.files),
            "created_at": scene.created_at,
        }
        for scene in _catalog(args).list_scenes()
    ]
    _log("scene list", f"Found {len(values)} scenes")
    _emit(args, values, _scenes_human(values))


def _result_list(args: argparse.Namespace) -> None:
    _log("result list", "Loading results")
    catalog = _catalog(args)
    scene_id = catalog.resolve_scene_id(args.scene) if args.scene is not None else None
    values = catalog.list_results(scene_id)
    _log("result list", f"Found {len(values)} results")
    _emit(args, values, _results_human(values))


def _render(args: argparse.Namespace) -> None:
    _log("render", f"Loading scene {args.scene}")
    catalog = _catalog(args)
    args.scene = catalog.resolve_scene_id(args.scene)
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
            Group(
                f"[green]✓[/] All {len(requested)} requested frame(s) already rendered",
                Text(f"Results: {output.short_id(result_id)}", style="cyan"),
            ),
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
            Group(
                Text(f"Job:     {job.id}", style="cyan"),
                Text(f"Results: {output.short_id(result_id)}", style="cyan"),
                f"Frames:  {len(existing)} cached, {len(missing)} submitted",
                Text(f"Track:   blender-modal job info {job.id}", style="dim"),
            ),
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
                raise CatalogError("One or more render shards failed; use job info JOB for details")
            _log("render", "Render completed")
        else:
            _log("render", "Render detached; use job info to monitor the job")


def _download(args: argparse.Namespace) -> None:
    _log("download", f"Loading results {args.results}")
    catalog = _catalog(args)
    args.results = catalog.resolve_result_id(args.results)
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
        with temporary.open("wb") as sink:
            for chunk in catalog.volume.read_file(frame_path(args.results, frame, "frame.png")):
                sink.write(chunk)
        if _file_hash(temporary) != metadata.get("sha256"):
            temporary.unlink(missing_ok=True)
            raise CatalogError(f"Downloaded checksum does not match for frame {frame}")
        temporary.replace(target)
        downloaded.append(frame)
    _log("download", f"Finished: {len(downloaded)} downloaded, {len(skipped)} already present")
    human = f"[green]✓[/] {len(downloaded)} frame(s) written to {args.output}"
    if skipped:
        human += f" [dim]({len(skipped)} already present)[/]"
    _emit(
        args,
        {"results": args.results, "downloaded_frames": downloaded, "skipped_frames": skipped},
        human,
    )


def _scene_remove(args: argparse.Namespace) -> None:
    catalog = _catalog(args)
    args.scene = catalog.resolve_scene_id(args.scene)
    _log("scene remove", f"Checking scene {args.scene} is not rendering")
    _ensure_no_active_scene(catalog, args.scene)
    _log("scene remove", f"{'Previewing' if args.dry_run else 'Removing'} scene {args.scene}")
    removed = catalog.remove_scene(args.scene, dry_run=args.dry_run)
    _emit_removal(args, removed)


def _result_remove(args: argparse.Namespace) -> None:
    catalog = _catalog(args)
    args.results = catalog.resolve_result_id(args.results)
    _log("result remove", f"Checking results {args.results} are not rendering")
    _ensure_no_active_render(catalog, args.results)
    frames = parse_frames(args.frames) if args.frames else None
    _log("result remove", f"{'Previewing' if args.dry_run else 'Removing'} results {args.results}")
    removed = catalog.remove_results(args.results, frames, dry_run=args.dry_run)
    _emit_removal(args, removed)


def _emit_removal(args: argparse.Namespace, removed: list[str]) -> None:
    _log("remove", f"Selected {len(removed)} path(s)")
    _emit(
        args,
        {"dry_run": args.dry_run, "removed": removed},
        _removal_human(removed, args.dry_run),
    )


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
        _removal_human(removed, args.dry_run),
    )


def _info(args: argparse.Namespace) -> None:
    catalog = _catalog(args)
    _log("job info", f"Loading job {args.job}")
    job = _refresh_job(catalog, args.job)
    _log("job info", "Loading worker status and billing")
    payload: dict[str, Any] = {
        "job": job.to_dict(),
        "workers": _worker_statuses(catalog, job),
        "billing": _job_billing(catalog, job),
    }
    if args.watch:
        _log("job info", "Watching for job updates")
        _watch(catalog, job.id, args, payload, _info_job_human(payload))
        return
    _emit(args, payload, _info_job_human(payload))


def _job_logs(args: argparse.Namespace) -> None:
    if args.as_json:
        raise CatalogError("job logs provides text output and does not support --json")
    _log("job logs", f"Loading job {args.job}")
    job = _catalog(args).job(args.job)
    if not job.app_id:
        raise CatalogError(f"Job {job.id} has no Modal app ID; logs are unavailable")
    command = [
        sys.executable, "-m", "modal", "app", "logs", job.app_id,
        "--timestamps", "--show-container-id",
    ]
    if args.environment is not None:
        command.extend(["--env", args.environment])
    if args.follow:
        command.append("--follow")
    else:
        command.extend(["--tail", str(args.tail if args.tail is not None else 100)])
    _log("job logs", f"{'Following' if args.follow else 'Fetching'} logs for {job.app_id}")
    output.stop_status()
    try:
        result = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except OSError as exc:
        raise CatalogError(f"Could not start Modal log viewer: {exc}") from exc
    if result.returncode:
        # Subprocesses use negative return codes for signals; shells use 128 + signal.
        raise SystemExit(
            result.returncode if result.returncode > 0 else 128 - result.returncode
        )


def _job_list(args: argparse.Namespace) -> None:
    _log("job list", "Loading jobs")
    jobs = _catalog(args).list_jobs()
    _emit(args, {"jobs": [job.to_dict() for job in jobs]}, _jobs_human(jobs))


def _workspace_billing(args: argparse.Namespace) -> None:
    _log("billing", "Loading workspace billing")
    rates, summary, report_error = _billing()
    _emit(
        args,
        {
            "billing": {
                "rates": rates,
                "summary": summary,
                "reported_costs": "pending",
                "error": report_error,
            },
        },
        _billing_human(summary, report_error),
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
    lines = [f"[green]✓[/] Cancelled job {job.id}"]
    if errors:
        lines.append(f"[yellow]{len(errors)} shard(s) reported cancellation errors[/]")
    _emit(args, {"job": job.id, "cancelled": True, "errors": errors}, Group(*lines))


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
            status = catalog.read_json(f"jobs/{job.id}/workers/{index}.json")
        except CatalogError:
            continue
        # Terminated containers may never publish a final status. Reconcile at read
        # time so this also covers jobs cancelled before this behavior was added.
        if job.status == "cancelled" and status.get("status") in {"queued", "running"}:
            status = {**status, "status": "cancelled"}
        statuses.append(status)
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
    catalog: Catalog,
    job_id: str,
    args: argparse.Namespace,
    payload: dict[str, Any],
    human: Any,
) -> None:
    import time

    while payload["job"]["status"] == "running":
        _emit(args, payload, human)
        time.sleep(2)
        job = _refresh_job(catalog, job_id)
        payload = {"job": job.to_dict(), "workers": _worker_statuses(catalog, job)}
        human = _info_job_human(payload)
    _emit(args, payload, human)


def _billing() -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    try:
        workspace = modal.Workspace.from_context()
        rates = {key: str(value) for key, value in workspace.billing.rates().items()}
        summary = workspace.billing.summary()
        return rates, {key: str(value) for key, value in asdict(summary).items()}, None
    except Exception as exc:
        return None, None, f"Billing unavailable or delayed: {exc}"


def _renderer_fingerprint() -> str:
    root = image_context()
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
        hour = timedelta(hours=1)
        start = datetime.fromisoformat(job.created_at).astimezone(UTC) - hour
        start = start.replace(minute=0, second=0, microsecond=0)
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        if job.completed_at:
            finished = datetime.fromisoformat(job.completed_at).astimezone(UTC)
            # Modal excludes partial final hours; include the hour of completion.
            end = min(end, finished.replace(minute=0, second=0, microsecond=0) + hour)
        items: list[BillingReportItem] = []
        while start < end:
            chunk_end = min(start + timedelta(days=7), end)
            report = workspace.billing.report(
                start=start, end=chunk_end, resolution="h", tag_names=["*"]
            )
            items.extend(item for item in report if item.object_id == job.app_id)
            start = chunk_end
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


def _log_tail(value: str) -> int:
    parsed = _positive(value)
    if parsed > 20_000:
        raise argparse.ArgumentTypeError("must not exceed 20000")
    return parsed


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
    output.log(command, message)


def _should_log_step(index: int, total: int) -> bool:
    return total <= 10 or index == 1 or index == total or index % 25 == 0


def _emit(args: argparse.Namespace, value: Any, human: Any = None) -> None:
    if args.as_json:
        output.emit_json(value)
        return
    output.emit_human(human if human is not None else str(value))


def _upload_human(manifest: SceneManifest, uploaded: bool) -> Any:
    if not uploaded:
        return Text(
            f"Scene {output.short_id(manifest.id)} already exists; upload skipped",
            style="yellow",
        )
    size = output.size_bytes(sum(file.size for file in manifest.files))
    return Group(
        Text(f"✓ Uploaded scene {output.short_id(manifest.id)}", style="green"),
        Text(
            f"{manifest.name or 'unnamed'} · {len(manifest.files)} file(s) · {size}",
            style="dim",
        ),
    )


def _scenes_human(values: list[dict[str, Any]]) -> Any:
    if not values:
        return Text("No scenes found", style="dim")
    entries: list[Any] = []
    for item in values:
        details = (
            f"{item.get('name') or 'unnamed'} · {item['entrypoint']} · "
            f"{item['files']} file(s) · {output.size_bytes(int(item['size_bytes']))} · "
            f"{_format_timestamp(str(item['created_at']))}"
        )
        entries.append(_entry(output.short_id(str(item["id"])), details))
    return Group(*_spaced(entries))


def _results_human(values: list[dict[str, Any]]) -> Any:
    if not values:
        return Text("No results found", style="dim")
    entries: list[Any] = []
    for item in values:
        spec = item.get("spec") or {}
        details = (
            f"{len(item.get('frames') or [])} frame(s) · "
            f"{len(item.get('jobs') or [])} job(s) · "
            f"{_format_timestamp(str(item.get('created_at', '')))}"
        )
        entries.append(
            _entry(
                output.short_id(str(item["id"])),
                details,
                f"scene {output.short_id(str(spec.get('scene_id', '-')))}",
            )
        )
    return Group(*_spaced(entries))


def _entry(identifier: str, *details: str) -> Any:
    return Group(
        Text(identifier, style="cyan"),
        *(Text(f"  {line}", style="dim") for line in details),
    )


def _spaced(entries: list[Any]) -> list[Any]:
    spaced: list[Any] = []
    for index, item in enumerate(entries):
        if index:
            spaced.append(Text(""))
        spaced.append(item)
    return spaced


def _removal_human(removed: list[str], dry_run: bool) -> Any:
    count = len(removed)
    if dry_run:
        header = Text(f"Would remove {count} path(s) (dry run)", style="yellow")
    else:
        header = Text(f"Removed {count} path(s)", style="green" if count else "dim")
    shown = removed[:50]
    lines: list[Any] = [header, *(Text(path, style="dim") for path in shown)]
    if len(removed) > len(shown):
        lines.append(Text(f"… and {len(removed) - len(shown)} more", style="dim"))
    return Group(*lines)


def _info_job_human(payload: dict[str, Any]) -> Any:
    job = payload["job"]
    status = str(job["status"])
    lines: list[Any] = [
        Text.assemble(
            ("Job: ", ""),
            (str(job["id"]), "cyan"),
            ("  ", ""),
            (f"[{status}]", output.status_style(status)),
        ),
        Text.assemble(("Results: ", ""), (output.short_id(str(job["result_id"])), "cyan")),
        f"Frames: {len(job['requested_frames'])} requested · "
        f"GPU: {job['gpu']} × {job['gpus_per_instance']} · Instances: {job['instances']}",
        Text.assemble(
            ("Created: ", ""),
            (_format_timestamp(str(job["created_at"])), "cyan"),
        ),
    ]
    workers = payload.get("workers") or []
    if workers:
        lines.append(Text("Workers:", style="bold"))
        for index, worker in enumerate(workers):
            worker_status = str(worker.get("status", "unknown"))
            completed = len(worker.get("completed_frames") or [])
            total = len(worker.get("frames") or [])
            elapsed = worker.get("elapsed_seconds")
            detail = f"{completed}/{total} frames"
            if elapsed is not None:
                detail += f" · {float(elapsed):.0f}s"
            if worker.get("error"):
                detail += f" · {worker['error']}"
            lines.append(
                Text.assemble(
                    (f"  #{index} ", ""),
                    (worker_status, output.status_style(worker_status)),
                    (f" {detail}", ""),
                )
            )
    billing = payload.get("billing") or {}
    estimate = billing.get("estimate") or {}
    if billing.get("reported_cost") not in (None, "pending"):
        lines.append(f"Cost: {output.cost(str(billing['reported_cost']))}")
        if billing.get("error"):
            lines.append(Text(str(billing["error"]), style="yellow"))
    elif estimate.get("gpu_seconds"):
        gpu_seconds = float(estimate["gpu_seconds"])
        lines.append(
            Text(f"Estimated GPU time: {gpu_seconds:.0f}s (cost pending)", style="dim")
        )
    return Group(*lines)


def _jobs_human(jobs: list[JobManifest]) -> Any:
    lines: list[Any] = []
    if jobs:
        entries = [
            Group(
                Text.assemble(
                    (job.id, "cyan"),
                    ("  ", ""),
                    (job.status, output.status_style(job.status)),
                ),
                Text(
                    f"  {len(job.requested_frames)} frame(s) · {job.gpu} × "
                    f"{job.gpus_per_instance} · {_format_timestamp(job.created_at)}",
                    style="dim",
                ),
            )
            for job in jobs
        ]
        lines.append(Group(*_spaced(entries)))
    else:
        lines.append(Text("No jobs found", style="dim"))
    return Group(*lines)


def _billing_human(summary: dict[str, Any] | None, report_error: str | None) -> Any:
    lines: list[Any] = []
    if summary:
        start = str(summary.get("start", ""))[:10]
        end = str(summary.get("end", ""))[:10]
        label = f"{start} → {end}" if start and end else "current period"
        lines.append(Text(f"Billing ({label}):", style="bold"))
        for key, value in sorted(summary.items()):
            if key in ("adjustments", "metered_cost_breakdown"):
                continue
            displayed = str(value) if key in ("start", "end") else output.cost(str(value))
            lines.append(Text(f"  {key}: {displayed}", style="dim"))
    if report_error:
        lines.append(Text(report_error, style="dim"))
    return Group(*lines)


def _format_timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


if __name__ == "__main__":
    main()
