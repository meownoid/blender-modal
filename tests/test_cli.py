from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, create_autospec

import pytest
from modal.types import BillingReportItem

from blender_modal import cli, output
from blender_modal.catalog import Catalog, CatalogError
from blender_modal.models import JobManifest, RenderSpec, SceneFile, SceneManifest

COMMANDS = [
    (["scene", "upload", "shot.blend"], "_upload"),
    (["scene", "list"], "_scene_list"),
    (["scene", "remove", "scene-id"], "_scene_remove"),
    (["scene", "render", "scene-id", "--frames", "1:3"], "_render"),
    (["job", "list"], "_job_list"),
    (["job", "info", "job-id"], "_info"),
    (["job", "cancel", "job-id"], "_cancel"),
    (["result", "list"], "_result_list"),
    (["result", "download", "result-id", "--output", "renders"], "_download"),
    (["result", "remove", "result-id"], "_result_remove"),
    (["cleanup"], "_cleanup"),
    (["billing"], "_workspace_billing"),
]


@pytest.fixture(autouse=True)
def isolated_output() -> Iterator[None]:
    output.configure(verbose=False)
    yield
    output.configure(verbose=False)


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> Mock:
    catalog = create_autospec(Catalog, instance=True)
    monkeypatch.setattr(cli, "Catalog", Mock(return_value=catalog))
    return catalog


@pytest.fixture
def job() -> JobManifest:
    return JobManifest(
        id="job-id",
        result_id="result-id",
        spec=RenderSpec("scene-id", "OPTIX", None, None, None, None, None, "fingerprint"),
        requested_frames=(1, 2),
        gpu="L4",
        gpus_per_instance=1,
        instances=1,
        created_at="2026-09-13T00:00:00+00:00",
        status="completed",
    )


@pytest.mark.parametrize(("argv", "handler"), COMMANDS)
def test_command_dispatch(
    argv: list[str],
    handler: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = Mock()
    monkeypatch.setattr(cli, handler, called)
    cli.main(argv)
    called.assert_called_once()
    args = called.call_args.args[0]
    assert isinstance(args, argparse.Namespace)
    assert (args.volume, args.environment, args.as_json, args.verbose) == (
        "blender-modal-v2",
        None,
        False,
        False,
    )


@pytest.mark.parametrize(("argv", "handler"), COMMANDS)
@pytest.mark.parametrize("position", [0, 1, 2])
def test_global_flags_at_each_level(argv: list[str], handler: str, position: int) -> None:
    flags = ["--volume", "custom", "--environment", "dev", "--json", "-v"]
    args = cli._parser().parse_args(argv[:position] + flags + argv[position:])
    assert (args.volume, args.environment, args.as_json, args.verbose) == (
        "custom",
        "dev",
        True,
        True,
    )


def test_later_global_values_win() -> None:
    args = cli._parser().parse_args(
        [
            "--volume",
            "first",
            "--environment",
            "first",
            "--json",
            "scene",
            "--volume",
            "second",
            "--environment",
            "second",
            "--verbose",
            "list",
            "--volume",
            "third",
            "--environment",
            "third",
            "--volume",
            "last",
        ]
    )
    assert (args.volume, args.environment, args.as_json, args.verbose) == (
        "last",
        "third",
        True,
        True,
    )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["scene"],
        ["job"],
        ["result"],
        ["job", "info"],
        ["job", "cancel"],
        ["scene", "remove"],
        ["result", "remove"],
        ["result", "download", "id"],
        ["scene", "render", "id"],
        ["scene", "render", "id", "--frames", "1", "--instances", "0"],
        ["scene", "render", "id", "--frames", "1", "--backend", "CPU"],
        ["upload", "shot.blend"],
        ["list", "scenes"],
        ["list", "results"],
        ["render", "id", "--frames", "1"],
        ["download", "id", "--output", "renders"],
        ["remove", "scene", "id"],
        ["remove", "results", "id"],
        ["info"],
        ["info", "id"],
        ["cancel", "id"],
        ["results", "list"],
        ["billing", "job-id"],
    ],
)
def test_invalid_commands(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parser().parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "path",
    [[], ["scene"], ["job"], ["result"]]
    + [argv[:2] if len(argv) > 1 else argv for argv, _ in COMMANDS],
)
def test_contextual_help(path: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([*path, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert f"usage: blender-modal {' '.join(path)}".rstrip() in help_text
    for flag in ("--volume", "--environment", "--json", "--verbose"):
        assert flag in help_text


def test_operation_options() -> None:
    parser = cli._parser()
    upload = parser.parse_args(
        [
            "scene",
            "upload",
            "project",
            "--blend",
            "shot.blend",
            "--include",
            "assets",
            "--include",
            "cache",
            "--name",
            "Shot",
        ]
    )
    assert upload.root == Path("project")
    assert upload.blend == Path("shot.blend")
    assert upload.include == [Path("assets"), Path("cache")]
    assert upload.name == "Shot"
    render = parser.parse_args(["scene", "render", "id", "--frames", "1"])
    assert (render.gpu, render.gpus_per_instance, render.instances, render.backend) == (
        "L4",
        1,
        1,
        "OPTIX",
    )
    assert not render.detach
    assert render.frames == "1"
    for name in ("samples", "tile_size", "resolution_x", "resolution_y", "resolution_percentage"):
        assert getattr(render, name) is None
    render = parser.parse_args(
        [
            "scene",
            "render",
            "id",
            "--frames",
            "1,2",
            "--gpu",
            "A100",
            "--gpus-per-instance",
            "2",
            "--instances",
            "3",
            "--backend",
            "CUDA",
            "--samples",
            "32",
            "--tile-size",
            "64",
            "--resolution-x",
            "1920",
            "--resolution-y",
            "1080",
            "--resolution-percentage",
            "50",
            "--detach",
        ]
    )
    assert (render.gpu, render.gpus_per_instance, render.instances, render.backend) == (
        "A100",
        2,
        3,
        "CUDA",
    )
    assert (
        render.samples,
        render.tile_size,
        render.resolution_x,
        render.resolution_y,
        render.resolution_percentage,
        render.detach,
        render.frames,
    ) == (
        32,
        64,
        1920,
        1080,
        50,
        True,
        "1,2",
    )
    download = parser.parse_args(
        [
            "result",
            "download",
            "id",
            "--output",
            "renders",
            "--frames",
            "1:3",
            "--overwrite",
        ]
    )
    assert (download.results, download.output, download.frames, download.overwrite) == (
        "id",
        Path("renders"),
        "1:3",
        True,
    )
    cleanup = parser.parse_args(["cleanup", "--dry-run", "--force"])
    assert cleanup.dry_run and cleanup.force


def test_job_list_avoids_billing_and_refresh(
    catalog: Mock,
    job: JobManifest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog.list_jobs.return_value = [job]
    billing = Mock(side_effect=AssertionError("job list must not fetch billing"))
    refresh = Mock(side_effect=AssertionError("job list must not refresh jobs"))
    monkeypatch.setattr(cli, "_billing", billing)
    monkeypatch.setattr(cli, "_refresh_job", refresh)
    cli.main(["job", "list", "--json", "--volume", "custom", "--environment", "dev"])
    assert json.loads(capsys.readouterr().out) == {"jobs": [job.to_dict()]}
    cli.Catalog.assert_called_once_with("custom", "dev")
    billing.assert_not_called()
    refresh.assert_not_called()


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
def test_billing_avoids_catalog(
    unavailable: bool,
    as_json: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    constructor = Mock(side_effect=AssertionError("billing must not open a catalog"))
    monkeypatch.setattr(cli, "Catalog", constructor)
    rates = None if unavailable else {"gpu": "0.1"}
    summary = None if unavailable else {"total": "1.0"}
    error = "Billing unavailable or delayed: offline" if unavailable else None
    monkeypatch.setattr(cli, "_billing", Mock(return_value=(rates, summary, error)))
    cli.main(["billing", *(["--json"] if as_json else [])])
    captured = capsys.readouterr().out
    if as_json:
        assert json.loads(captured) == {
            "billing": {
                "rates": rates,
                "summary": summary,
                "reported_costs": "pending",
                "error": error,
            }
        }
    else:
        assert (error or "total: 1.0") in captured
        assert "No jobs found" not in captured
    constructor.assert_not_called()


@pytest.mark.parametrize("watch", [False, True])
def test_job_info_preserves_payload_and_watch(
    watch: bool,
    catalog: Mock,
    job: JobManifest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refresh = Mock(return_value=job)
    monkeypatch.setattr(cli, "_refresh_job", refresh)
    monkeypatch.setattr(cli, "_worker_statuses", Mock(return_value=[]))
    monkeypatch.setattr(cli, "_job_billing", Mock(return_value={"reported_cost": "pending"}))
    watcher = Mock()
    monkeypatch.setattr(cli, "_watch", watcher)
    cli.main(["job", "info", job.id, "--json", *(["--watch"] if watch else [])])
    payload = {"job": job.to_dict(), "workers": [], "billing": {"reported_cost": "pending"}}
    refresh.assert_called_once_with(catalog, job.id)
    if watch:
        watcher.assert_called_once()
        assert watcher.call_args.args[:2] == (catalog, job.id)
        assert watcher.call_args.args[3] == payload
    else:
        assert json.loads(capsys.readouterr().out) == payload
        watcher.assert_not_called()


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_job_info_reconciles_cancelled_workers(
    as_json: bool,
    cancelled: bool,
    catalog: Mock,
    job: JobManifest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job = replace(
        job, instances=4, requested_frames=(1, 2, 3, 4),
        status="cancelled" if cancelled else "running", cancelled=cancelled,
    )
    stored = [
        {"status": state, "frames": [index], "completed_frames": [index] if index else []}
        for index, state in enumerate(["running", "queued", "completed", "failed"])
    ]
    monkeypatch.setattr(cli, "_refresh_job", Mock(return_value=job))
    monkeypatch.setattr(cli, "_job_billing", Mock(return_value={"reported_cost": "pending"}))
    catalog.read_json.side_effect = stored
    cli.main(["job", "info", job.id, *(["--json"] if as_json else [])])
    captured = capsys.readouterr().out
    expected = ["cancelled", "cancelled", "completed", "failed"] if cancelled else [
        "running", "queued", "completed", "failed"
    ]
    if as_json:
        workers = json.loads(captured)["workers"]
        assert [worker["status"] for worker in workers] == expected
        assert [worker["completed_frames"] for worker in workers] == [[], [1], [2], [3]]
    else:
        for index, state in enumerate(expected):
            assert f"#{index} {state}" in captured
    assert [worker["status"] for worker in stored] == ["running", "queued", "completed", "failed"]
    catalog.write_json.assert_not_called()


@pytest.mark.parametrize(
    ("finished", "end", "chunk_count"),
    [
        ("2026-09-05T12:13:02+00:00", "2026-09-05T13:00:00+00:00", 1),
        ("2026-09-20T12:13:02+00:00", "2026-09-20T13:00:00+00:00", 3),
        (None, "2026-09-30T14:00:00+00:00", 4),
        ("2026-09-30T14:05:00+00:00", "2026-09-30T14:00:00+00:00", 4),
    ],
)
def test_job_billing_bounds_and_splits_hourly_reports(
    finished: str | None,
    end: str,
    chunk_count: int,
    catalog: Mock,
    job: JobManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = replace(
        job, app_id="app-id", created_at="2026-09-05T11:35:21+00:00",
        completed_at=finished, status="completed" if finished else "running",
    )
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime(2026, 9, 30, 14, 30, tzinfo=UTC)
    monkeypatch.setattr(cli, "datetime", clock)
    catalog.read_json.return_value = {"elapsed_seconds": 30}
    workspace = Mock()
    monkeypatch.setattr(cli.modal.Workspace, "from_context", Mock(return_value=workspace))
    item = BillingReportItem(
        object_id=job.app_id, description="render", environment_name="main",
        interval_start=datetime(2026, 9, 5, 11, tzinfo=UTC),
        cost=Decimal("1.25"), cost_by_resource={"L4": Decimal("1.00"), "CPU": Decimal("0.25")},
        tags={},
    )
    workspace.billing.report.return_value = [item, replace(item, object_id="unrelated-app")]

    billing = cli._job_billing(catalog, job)

    calls = workspace.billing.report.call_args_list
    assert len(calls) == chunk_count
    assert calls[0].kwargs["start"] == datetime(2026, 9, 5, 10, tzinfo=UTC)
    assert calls[-1].kwargs["end"] == datetime.fromisoformat(end)
    for call in calls:
        start, stop = call.kwargs["start"], call.kwargs["end"]
        assert timedelta(0) < stop - start <= timedelta(days=7)
        assert start.minute == stop.minute == start.second == stop.second == 0
        assert call.kwargs["resolution"] == "h"
    for previous, following in zip(calls, calls[1:], strict=False):
        assert previous.kwargs["end"] == following.kwargs["start"]
    assert billing["reported_cost"] == str(Decimal("1.25") * chunk_count)
    assert billing["cost_by_resource"] == {
        "L4": str(Decimal("1.00") * chunk_count), "CPU": str(Decimal("0.25") * chunk_count),
    }
    assert billing["estimate"]["gpu_seconds"] == 30


@pytest.mark.parametrize("outcome", ["no-app", "empty", "error"])
def test_job_billing_unavailable_or_pending(
    outcome: str,
    catalog: Mock,
    job: JobManifest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job = replace(
        job, app_id=None if outcome == "no-app" else "app-id",
        created_at="2026-09-05T11:35:21+00:00", completed_at="2026-09-05T12:13:02+00:00",
    )
    catalog.read_json.return_value = {}
    workspace = Mock()
    monkeypatch.setattr(cli.modal.Workspace, "from_context", Mock(return_value=workspace))
    workspace.billing.report.return_value = []
    if outcome == "error":
        workspace.billing.report.side_effect = RuntimeError("Billing access denied [details]")
    billing = cli._job_billing(catalog, job)
    if outcome == "error":
        assert billing["reported_cost"] == "unavailable"
        output.emit_human(cli._info_job_human({"job": job.to_dict(), "billing": billing}))
        captured = capsys.readouterr().out
        assert "Cost: unavailable" in captured
        assert "Billing access denied [details]" in captured
    else:
        assert billing["reported_cost"] == "pending"
        assert "error" not in billing
        if outcome == "no-app":
            workspace.billing.report.assert_not_called()


def test_scene_and_result_lists(catalog: Mock, capsys: pytest.CaptureFixture[str]) -> None:
    scene = SceneManifest(
        "scene-id",
        "shot.blend",
        (SceneFile("shot.blend", "hash", 42),),
        "2026-09-13T00:00:00+00:00",
        "Shot",
    )
    catalog.list_scenes.return_value = [scene]
    cli.main(["scene", "list", "--json"])
    assert json.loads(capsys.readouterr().out) == {
        "id": scene.id,
        "name": "Shot",
        "entrypoint": "shot.blend",
        "files": 1,
        "size_bytes": 42,
        "created_at": scene.created_at,
    }
    results = [{"id": "result-id", "frames": [1], "jobs": ["job-id"]}]
    catalog.list_results.return_value = results
    cli.main(["result", "list", "--scene", scene.id, "--json"])
    catalog.list_results.assert_called_once_with(scene.id)
    assert json.loads(capsys.readouterr().out) == results[0]


@pytest.mark.parametrize("resource", ["scene", "result"])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_removal_preserves_guards_and_arguments(
    resource: str,
    dry_run: bool,
    active: bool,
    catalog: Mock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    guard = Mock(side_effect=CatalogError("active render") if active else None)
    monkeypatch.setattr(
        cli, "_ensure_no_active_scene" if resource == "scene" else "_ensure_no_active_render", guard
    )
    removal = catalog.remove_scene if resource == "scene" else catalog.remove_results
    removal.return_value = ["some/path"]
    argv = [resource, "remove", "id", "--json", *(["--dry-run"] if dry_run else [])]
    if resource == "result":
        argv += ["--frames", "1:3"]
    if active:
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
        removal.assert_not_called()
        assert "active render" in capsys.readouterr().err
    else:
        cli.main(argv)
        if resource == "scene":
            removal.assert_called_once_with("id", dry_run=dry_run)
        else:
            removal.assert_called_once_with("id", (1, 2, 3), dry_run=dry_run)
        assert json.loads(capsys.readouterr().out) == {"dry_run": dry_run, "removed": ["some/path"]}
    guard.assert_called_once_with(catalog, "id")
