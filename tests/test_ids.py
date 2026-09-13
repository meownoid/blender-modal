from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from blender_modal import cli, output
from blender_modal.catalog import (
    Catalog,
    CatalogError,
    frame_path,
    result_manifest_path,
    scene_path,
)
from blender_modal.models import JobManifest, RenderSpec, SceneManifest

SCENE_ID = "a" * 64
RESULT_ID = "b" * 64


class MemoryCatalog(Catalog):
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.volume = Mock()
        self.writes: list[str] = []

    def exists(self, path: str) -> bool:
        return path in self.records

    def read_json(self, path: str) -> dict[str, Any]:
        return self.records[path].copy()

    def write_json(self, path: str, value: dict[str, Any], *, overwrite: bool = True) -> None:
        assert overwrite or path not in self.records
        self.writes.append(path)
        self.records[path] = value

    def _files(self, path: str) -> list[Any]:
        return [SimpleNamespace(path=key) for key in self.records if key.startswith(path + "/")]


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> MemoryCatalog:
    catalog = MemoryCatalog()
    monkeypatch.setattr(cli, "_catalog", lambda args: catalog)
    return catalog


@pytest.mark.parametrize("kind", ["scene", "result"])
def test_resolve_exact_unique_and_ambiguous_ids(catalog: MemoryCatalog, kind: str) -> None:
    root = "scenes" if kind == "scene" else "results"
    resolve = catalog.resolve_scene_id if kind == "scene" else catalog.resolve_result_id
    first = "a" * 63 + "1"
    second = "a" * 63 + "2"
    catalog.records[f"{root}/{first}/manifest.json"] = {}
    assert resolve(first) == first
    assert resolve("a") == first
    assert resolve(first[:12]) == first

    catalog.records[f"{root}/{second}/manifest.json"] = {}
    assert resolve(first) == first
    with pytest.raises(CatalogError, match=f"Ambiguous {kind}") as error:
        resolve(first[:12])
    assert first in str(error.value)
    assert second in str(error.value)
    assert "longer prefix or full ID" in str(error.value)

    # Exact IDs win even when they are a prefix of another stored ID.
    catalog.records[f"{root}/a/manifest.json"] = {}
    assert resolve("a") == "a"
    with pytest.raises(CatalogError, match=f"No {kind} matches"):
        resolve("missing")
    for invalid in ("", ".", "..", "../a", "a/b"):
        with pytest.raises(CatalogError, match=f"Invalid {kind}"):
            resolve(invalid)
    assert not catalog.writes


def test_resolution_uses_separate_namespaces_and_only_manifests(catalog: MemoryCatalog) -> None:
    catalog.records[scene_path("abc1")] = {}
    catalog.records[result_manifest_path("abc2")] = {}
    catalog.records["scenes/abc3/tree/manifest.json"] = {}
    catalog.records["results/abc4/frames/1/metadata.json"] = {}
    assert catalog.resolve_scene_id("abc") == "abc1"
    assert catalog.resolve_result_id("abc") == "abc2"


def test_prefix_and_full_scene_id_reuse_cached_result(
    catalog: MemoryCatalog, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scene = SceneManifest(SCENE_ID, "shot.blend", (), "2026-09-13T00:00:00+00:00")
    catalog.records[scene_path(SCENE_ID)] = scene.to_dict()
    spec = RenderSpec(SCENE_ID, "OPTIX", None, None, None, None, None, "fingerprint")
    result_id = catalog.create_result_set(spec)
    catalog.records[frame_path(result_id, 1, "metadata.json")] = {"status": "completed"}
    catalog.records[frame_path(result_id, 1, "frame.png")] = {}
    monkeypatch.setattr(cli, "_renderer_fingerprint", lambda: "fingerprint")
    for identifier in (SCENE_ID[:12], SCENE_ID):
        cli.main(["scene", "render", identifier, "--frames", "1", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["results"] == result_id
        assert payload["cached_frames"] == [1]
        assert payload["submitted_frames"] == []
    assert catalog.writes == [result_manifest_path(result_id)]
    assert catalog.records[result_manifest_path(result_id)]["spec"]["scene_id"] == SCENE_ID
    cli.main(["scene", "render", SCENE_ID[:12], "--frames", "1"])
    human = capsys.readouterr().out
    assert result_id[:12] in human
    assert result_id not in human


def test_result_filter_resolves_scene_prefix(
    catalog: MemoryCatalog, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog.records[scene_path(SCENE_ID)] = {}
    catalog.records[result_manifest_path(RESULT_ID)] = {"spec": {"scene_id": SCENE_ID}}
    cli.main(["result", "list", "--scene", SCENE_ID[:12], "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == RESULT_ID
    assert payload["spec"]["scene_id"] == SCENE_ID


@pytest.mark.parametrize("as_json", [False, True])
def test_submitted_workers_and_job_records_receive_full_ids(
    catalog: MemoryCatalog,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    scene = SceneManifest(SCENE_ID, "shot.blend", (), "2026-09-13T00:00:00+00:00")
    catalog.records[scene_path(SCENE_ID)] = scene.to_dict()
    monkeypatch.setattr(cli, "_renderer_fingerprint", lambda: "fingerprint")
    app = Mock(app_id="app-id")
    app.run.return_value = nullcontext()
    render_shard = Mock()
    spawn = render_shard.with_options.return_value.spawn
    spawn.return_value.object_id = "call-id"
    monkeypatch.setitem(
        sys.modules, "blender_modal.worker", SimpleNamespace(app=app, render_shard=render_shard)
    )
    monkeypatch.setattr(cli.modal.Volume, "from_name", Mock())
    cli.main(
        [
            "scene",
            "render",
            SCENE_ID[:12],
            "--frames",
            "1",
            "--detach",
            *(["--json"] if as_json else []),
        ]
    )
    job = catalog.list_jobs()[0]
    assert job.spec.scene_id == SCENE_ID
    assert job.result_id == job.spec.id
    assert len(job.result_id) == 64
    spawn.assert_called_once_with(
        "blender-modal-v2", job.id, job.result_id, 0, [1], job.spec.to_dict()
    )
    captured = capsys.readouterr().out
    if as_json:
        payload = json.loads(captured)
        assert payload["results"] == job.result_id
        assert payload["job"]["spec"]["scene_id"] == SCENE_ID
        assert payload["job"]["result_id"] == job.result_id
    else:
        assert job.result_id[:12] in captured
        assert job.result_id not in captured
        assert job.id in captured


def test_download_resolves_prefix_before_reading_frames(
    catalog: MemoryCatalog, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = b"png bytes"
    catalog.records[result_manifest_path(RESULT_ID)] = {}
    catalog.records[frame_path(RESULT_ID, 1, "metadata.json")] = {
        "status": "completed",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    catalog.records[frame_path(RESULT_ID, 1, "frame.png")] = {}
    catalog.volume.read_file.return_value = [content]
    cli.main(["result", "download", RESULT_ID[:12], "--output", str(tmp_path), "--json"])
    assert json.loads(capsys.readouterr().out)["results"] == RESULT_ID
    catalog.volume.read_file.assert_called_once_with(frame_path(RESULT_ID, 1, "frame.png"))
    assert (tmp_path / "frame_000001.png").read_bytes() == content


@pytest.mark.parametrize("kind", ["scene", "result"])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_remove_checks_full_ids_before_mutation(
    catalog: MemoryCatalog,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    dry_run: bool,
    active: bool,
) -> None:
    catalog.records[scene_path(SCENE_ID)] = {}
    catalog.records[result_manifest_path(RESULT_ID)] = {}
    full_id = SCENE_ID if kind == "scene" else RESULT_ID
    guard = Mock(side_effect=CatalogError("active render") if active else None)
    monkeypatch.setattr(
        cli, "_ensure_no_active_scene" if kind == "scene" else "_ensure_no_active_render", guard
    )
    removal = Mock(return_value=[])
    monkeypatch.setattr(catalog, "remove_scene" if kind == "scene" else "remove_results", removal)
    argv = [kind, "remove", full_id[:12], *(["--dry-run"] if dry_run else [])]
    if active:
        with pytest.raises(SystemExit):
            cli.main(argv)
        removal.assert_not_called()
    else:
        cli.main(argv)
        expected = (full_id,) if kind == "scene" else (full_id, None)
        removal.assert_called_once_with(*expected, dry_run=dry_run)
    guard.assert_called_once_with(catalog, full_id)


@pytest.mark.parametrize("kind", ["scene", "result"])
def test_ambiguous_removal_stops_before_guard_or_mutation(
    catalog: MemoryCatalog, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = "scenes" if kind == "scene" else "results"
    for suffix in ("1", "2"):
        catalog.records[f"{root}/{'a' * 63}{suffix}/manifest.json"] = {}
    guard = Mock()
    monkeypatch.setattr(
        cli, "_ensure_no_active_scene" if kind == "scene" else "_ensure_no_active_render", guard
    )
    with pytest.raises(SystemExit):
        cli.main([kind, "remove", "a" * 12])
    guard.assert_not_called()
    catalog.volume.remove_file.assert_not_called()
    assert not catalog.writes


def test_human_labels_are_short_and_json_keeps_full_ids(
    catalog: MemoryCatalog, capsys: pytest.CaptureFixture[str]
) -> None:
    scene = SceneManifest(SCENE_ID, "shot.blend", (), "2026-09-13T00:00:00+00:00")
    catalog.records[scene_path(SCENE_ID)] = scene.to_dict()
    catalog.records[result_manifest_path(RESULT_ID)] = {"spec": {"scene_id": SCENE_ID}}
    for resource, identifier in (("scene", SCENE_ID), ("result", RESULT_ID)):
        cli.main([resource, "list"])
        human = capsys.readouterr().out
        assert identifier[:12] in human
        assert identifier not in human
        assert SCENE_ID not in human
        cli.main([resource, "list", "--json"])
        assert json.loads(capsys.readouterr().out)["id"] == identifier
    for uploaded in (False, True):
        output.emit_human(cli._upload_human(scene, uploaded))
        human = capsys.readouterr().out
        assert SCENE_ID[:12] in human
        assert SCENE_ID not in human
    job = JobManifest(
        "12345678-1234-1234-1234-123456789abc",
        RESULT_ID,
        RenderSpec(SCENE_ID, "OPTIX", None, None, None, None, None, "fingerprint"),
        (1,),
        "L4",
        1,
        1,
        scene.created_at,
    )
    output.emit_human(cli._info_job_human({"job": job.to_dict()}))
    human = capsys.readouterr().out
    assert job.id in human
    assert RESULT_ID[:12] in human
    assert RESULT_ID not in human
    assert output.short_id("short") == "short"
