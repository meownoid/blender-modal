from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest
from rich.text import Text

from blender_modal import output
from blender_modal.catalog import Catalog, CatalogError, build_scene
from blender_modal.cli import _emit, _log, _parser, _upload_includes, _upload_root_and_blend
from blender_modal.frames import parse_frames


@pytest.fixture(autouse=True)
def reset_output() -> Iterator[None]:
    output.configure(verbose=False)
    yield
    output.configure(verbose=False)


def test_scene_identity_is_stable_and_content_based(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    blend = root / "scene.blend"
    texture = root / "textures" / "albedo.png"
    texture.parent.mkdir()
    blend.write_bytes(b"blend")
    texture.write_bytes(b"texture")

    first, _ = build_scene(root, blend, [], "One")
    second, _ = build_scene(root, blend, [], "Another name")

    assert first.id == second.id
    texture.write_bytes(b"changed")
    changed, _ = build_scene(root, blend, [], None)
    assert changed.id != first.id


def test_selected_upload_keeps_entrypoint_and_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    blend = root / "scene.blend"
    resource = root / "assets" / "image.png"
    ignored = root / "ignored.bin"
    resource.parent.mkdir()
    blend.write_bytes(b"blend")
    resource.write_bytes(b"image")
    ignored.write_bytes(b"ignored")

    manifest, _ = build_scene(root, blend, [resource], None)

    assert manifest.entrypoint == "scene.blend"
    assert [file.path for file in manifest.files] == ["assets/image.png", "scene.blend"]


def test_rejects_paths_outside_project(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    blend = root / "scene.blend"
    blend.write_bytes(b"blend")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")

    with pytest.raises(CatalogError, match="outside"):
        build_scene(root, blend, [outside], None)


def test_rejects_symlinked_resource(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    blend = root / "scene.blend"
    blend.write_bytes(b"blend")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = root / "linked.bin"
    link.symlink_to(outside)

    with pytest.raises(CatalogError, match="Symlinks"):
        build_scene(root, blend, [], None)


def test_upload_accepts_a_blend_file_without_explicit_root(tmp_path: Path) -> None:
    blend = tmp_path / "project" / "scene.blend"
    args = _parser().parse_args(["upload", str(blend)])

    root, entrypoint = _upload_root_and_blend(args)

    assert root == blend.parent
    assert entrypoint == blend
    assert _upload_includes(args, root, entrypoint) == [blend]


def test_upload_keeps_explicit_root_and_relative_blend(tmp_path: Path) -> None:
    root = tmp_path / "project"
    args = _parser().parse_args(["upload", str(root), "--blend", "scenes/shot.blend"])

    upload_root, entrypoint = _upload_root_and_blend(args)

    assert upload_root == root
    assert entrypoint == root / "scenes" / "shot.blend"
    assert _upload_includes(args, upload_root, entrypoint) == []


def test_build_scene_reports_hashing_progress(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    blend = root / "scene.blend"
    blend.write_bytes(b"blend")
    progress: list[str] = []

    build_scene(root, blend, [], None, progress=progress.append)

    assert progress == ["Scanning project files", "Hashing 1/1: scene.blend"]


def test_catalog_missing_remote_path_does_not_exist() -> None:
    class MissingPathVolume:
        def listdir(self, path: str) -> list[object]:
            raise modal.exception.NotFoundError(f'path "/{path}" does not exist')

    catalog = Catalog.__new__(Catalog)
    catalog.volume = MissingPathVolume()

    assert not catalog.exists("blobs/sha256/missing")


def test_forced_cleanup_removes_fresh_unreferenced_uploads() -> None:
    class CleanupCatalog(Catalog):
        def __init__(self) -> None:
            pass

        def list_scenes(self) -> list[object]:
            return []

        def _files(self, path: str) -> list[object]:
            return [
                SimpleNamespace(
                    path="blobs/sha256/partial-upload",
                    mtime=int(datetime.now(UTC).timestamp()),
                )
            ] if path == "blobs/sha256" else []

    catalog = CleanupCatalog()

    assert catalog.cleanup(dry_run=True) == []
    assert catalog.cleanup(dry_run=True, force=True) == ["blobs/sha256/partial-upload"]


def test_cleanup_accepts_force_flag() -> None:
    assert _parser().parse_args(["cleanup", "--force"]).force


def test_verbose_log_writes_to_standard_error(capsys: pytest.CaptureFixture[str]) -> None:
    output.configure(verbose=True)

    _log("list", "Loading scenes")

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == "list: Loading scenes\n"


def test_quiet_log_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    _log("list", "Loading scenes")

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_emit_json_prints_machine_readable_output(capsys: pytest.CaptureFixture[str]) -> None:
    args = _parser().parse_args(["--json", "list", "scenes"])

    _emit(args, {"results": "abc"}, "human text")

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"results": "abc"}
    assert "human text" not in captured.out


def test_emit_human_prints_plain_text(capsys: pytest.CaptureFixture[str]) -> None:
    args = _parser().parse_args(["list", "scenes"])

    _emit(args, {"results": "abc"}, Text("human text", style="green"))

    captured = capsys.readouterr()

    assert captured.out == "human text\n"


def test_verbose_flag() -> None:
    assert _parser().parse_args(["-v", "list", "scenes"]).verbose
    assert not _parser().parse_args(["list", "scenes"]).verbose


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("1", (1,)), ("1:4", (1, 2, 3, 4)), ("1:7:3,2", (1, 2, 4, 7))],
)
def test_frame_selection(expression: str, expected: tuple[int, ...]) -> None:
    assert parse_frames(expression) == expected


@pytest.mark.parametrize("expression", ["", "4:1", "1:4:0", "1::2"])
def test_invalid_frame_selection(expression: str) -> None:
    with pytest.raises(ValueError):
        parse_frames(expression)
