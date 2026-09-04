from __future__ import annotations

from pathlib import Path

import pytest

from blender_modal.catalog import CatalogError, build_scene
from blender_modal.frames import parse_frames


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
