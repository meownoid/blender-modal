from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from blender_modal import cli


def test_built_distribution_runs_without_the_checkout(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to test source and wheel builds")
    root = Path(__file__).resolve().parents[1]
    # Build through the sdist so missing files in either distribution are caught.
    subprocess.run(
        [uv, "build", "--out-dir", str(tmp_path / "dist")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = tmp_path / "installed"
    with ZipFile(next((tmp_path / "dist").glob("*.whl"))) as wheel:
        wheel.extractall(installed)
        assert not any("__pycache__" in name or "/.git/" in name for name in wheel.namelist())

    context = installed / "blender_modal" / "_image"
    for relative in ("Dockerfile", ".dockerignore", "pyproject.toml", "README.md"):
        assert (context / relative).read_bytes() == (root / relative).read_bytes()
    for source in (root / "blender_modal").glob("*.py"):
        assert (context / "blender_modal" / source.name).read_bytes() == source.read_bytes()

    # Use an unrelated working directory and explicitly select the wheel's code,
    # even when the test runner's interpreter has an editable checkout installed.
    script = """
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
import modal
from blender_modal import cli
from blender_modal.resources import image_context

context = Path(sys.argv[1]) / 'blender_modal' / '_image'
assert Path(cli.__file__).parent == context.parent
assert image_context() == context
assert cli._renderer_fingerprint() == sys.argv[2]
with patch.object(modal.Image, 'from_dockerfile', wraps=modal.Image.from_dockerfile) as build:
    from blender_modal import worker
    build.assert_called_once_with(context / 'Dockerfile', context_dir=context)
cli.main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed), cli._renderer_fingerprint()],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage: blender-modal" in result.stdout
