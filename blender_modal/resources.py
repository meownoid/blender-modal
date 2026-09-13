"""Locate renderer inputs in both source checkouts and installed wheels."""

from pathlib import Path


def image_context() -> Path:
    package = Path(__file__).resolve().parent
    bundled = package / "_image"
    return bundled if bundled.is_dir() else package.parent
