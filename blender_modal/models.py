"""Stable, JSON-serializable records used by the CLI and GPU workers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1
PREPARATION_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SceneFile:
    path: str
    sha256: str
    size: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneFile:
        return cls(path=str(value["path"]), sha256=str(value["sha256"]), size=int(value["size"]))


@dataclass(frozen=True, slots=True)
class SceneManifest:
    id: str
    entrypoint: str
    files: tuple[SceneFile, ...]
    created_at: str
    name: str | None = None
    schema_version: int = SCHEMA_VERSION
    preparation_version: int = PREPARATION_VERSION

    @classmethod
    def create(cls, entrypoint: str, files: list[SceneFile], name: str | None) -> SceneManifest:
        ordered = tuple(sorted(files, key=lambda file: file.path))
        identifier = digest(
            {
                "preparation_version": PREPARATION_VERSION,
                "entrypoint": entrypoint,
                "files": [asdict(file) for file in ordered],
            }
        )
        return cls(
            id=identifier, entrypoint=entrypoint, files=ordered, created_at=utc_now(), name=name
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["files"] = [asdict(file) for file in self.files]
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneManifest:
        return cls(
            id=str(value["id"]),
            entrypoint=str(value["entrypoint"]),
            files=tuple(SceneFile.from_dict(item) for item in value["files"]),
            created_at=str(value["created_at"]),
            name=value.get("name"),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
            preparation_version=int(value.get("preparation_version", PREPARATION_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class RenderSpec:
    scene_id: str
    backend: str
    samples: int | None
    tile_size: int | None
    resolution_x: int | None
    resolution_y: int | None
    resolution_percentage: int | None
    renderer_fingerprint: str
    output_format: str = "PNG"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RenderSpec:
        return cls(
            scene_id=str(value["scene_id"]),
            backend=str(value["backend"]),
            samples=_optional_int(value.get("samples")),
            tile_size=_optional_int(value.get("tile_size")),
            resolution_x=_optional_int(value.get("resolution_x")),
            resolution_y=_optional_int(value.get("resolution_y")),
            resolution_percentage=_optional_int(value.get("resolution_percentage")),
            renderer_fingerprint=str(value["renderer_fingerprint"]),
            output_format=str(value.get("output_format", "PNG")),
        )

    @property
    def id(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class JobManifest:
    id: str
    result_id: str
    spec: RenderSpec
    requested_frames: tuple[int, ...]
    gpu: str
    gpus_per_instance: int
    instances: int
    created_at: str
    app_id: str | None = None
    call_ids: tuple[str, ...] = ()
    status: str = "queued"
    cancelled: bool = False
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        payload["requested_frames"] = list(self.requested_frames)
        payload["call_ids"] = list(self.call_ids)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JobManifest:
        return cls(
            id=str(value["id"]),
            result_id=str(value["result_id"]),
            spec=RenderSpec.from_dict(value["spec"]),
            requested_frames=tuple(int(frame) for frame in value["requested_frames"]),
            gpu=str(value["gpu"]),
            gpus_per_instance=int(value["gpus_per_instance"]),
            instances=int(value["instances"]),
            created_at=str(value["created_at"]),
            app_id=value.get("app_id"),
            call_ids=tuple(str(item) for item in value.get("call_ids", [])),
            status=str(value.get("status", "queued")),
            cancelled=bool(value.get("cancelled", False)),
            completed_at=value.get("completed_at"),
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
