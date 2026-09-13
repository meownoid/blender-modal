from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from blender_modal import worker
from blender_modal.models import RenderSpec


@pytest.mark.parametrize("return_code", [0, 1])
@pytest.mark.filterwarnings("ignore:The render_shard function is executing locally:UserWarning")
def test_blender_logs_are_forwarded_without_changing_frame_publication(
    return_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    volume = Mock()
    monkeypatch.setattr(worker, "VOLUME_MOUNT", str(tmp_path))
    monkeypatch.setattr(worker.modal.Volume, "from_name", Mock(return_value=volume))
    worker._write_json(
        tmp_path / "scenes/scene-id/manifest.json",
        {"entrypoint": "scene.blend", "files": []},
    )
    event = 'BR {"type":"frame_completed","frame":1,"seconds":2.5}\n'
    lines = ["Blender starting\n", "BR invalid-json\n", event, "Final diagnostic"]
    process = Mock(stdout=iter(lines))
    process.wait.return_value = return_code
    process.poll.return_value = return_code
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 4, 3)

    def start_blender(command: list[str], **kwargs: object) -> Mock:
        config = json.loads(Path(command[-1]).read_text())
        (Path(config["output_dir"]) / "frame_000001.png").write_bytes(png)
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", start_blender)
    forwarded = Mock(wraps=print)
    monkeypatch.setattr(worker, "print", forwarded, raising=False)
    spec = RenderSpec("scene-id", "OPTIX", None, None, None, None, None, "fingerprint")
    args = ("test-volume", "job-id", "result-id", 2, [1], spec.to_dict())

    if return_code:
        with pytest.raises(RuntimeError, match="Blender exited with status 1"):
            worker.render_shard.local(*args)
    else:
        result = worker.render_shard.local(*args)
        assert result["status"] == "completed"
        assert result["completed_frames"] == [1]

    assert capsys.readouterr().out == "".join(f"[worker 2] {line}" for line in lines)
    assert forwarded.call_args_list == [
        call(f"[worker 2] {line}", end="", flush=True) for line in lines
    ]
    published = tmp_path / "results/result-id/frames/1"
    assert (published / "frame.png").read_bytes() == png
    metadata = json.loads((published / "metadata.json").read_text())
    assert metadata["worker_index"] == 2
    assert metadata["render_seconds"] == 2.5
    status = json.loads((tmp_path / "jobs/job-id/workers/2.json").read_text())
    assert status["status"] == ("failed" if return_code else "completed")
    assert status["completed_frames"] == [1]
    if return_code:
        assert status["error"] == "Blender exited with status 1"
    assert volume.commit.call_count == 3
