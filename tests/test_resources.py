import json
from pathlib import Path

import pytest

from erag.resources import DEFAULT_MANIFEST, download, downloaded_path, read_manifest
from erag.services.launch_local_models import launch_spec, main


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], {"llm", "embedding", "reranker"}),
        (["--only", "mtrag"], {"mtrag"}),
        (["--all"], {"llm", "embedding", "reranker", "retrieval-data", "retrieval-labels", "mtrag"}),
    ],
)
def test_download_plan_does_not_fetch_datasets_implicitly(arguments, expected, capsys):
    from erag.resources import main as download_main

    download_main([*arguments, "--dry-run"])
    assert {r["name"] for r in json.loads(capsys.readouterr().out)["resources"]} == expected


def test_manifest_pins_every_model_and_dataset():
    resources = read_manifest(DEFAULT_MANIFEST)
    assert {r["name"] for r in resources} == {
        "llm",
        "embedding",
        "reranker",
        "retrieval-data",
        "retrieval-labels",
        "mtrag",
    }


def test_download_is_versioned_and_partial_download_cannot_launch(tmp_path):
    resource = {"name": "embedding", "repo_id": "Qwen/model", "repo_type": "model", "revision": "a" * 40}
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

    download([resource], tmp_path, metadata_only=True, downloader=downloader)
    with pytest.raises(ValueError, match="尚未完整下载"):
        downloaded_path(tmp_path, "embedding")
    download([resource], tmp_path, downloader=downloader)
    assert downloaded_path(tmp_path, "embedding").endswith("a" * 40)
    assert calls[-1]["revision"] == "a" * 40
    assert calls[-1]["allow_patterns"] is None
    download([resource], tmp_path, metadata_only=True, downloader=downloader)
    assert json.loads((tmp_path / "resources.lock.json").read_text())["resources"]["embedding"]["complete"]


def test_failed_download_never_marks_resource_complete(tmp_path):
    def fail(**kwargs):
        raise OSError("network error")

    resource = {"name": "llm", "repo_id": "Qwen/model", "repo_type": "model", "revision": "a" * 40}
    with pytest.raises(OSError):
        download([resource], tmp_path, downloader=fail)
    assert not (tmp_path / "resources.lock.json").exists()


def test_launch_uses_registered_name_and_explicit_uid(capsys):
    main(["--model-path", "/models/Qwen3-8B", "--dry-run"])
    spec = json.loads(capsys.readouterr().out)[0]
    assert spec["model_name"] == "qwen3"
    assert spec["model_uid"] == "qwen3-instruct"
    assert spec["model_size_in_billions"] == "8"
    embedding = launch_spec(name="Qwen3-Embedding-0.6B", uid="em", path="/models/em", model_type="embedding")
    assert "model_engine" not in embedding


def test_launch_can_place_models_on_separate_cards(capsys):
    main(
        [
            "--model-path",
            "/models/llm",
            "--embedding-path",
            "/models/embedding",
            "--reranker-path",
            "/models/reranker",
            "--llm-gpu",
            "0",
            "--embedding-gpu",
            "1",
            "--reranker-gpu",
            "2",
            "--dry-run",
        ]
    )
    specs = json.loads(capsys.readouterr().out)
    assert [spec["gpu_idx"] for spec in specs] == [[0], [1], [2]]
    assert all(spec["n_gpu"] == 1 for spec in specs)


def test_pinned_https_download_verifies_checksum_before_publication(tmp_path, monkeypatch):
    import hashlib
    import io

    from erag.resources import download_file

    content = b"public benchmark"
    resource = {
        "filename": "data.json",
        "url": "https://example.invalid/data.json",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(content))
    download_file(resource, tmp_path)
    assert (tmp_path / "data.json").read_bytes() == content
    (tmp_path / "data.json").unlink()
    resource["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        download_file(resource, tmp_path)
    assert not (tmp_path / "data.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_redownload_invalidates_old_complete_marker(tmp_path):
    resource = {"name": "llm", "repo_id": "Qwen/model", "repo_type": "model", "revision": "a" * 40}
    download([resource], tmp_path, downloader=lambda **kw: Path(kw["local_dir"]).mkdir(parents=True))

    def fail(**kwargs):
        raise OSError("offline")

    with pytest.raises(OSError):
        download([resource], tmp_path, downloader=fail)
    assert not json.loads((tmp_path / "resources.lock.json").read_text())["resources"]["llm"]["complete"]
