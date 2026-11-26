"""Download pinned models by default; fetch evaluation datasets only when requested."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import uuid
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).with_name("resource-manifest.json")


def downloaded_path(root: Path, name: str) -> str:
    lock = json.loads((root / "resources.lock.json").read_text(encoding="utf-8"))
    item = lock["resources"].get(name)
    if not item or not item.get("complete"):
        raise ValueError(f"{name} 尚未完整下载，请运行 erag-download --only {name}")
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_dir():
        raise ValueError(f"{name} 的下载目录无效")
    return str(path)


def read_manifest(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("resources"):
        raise ValueError("资源清单为空或版本不支持")
    resources = manifest["resources"]
    for item in resources:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", item["repo_id"]):
            raise ValueError("资源 repo_id 无效")
        if not re.fullmatch(r"[a-zA-Z0-9-]+", item["name"]):
            raise ValueError("资源名称无效")
        if not re.fullmatch(r"[0-9a-f]{40}", item["revision"]):
            raise ValueError("revision 必须固定到完整 commit SHA")
        if item["repo_type"] not in {"model", "dataset"}:
            raise ValueError("repo_type 必须为 model 或 dataset")
        if item.get("source", "huggingface") == "https":
            if not item.get("url", "").startswith("https://"):
                raise ValueError("文件资源必须使用 HTTPS")
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
                raise ValueError("文件资源必须提供 SHA256")
            if not re.fullmatch(r"[\w-][\w.-]*", item.get("filename", "")):
                raise ValueError("文件资源名称无效")
        elif item.get("source", "huggingface") != "huggingface":
            raise ValueError("未知的资源来源")
    if len({r["name"] for r in resources}) != len(resources):
        raise ValueError("资源名称重复")
    return resources


def download_file(resource, destination):
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / resource["filename"]
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ValueError("资源文件不能链接到下载目录外")
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == resource["sha256"]:
        return
    temporary = destination / f".download-{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(resource["url"], timeout=90) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                digest.update(block)
                stream.write(block)
        if digest.hexdigest() != resource["sha256"]:
            raise ValueError("下载文件 SHA256 不匹配，未发布资源")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_lock(root, lock):
    temporary = root / f".resources-{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(root / "resources.lock.json")
    finally:
        temporary.unlink(missing_ok=True)


def download(
    resources: list[dict], root: Path, *, metadata_only: bool = False, max_workers: int = 4, downloader=None
) -> dict:
    if downloader is None and any(r.get("source", "huggingface") == "huggingface" for r in resources):
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "resources.lock.json"
    lock = (
        json.loads(lock_path.read_text(encoding="utf-8"))
        if lock_path.exists()
        else {"version": 1, "resources": {}}
    )
    for resource in resources:
        relative = (
            Path(resource["repo_type"] + "s") / resource["repo_id"].split("/")[1] / resource["revision"]
        )
        destination = root / relative
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("资源目录不能链接到下载目录外")
        if not metadata_only and resource["name"] in lock["resources"]:
            lock["resources"][resource["name"]] = {**resource, "path": relative.as_posix(), "complete": False}
            save_lock(root, lock)
        print(f"下载 {resource['repo_id']} -> {destination}", flush=True)
        if resource.get("source") == "https":
            if not metadata_only:
                download_file(resource, destination)
        else:
            downloader(
                repo_id=resource["repo_id"],
                repo_type=resource["repo_type"],
                revision=resource["revision"],
                local_dir=str(destination),
                max_workers=max_workers,
                allow_patterns=["README.md", "LICENSE*", "config.json"] if metadata_only else None,
            )
        previous = lock["resources"].get(resource["name"], {})
        complete = not metadata_only or (
            previous.get("revision") == resource["revision"] and previous.get("complete", False)
        )
        lock["resources"][resource["name"]] = {**resource, "path": relative.as_posix(), "complete": complete}
        save_lock(root, lock)
    return lock


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=Path("resources"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--only", nargs="+", help="llm embedding reranker retrieval-data retrieval-labels mtrag；默认仅模型"
    )
    selection.add_argument("--all", action="store_true", help="下载全部模型和可选评测数据")
    parser.add_argument("--dry-run", action="store_true", help="只显示下载计划，不访问网络")
    parser.add_argument("--metadata-only", action="store_true", help="只下载说明、许可证和配置，用于验证连接")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    resources = read_manifest(args.manifest)
    if args.only:
        unknown = set(args.only) - {r["name"] for r in resources}
        if unknown:
            parser.error(f"未知资源: {sorted(unknown)}")
        resources = [r for r in resources if r["name"] in args.only]
    elif not args.all:
        resources = [r for r in resources if r["repo_type"] == "model"]
    if not 1 <= args.max_workers <= 32:
        parser.error("max-workers 必须在 1 到 32 之间")
    print(
        json.dumps(
            {
                "root": str(args.root.resolve()),
                "metadata_only": args.metadata_only,
                "full_download_gb": round(sum(r.get("bytes", 0) for r in resources) / 1e9, 3),
                "resources": resources,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.dry_run:
        download(resources, args.root, metadata_only=args.metadata_only, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
