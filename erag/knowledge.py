"""Document splitting, atomic index snapshots and batched exact retrieval."""

from __future__ import annotations

import csv
import json
import re
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .lexical import BM25, reciprocal_rank_fusion
from .providers import Providers
from .schemas import Chunk, Hit, SubQuery
from .settings import Settings

PAGE = re.compile(r"^\[第(\d+)页\]\s*$")
HEADING = re.compile(r"^(?:#{1,6}\s+|第[一二三四五六七八九十百零〇0-9]+[章节]\s*|\d+(?:\.\d+)*\s+)\S.*")


def read_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Keep column names with each record so chunks retain their meaning.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            headers = next(rows, [])
            return "\n\n".join(
                "；".join(
                    f"{headers[i] if i < len(headers) else f'列{i + 1}'}：{value}"
                    for i, value in enumerate(row)
                )
                for row in rows
                if any(value.strip() for value in row)
            )
    if suffix in {".txt", ".md", ".markdown"}:
        # Failing explicitly is preferable to silently dropping textbook characters.
        return path.read_text(encoding="utf-8-sig")
    if suffix == ".pdf":
        from pypdf import PdfReader

        pages = []
        for number, page in enumerate(PdfReader(str(path)).pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[第{number}页]\n{text}")
        return "\n\n".join(pages)
    raise ValueError(f"不支持的文件格式: {suffix}（支持 PDF/TXT/MD/CSV）")


def split_semantic(
    text: str, *, max_chars: int = 900, overlap: int = 120
) -> list[tuple[str, str, int | None]]:
    """Prefer paragraph/sentence boundaries, with a hard length bound.

    Page and chapter changes flush the previous section before updating metadata.
    Overlap applies only inside a section, never across page/chapter boundaries.
    """
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("分块参数必须满足 max_chars > 0 且 0 <= overlap < max_chars")
    chunks: list[tuple[str, str, int | None]] = []
    section: list[str] = []
    chapter, page = "未分类", None

    def flush() -> None:
        body = "\n".join(section).strip()
        start = 0
        while start < len(body):
            end = min(len(body), start + max_chars)
            if end < len(body):
                # Avoid tiny chunks and guarantee progress even with large overlap.
                minimum = max(max_chars // 2, overlap + 1)
                boundaries = list(re.finditer(r"\n\s*\n|[。！？!?；;]\s*|\.\s+|\n", body[start:end]))
                end = next((start + m.end() for m in reversed(boundaries) if m.end() >= minimum), end)
            content = body[start:end].strip()
            if content:
                chunks.append((content, chapter, page))
            if end == len(body):
                break
            start = end - overlap
            while start < len(body) and body[start].isspace():
                start += 1
        section.clear()

    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = line.strip()
        if match := PAGE.fullmatch(stripped):
            # A heading immediately before the first PDF marker belongs to that page.
            if page is not None or any(s.strip() and not HEADING.match(s.strip()) for s in section):
                flush()
            page = int(match.group(1))
        elif HEADING.match(stripped):
            flush()
            chapter = stripped.lstrip("# ")[:120]
            section.append(line)
        else:
            section.append(line)
    flush()
    return chunks


def validate_name(name: str) -> str:
    if (
        not name
        or name != name.strip()
        or name.startswith(".")
        or any(c in name for c in "/\\:")
        or any(ord(c) < 32 for c in name)
    ):
        raise ValueError("知识库名称不能为空、以点开头或包含路径分隔符/控制字符")
    return name


def list_knowledge_bases(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and not p.is_symlink()
        and not p.name.startswith(".")
        and (p / "manifest.json").is_file()
    )


class KnowledgeBase:
    def __init__(self, root: Path, name: str, providers: Providers, settings: Settings):
        self.name = validate_name(name)
        base = Path(root).resolve()
        self.root = base / self.name
        if self.root.is_symlink() or self.root.resolve().parent != base:
            raise ValueError("知识库目录不能链接到其他路径")
        self.providers, self.settings = providers, settings
        self.chunks: list[Chunk] = []
        self.summary_vectors: np.ndarray | None = None
        self.tag_vectors: np.ndarray | None = None
        self.content_vectors: np.ndarray | None = None
        self._lexical: BM25 | None = None
        self.manifest: dict = {}

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def location(self) -> str:
        return str(self.root)

    @property
    def chapters(self) -> list[str]:
        return sorted({c.chapter for c in self.chunks if c.chapter})

    def embedding_identity(self) -> dict:
        identity = getattr(self.providers, "embedding_identity", self.settings.embedding_identity)
        return identity()

    def cache_token(self) -> tuple:
        stat = self.manifest_path.stat()
        return (
            str(self.root),
            stat.st_mtime_ns,
            stat.st_size,
            stat.st_ino,
            json.dumps(self.embedding_identity(), sort_keys=True),
        )

    @staticmethod
    def _validate_vectors(vectors: np.ndarray, count: int) -> np.ndarray:
        if (
            vectors.ndim != 2
            or vectors.shape[0] != count
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("知识库向量的数量、维度或数值无效，请重新构建")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("知识库向量未归一化，请重新构建")
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def build(self, chunks: Iterable[Chunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            raise ValueError("没有可写入知识库的内容；扫描版 PDF 需要先做 OCR")
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("知识块 ID 重复")
        content_texts = [c.content for c in chunks]
        content = self._validate_vectors(self.providers.embed(content_texts), len(chunks))
        summary_texts = [c.summary or c.content for c in chunks]
        summaries = (
            content
            if summary_texts == content_texts
            else self._validate_vectors(self.providers.embed(summary_texts), len(chunks))
        )
        tags = self._validate_vectors(
            self.providers.embed(["；".join(c.tags) or c.content[:100] for c in chunks]), len(chunks)
        )
        if summaries.shape != tags.shape or content.shape != summaries.shape:
            raise ValueError("摘要和标签向量维度不一致")

        # Immutable files + one atomic manifest replacement: readers see one full build.
        snapshot = uuid.uuid4().hex
        directory = self.root / "snapshots" / snapshot
        if not directory.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("知识库快照路径无效")
        directory.mkdir(parents=True)
        temporary_manifest = self.root / f".manifest-{snapshot}.tmp"
        try:
            (directory / "chunks.json").write_text(
                json.dumps([c.model_dump() for c in chunks], ensure_ascii=False), encoding="utf-8"
            )
            np.save(directory / "summary.npy", summaries, allow_pickle=False)
            np.save(directory / "tags.npy", tags, allow_pickle=False)
            np.save(directory / "content.npy", content, allow_pickle=False)
            manifest = {
                "version": 4,
                "name": self.name,
                "snapshot": snapshot,
                "embedding": self.embedding_identity(),
                "chunk_count": len(chunks),
                "dimensions": summaries.shape[1],
            }
            temporary_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_manifest.replace(self.manifest_path)
        except Exception:
            temporary_manifest.unlink(missing_ok=True)
            shutil.rmtree(directory, ignore_errors=True)
            raise
        self.chunks, self.summary_vectors, self.tag_vectors = chunks, summaries, tags
        self.content_vectors, self._lexical = content, None
        self.manifest = manifest
        return len(chunks)

    def load(self) -> KnowledgeBase:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"知识库不存在或未完成构建: {self.name}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") not in {2, 3, 4}:
            raise ValueError("不支持的知识库版本，请重新构建")
        if manifest.get("embedding") != self.embedding_identity():
            raise ValueError("知识库使用的 embedding 与当前配置不一致，请重新构建（避免检索失真）")
        directory = self.root
        if manifest["version"] >= 3:
            snapshot = manifest.get("snapshot", "")
            if not isinstance(snapshot, str) or not re.fullmatch(r"[0-9a-f]{32}", snapshot):
                raise ValueError("知识库快照标识无效")
            directory = self.root / "snapshots" / snapshot
            if not directory.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("知识库快照路径无效")
        chunks = [
            Chunk.model_validate(x)
            for x in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        ]
        if (
            not chunks
            or len(chunks) != manifest.get("chunk_count")
            or len({c.id for c in chunks}) != len(chunks)
        ):
            raise ValueError("知识库块数量或 ID 不一致，请重新构建")
        summaries = self._validate_vectors(
            np.load(directory / "summary.npy", allow_pickle=False), len(chunks)
        )
        tags = self._validate_vectors(np.load(directory / "tags.npy", allow_pickle=False), len(chunks))
        if (
            summaries.shape != tags.shape
            or manifest.get("dimensions", summaries.shape[1]) != summaries.shape[1]
        ):
            raise ValueError("知识库向量维度不一致，请重新构建")
        self.chunks, self.summary_vectors, self.tag_vectors = chunks, summaries, tags
        self.content_vectors = None
        if manifest["version"] >= 4:
            self.content_vectors = self._validate_vectors(
                np.load(directory / "content.npy", allow_pickle=False), len(chunks)
            )
            if self.content_vectors.shape != summaries.shape:
                raise ValueError("原文向量维度不一致，请重新构建")
        self._lexical = None
        self.manifest = manifest
        return self

    @staticmethod
    def _top(scores: np.ndarray, eligible: np.ndarray, threshold: float, k: int) -> np.ndarray:
        candidates = eligible[scores[eligible] >= threshold]
        if len(candidates) > k:
            candidates = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        return candidates[np.argsort(-scores[candidates], kind="stable")]

    def _candidates(
        self, summary_scores: np.ndarray, tag_scores: np.ndarray | None, eligible: np.ndarray, k: int
    ) -> np.ndarray:
        selected = self._top(summary_scores, eligible, self.settings.summary_min_score, k)
        if tag_scores is not None:
            selected = np.union1d(selected, self._top(tag_scores, eligible, self.settings.tag_min_score, k))
        return selected

    def search_many(
        self,
        queries: list[SubQuery],
        *,
        key_query: str | None = None,
        top_k: int = 8,
        embedding_cache: dict[str, np.ndarray] | None = None,
    ) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        if not queries:
            return []
        if self.summary_vectors is None:
            self.load()
        assert self.summary_vectors is not None and self.tag_vectors is not None
        if self.settings.retrieval_mode != "dual" and self.content_vectors is None:
            raise ValueError("原文/混合检索需要 v4 索引，请重新构建知识库，或设置 ERAG_RETRIEVAL_MODE=dual")
        cache = embedding_cache if embedding_cache is not None else {}
        texts = list(dict.fromkeys([s.query for s in queries] + ([key_query] if key_query else [])))
        missing = [text for text in texts if text not in cache]
        if missing:
            vectors = self._validate_vectors(self.providers.embed(missing, query=True), len(missing))
            if vectors.shape[1] != self.summary_vectors.shape[1]:
                raise ValueError("查询向量与索引维度不一致，请检查 embedding 服务并重新构建")
            cache.update(zip(missing, vectors, strict=True))
        scores = np.stack([cache[s.query] for s in queries]) @ self.summary_vectors.T
        tag_scores = cache[key_query] @ self.tag_vectors.T if key_query else None
        all_indices = np.arange(len(self.chunks))
        hits: list[Hit] = []
        for sub, summary_scores in zip(queries, scores, strict=True):
            if self.settings.retrieval_mode != "dual":
                # Global recall avoids an uncertain planner chapter becoming a hard exclusion.
                raw_scores = cache[sub.query] @ self.content_vectors.T
                chosen = self._top(raw_scores, all_indices, self.settings.summary_min_score, top_k)
                if self.settings.retrieval_mode == "hybrid":
                    if self._lexical is None:
                        self._lexical = BM25([c.content for c in self.chunks])
                    lexical_scores = self._lexical.scores(sub.query)
                    lexical = self._top(lexical_scores, all_indices, np.finfo(np.float32).tiny, top_k)
                    fused = reciprocal_rank_fusion([chosen.tolist(), lexical.tolist()])[:top_k]
                else:
                    fused = [(int(i), float(raw_scores[i])) for i in chosen]
                hits.extend(
                    Hit(
                        chunk=self.chunks[i],
                        score=score,
                        sub_query=sub.query,
                        route=self.settings.retrieval_mode,
                    )
                    for i, score in fused
                )
                continue
            eligible = np.array(
                [
                    i
                    for i, c in enumerate(self.chunks)
                    if sub.chapter and c.chapter and (sub.chapter in c.chapter or c.chapter in sub.chapter)
                ],
                dtype=int,
            )
            routed = bool(len(eligible))
            if not routed:
                eligible = all_indices

            selected = self._candidates(summary_scores, tag_scores, eligible, top_k)
            if not len(selected) and routed:
                selected = self._candidates(summary_scores, tag_scores, all_indices, top_k)
                routed = False
            # Independent recall channels: a tag can recover evidence missed by summaries.
            blended = summary_scores if tag_scores is None else 0.7 * summary_scores + 0.3 * tag_scores
            selected = selected[np.argsort(-blended[selected], kind="stable")][:top_k]
            hits.extend(
                Hit(
                    chunk=self.chunks[int(i)],
                    score=float(blended[i]),
                    sub_query=sub.query,
                    route=f"chapter:{sub.chapter}" if routed else "global",
                )
                for i in selected
            )
        return hits

    def search(
        self, query: str, *, key_query: str | None = None, chapter: str = "", top_k: int = 8
    ) -> list[Hit]:
        return self.search_many([SubQuery(query=query, chapter=chapter)], key_query=key_query, top_k=top_k)
