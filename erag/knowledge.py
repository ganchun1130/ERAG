"""Knowledge ingestion and retrieval primitives for the thesis-aligned ERAG flow.

The store deliberately keeps plain JSON metadata next to NumPy/FAISS vectors. This
makes an index inspectable, portable, and safe to rebuild when the embedding model
changes (the manifest contains the embedding identity).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

from .providers import Providers
from .schemas import Chunk, Hit
from .settings import Settings


def read_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PDF 解析需要 pypdf，请安装项目依赖") from exc
        pages = []
        for number, page in enumerate(PdfReader(str(path)).pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[第{number}页]\n{text}")
        return "\n\n".join(pages)
    raise ValueError(f"不支持的文件格式: {suffix}（支持 PDF/TXT/MD/CSV）")


def split_semantic(text: str, *, max_chars: int = 900, overlap: int = 120) -> list[tuple[str, str, int | None]]:
    """A deterministic semantic baseline: preserve headings, paragraphs and page marks.

    LLM enrichment is optional; this splitter is intentionally usable offline and avoids
    the old fixed-character truncation that could cut formulas or definitions in half.
    """
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    units = [u.strip() for u in re.split(r"\n\s*\n+", text) if u.strip()]
    chunks: list[tuple[str, str, int | None]] = []
    current: list[str] = []
    chapter = "未分类"
    page: int | None = None
    for unit in units:
        page_match = re.search(r"\[第(\d+)页\]", unit)
        if page_match:
            page = int(page_match.group(1))
            unit = re.sub(r"\[第\d+页\]\s*", "", unit).strip()
        heading = re.match(r"^(#{1,6}|第[一二三四五六七八九十0-9]+章|[0-9]+(?:\.[0-9]+)*\s+).+", unit)
        if heading:
            chapter = heading.group(0).strip().lstrip("# ")[:120]
        if sum(map(len, current)) + len(unit) + 1 > max_chars and current:
            body = "\n\n".join(current).strip()
            chunks.append((body, chapter, page))
            tail = body[-overlap:] if overlap else ""
            current = [tail] if tail else []
        current.append(unit)
    if current:
        chunks.append(("\n\n".join(current).strip(), chapter, page))
    return chunks


class KnowledgeBase:
    def __init__(self, root: Path, name: str, providers: Providers, settings: Settings):
        self.root = Path(root) / name
        self.name, self.providers, self.settings = name, providers, settings
        self.chunks: list[Chunk] = []
        self.summary_vectors: np.ndarray | None = None
        self.tag_vectors: np.ndarray | None = None
        self.qa_vectors: np.ndarray | None = None

    @property
    def manifest_path(self): return self.root / "manifest.json"

    def _index(self, vectors: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(vectors))
        if faiss is not None:
            idx = faiss.IndexFlatIP(vectors.shape[1]); idx.add(vectors)
            return idx.search(query, k)
        scores = query @ vectors.T
        order = np.argsort(-scores[0])[:k]
        return scores[:, order], order.reshape(1, -1)

    def build(self, chunks: Iterable[Chunk]) -> int:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("没有可写入知识库的内容")
        summaries = [c.summary or c.content for c in self.chunks]
        tags = ["；".join(c.tags) or c.content[:100] for c in self.chunks]
        # Keep a separate QA/global view. QA chunks are intentionally broad and are
        # searched with key_query during contextual retrieval.
        qa = [c.content for c in self.chunks if c.kind == "qa"] or summaries
        self.summary_vectors = self.providers.embed(summaries)
        self.tag_vectors = self.providers.embed(tags)
        self.qa_vectors = self.providers.embed(qa)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "chunks.json").write_text(json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(self.root / "summary.npy", self.summary_vectors)
        np.save(self.root / "tags.npy", self.tag_vectors)
        np.save(self.root / "qa.npy", self.qa_vectors)
        self.manifest_path.write_text(json.dumps({"version": 2, "name": self.name,
            "embedding": self.settings.embedding_identity(), "chunk_count": len(self.chunks)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(self.chunks)

    def load(self) -> "KnowledgeBase":
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"知识库不存在或未完成构建: {self.name}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding") != self.settings.embedding_identity():
            raise ValueError("知识库使用的 embedding 与当前配置不一致，请重新构建（避免检索失真）")
        self.chunks = [Chunk.model_validate(x) for x in json.loads((self.root / "chunks.json").read_text(encoding="utf-8"))]
        self.summary_vectors = np.load(self.root / "summary.npy")
        self.tag_vectors = np.load(self.root / "tags.npy")
        self.qa_vectors = np.load(self.root / "qa.npy")
        return self

    def search(self, query: str, *, key_query: str | None = None, chapter: str = "", top_k: int = 8) -> list[Hit]:
        if self.summary_vectors is None: self.load()
        assert self.summary_vectors is not None and self.tag_vectors is not None
        q = self.providers.embed([query], query=True)
        scores, indices = self._index(self.summary_vectors, q, min(top_k * 3, len(self.chunks)))
        key_scores = None
        if key_query:
            kq = self.providers.embed([key_query], query=True)
            key_scores, _ = self._index(self.tag_vectors, kq, len(self.chunks))
            key_scores = (kq @ self.tag_vectors.T)[0]
        hits: list[Hit] = []
        for score, idx in zip(scores[0], indices[0]):
            c = self.chunks[int(idx)]
            if chapter and c.chapter and chapter not in c.chapter and c.chapter not in chapter:
                continue
            if float(score) < self.settings.summary_min_score:
                continue
            if key_scores is not None and float(key_scores[int(idx)]) < self.settings.tag_min_score:
                continue
            blended = float(score) * 0.7 + (float(key_scores[int(idx)]) * 0.3 if key_scores is not None else 0)
            hits.append(Hit(chunk=c, score=blended, sub_query=query, route=c.chapter or "global"))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]

    def list_names(self) -> list[str]:
        base = Path(self.settings.knowledge_base_root)
        return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "manifest.json").exists()) if base.exists() else []
