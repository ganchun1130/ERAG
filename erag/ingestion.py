"""Default ingestion with independently replaceable readers and splitters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from .knowledge import read_document, split_semantic
from .schemas import Chunk
from .settings import Settings


class DocumentPipeline:
    def __init__(self, reader: Callable[[Path], str] = read_document, splitter: Callable = split_semantic):
        self.reader, self.splitter = reader, splitter

    def prepare(self, files: list[Path], settings: Settings) -> Iterable[Chunk]:
        seen = set()
        for path in dict.fromkeys(p.resolve() for p in files):
            text = self.reader(path)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"文件没有可提取文本: {path.name}；扫描版 PDF 需要先做 OCR")
            document_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if document_id in seen:
                continue
            seen.add(document_id)
            parts = list(
                self.splitter(text, max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap)
            )
            if not parts:
                raise ValueError(f"文件没有可提取文本: {path.name}")
            for i, (content, chapter, page) in enumerate(parts):
                identity = json.dumps([document_id, i, page, chapter, content], ensure_ascii=False)
                yield Chunk(
                    id=hashlib.sha256(identity.encode()).hexdigest(),
                    content=content,
                    summary=content,
                    tags=[chapter] if chapter else [],
                    source=str(path),
                    chapter=chapter,
                    page=page,
                )
