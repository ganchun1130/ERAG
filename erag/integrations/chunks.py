"""Accept user-prepared Chunk JSONL alongside the standard document formats."""

import json

from erag.extensions import Extensions
from erag.ingestion import DocumentPipeline
from erag.schemas import Chunk


class JsonlPipeline:
    def prepare(self, files, settings):
        standard = DocumentPipeline()
        for path in dict.fromkeys(p.resolve() for p in files):
            if path.suffix.lower() == ".jsonl":
                with path.open(encoding="utf-8-sig") as stream:
                    for number, line in enumerate(stream, 1):
                        if line.strip():
                            try:
                                yield Chunk.model_validate(json.loads(line))
                            except ValueError as exc:
                                raise ValueError(f"{path.name} 第 {number} 行不是有效的 Chunk JSON") from exc
            else:
                yield from standard.prepare([path], settings)


def create_extensions(settings):
    return Extensions(
        ingestion=JsonlPipeline(),
        upload_extensions=[".pdf", ".txt", ".md", ".markdown", ".csv", ".jsonl"],
    )
