"""Fail fast on real chat, embedding and reranking protocol/semantic failures."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from erag import Settings
from erag.providers import Providers
from erag.schemas import Chunk, Hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    settings = Settings(_env_file=args.env_file)
    providers = Providers(settings)
    results = {}
    try:
        started = perf_counter()
        result = providers.json("计算输入的加法，只输出 JSON：sum(integer)。", {"a": 17, "b": 25})
        if result.get("sum") != 42:
            raise ValueError(f"结构化 LLM 检查失败: {result}")
        results["chat"] = {"result": result, "elapsed_ms": (perf_counter() - started) * 1000}
        documents = ["北京是中国的首都。", "鲸鱼生活在海洋中。"]
        started = perf_counter()
        vectors = providers.embed(documents)
        query = "中国的首都是哪里？"
        similarities = (providers.embed([query], query=True) @ vectors.T)[0]
        if int(np.argmax(similarities)) != 0:
            raise ValueError("Embedding 将不相关文档排在首都事实之前")
        results["embedding"] = {
            "shape": list(vectors.shape),
            "norms": np.linalg.norm(vectors, axis=1).tolist(),
            "similarities": similarities.tolist(),
            "elapsed_ms": (perf_counter() - started) * 1000,
        }
        candidates = [
            Hit(
                chunk=Chunk(id=str(i), content=text, source="health-check"),
                score=0,
                sub_query=query,
                route="check",
            )
            for i, text in enumerate(documents)
        ]
        started = perf_counter()
        ranked = providers.rerank(query, candidates)
        if ranked[0].chunk.id != "0":
            raise ValueError("Reranker 将不相关文档排在首都事实之前")
        results["reranker"] = {
            "ranking": [{"id": hit.chunk.id, "score": hit.score} for hit in ranked],
            "elapsed_ms": (perf_counter() - started) * 1000,
        }
        results["status"] = "passed"
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        providers.close()


if __name__ == "__main__":
    main()
