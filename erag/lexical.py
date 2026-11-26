"""Small dependency-free BM25 index for exact terms and Chinese character bigrams."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np


def tokens(text: str) -> list[str]:
    result = []
    for word in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        result.extend(
            [word]
            if not re.match(r"[\u4e00-\u9fff]", word) or len(word) == 1
            else [word[i : i + 2] for i in range(len(word) - 1)]
        )
    return result


class BM25:
    def __init__(self, documents: list[str]):
        postings = defaultdict(list)
        lengths = []
        for index, text in enumerate(documents):
            counts = Counter(tokens(text))
            lengths.append(sum(counts.values()))
            for term, count in counts.items():
                postings[term].append((index, count))
        self.size = len(documents)
        lengths = np.asarray(lengths, dtype=np.float32)
        average = float(lengths.mean()) if len(lengths) else 1.0
        self.norm = 1.2 * (0.25 + 0.75 * lengths / max(average, 1.0))
        self.postings = {}
        for term, entries in postings.items():
            ids, frequencies = np.asarray(entries).T
            self.postings[term] = (
                ids,
                frequencies.astype(np.float32),
                math.log(1 + (self.size - len(ids) + 0.5) / (len(ids) + 0.5)),
            )

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float32)
        for term in set(tokens(query)):
            if term in self.postings:
                ids, frequencies, idf = self.postings[term]
                result[ids] += idf * frequencies * 2.2 / (frequencies + self.norm[ids])
        return result


def reciprocal_rank_fusion(rankings: list[list[int]], *, constant: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(dict.fromkeys(ranking), 1):
            scores[index] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
