import pytest

from benchmarks.prepare_t2_fixture import select_subset
from erag.knowledge import split_semantic


def test_subset_preserves_all_positive_documents_and_labels_without_truncation():
    corpus = {"a": "1 概念定义", "b": "  补充\n规则  ", "c": "无关资料", "long": "长" * 40}
    queries = {"one": "比较", "excluded": "长文"}
    relevant = {"one": {"a", "b"}, "excluded": {"a", "long"}}
    args = dict(count=1, distractors=1, max_chars=20, seed=27)
    course, cases, metadata = select_subset(corpus, queries, relevant, **args)
    assert metadata["query_ids"] == ["one"]
    assert metadata["document_ids"] == ["a", "b", "c"]
    assert cases[0]["chapters"] == ["文档 a", "文档 b"]
    chunks = split_semantic(course)
    assert len(chunks) == 3
    assert {chapter for _, chapter, _ in chunks} == {"文档 a", "文档 b", "文档 c"}
    assert (course, cases, metadata) == select_subset(corpus, queries, relevant, **args)
    with pytest.raises(ValueError, match="完整保留"):
        select_subset(corpus, queries, relevant, count=2, max_chars=20)
