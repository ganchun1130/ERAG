from erag.knowledge import split_semantic


def test_split_semantic_keeps_page_and_heading_metadata():
    text = "# 第一章 基础\n\n[第2页]\n定义与原理。\n\n应用场景与限制。"
    chunks = split_semantic(text, max_chars=1000)
    assert len(chunks) == 1
    content, chapter, page = chunks[0]
    assert "定义与原理" in content
    assert chapter.startswith("第一章")
    assert page == 2


def test_split_semantic_splits_large_paragraphs_with_overlap():
    text = "\n\n".join([f"段落{i}：" + "内容" * 40 for i in range(4)])
    chunks = split_semantic(text, max_chars=100, overlap=10)
    assert len(chunks) > 1
    assert all(item[0] for item in chunks)
