# 评测与消融

目标是判断一个改动是否改善教学问答，而不是让流程看起来更复杂。先评价检索能否找到证据，再评价模型能否正确利用证据。

## 1. 准备教学测试集

建议从真实课程资料中人工挑选 100–200 个问题作为起点，包含定义、比较、跨章节组合、多轮指代和无法回答的问题。这个数量是项目起步建议，不是统计功效保证。

先按文档/主题划分开发集与保留测试集，再生成或改写问题，防止相同事实、相似改写泄漏到两个集合。LLM 可以协助起草问题，但参考答案和证据关联应由人核对。不能仅让同一个回答模型给自己打分。

当前 `evaluate` 测量**单轮检索**，输入为 JSONL，每行包含一个可回答问题和一组相关块 ID：

```json
{"query":"形成性评价与总结性评价分别在何时发挥作用？","relevant_chunk_ids":["实际形成性评价块ID","实际总结性评价块ID"]}
```

在知识库 manifest 指向的 `snapshots/<版本>/chunks.json` 中查看块 ID 和原文；v2 索引的 `chunks.json` 位于知识库根目录。标注应涵盖所有必要证据，不能只把当前检索出的第一条标成答案。改变分块规则后重新核对标注；不存在的 ID 会使评测在调用服务前失败。

无法回答、多轮和跨章节完整性问题应放入单独的端到端测试集。当前单轮指标要求至少一个相关块，不混入无答案样本。

## 2. 保持索引与预算相同，比较检索方案

以下命令需要可用的 embedding 服务；只有显式指定 `--rerank` 才调用 reranker，不调用规划或回答 LLM：

```bash
# A：只用摘要向量
python -m erag.cli evaluate --kb 计算教育学 --cases data/eval/teaching.jsonl \
  --summary-only --top-k 5 --output data/eval/summary.json

# B：摘要 + 标签独立召回
python -m erag.cli evaluate --kb 计算教育学 --cases data/eval/teaching.jsonl \
  --top-k 5 --output data/eval/dual.json

# C：双路召回 + 重排
python -m erag.cli evaluate --kb 计算教育学 --cases data/eval/teaching.jsonl \
  --top-k 5 --rerank --output data/eval/dual-rerank.json
```

三个实验使用相同的 `max(top_k, ERAG_FINAL_TOP_K)` 候选预算。阈值不同也会影响结果，报告中会保存阈值、模型、embedding 身份和实际读取的索引快照。不要用不同索引的结果直接声称某个检索方法更好。

| 输出指标 | 定义 | 用途 |
|---|---|---|
| recall | 前 k 个块命中的相关块数 / 全部标注相关块数 | 证据是否齐全 |
| precision | 前 k 个块命中的相关块数 / k | 固定上下文位置的有效利用率；不足 k 条仍以 k 为分母 |
| hit_rate | 是否至少命中一个相关块，对样本取平均 | 发现彻底漏召回 |
| mrr | 第一个相关块排名的倒数，无命中为 0 | 第一条有效证据出现多早 |
| ndcg | 二元相关性 DCG 与理想前 k 排名之比 | 多个相关证据的排序质量 |
| latency_ms.p50 / p95 | 每题检索与可选重排耗时 | 包含查询 API 时间，不包含初始知识库加载 |

报告同时记录逐题返回 ID、JSONL SHA256 和配置。这里没有 LLM 自动评审分数。服务失败会终止该实验，不悄悄降级后继续计算指标。

## 3. 端到端评测

保留相同语料、问题、模型和提示版本，分别设置 `ERAG_TCT_ENABLED=false`、`ERAG_MAX_RETRIEVAL_ROUNDS=1` 和 `3`。用 `ask --json` 保存来源、计划、轨迹、警告，并由人工核对：

- 是否回答原问题的全部子任务，是否保留多轮指代对象。
- 每个事实是否由引用原文支持，来源编号和页码是否正确。
- 无相关资料时是否明确说明不足，是否把自己的常识当成教材证据。
- 每轮新增了什么证据，以及总延迟和实际模型请求费用。

不要把“审查模型说 complete=true”当成人工正确性标签。规划器可能误判不需检索，引用编号合法也可能引用了不支持论断的文本。必要时将检索策略改成强制检索，但应先用误判样本确认收益。

对检索消融可以比较逐题差异或配对 bootstrap；对于主观答案质量，至少准备独立核对或双人标注。小幅平均提升不一定稳定，应保留失败样本和不确定性。

## 4. 公开数据与领域数据的关系

资源脚本可下载 [C-MTEB/T2Retrieval](https://huggingface.co/datasets/C-MTEB/T2Retrieval) 及相关性标注。固定版本的 corpus/queries 字段是 `id,text`，qrels 字段是 `qid,pid,score`，文件为 Parquet。

这是原始 benchmark 数据，不会自动成为 ERAG 教学知识库。若接入本项目评测，应保留 passage ID 与 `Chunk.id` 的映射，用正相关 qrels 生成 `relevant_chunk_ids`；若把原 passage 再分块，需要重新核对证据级标签，不能无条件把所有子块都视为相关。采样测试必须记录查询选择、负例选择、随机种子和样本量，不能把小候选子集上的分数冒充完整官方 benchmark 结果。

现有 `python -m benchmarks.prepare_t2_fixture` 提供固定抽样转换：只选全部正例都能完整保留为单块的短文档查询，保留全部正例，再加入随机干扰文档；原始 passage ID 写入章节名，运行时映射到稳定 Chunk ID。需安装 `.[benchmark]`。完整命令和实际运行结果分别见 [远程验证](remote-validation.md) 与 [实验报告](real-model-results.md)。

优先使用自己的教学语料验证产品目标，再用公开数据检查方法是否过拟合。OCR、公式、教学表达适配等问题不能由通用网页检索分数代替。
