# 2026-11-26 实测记录

方法、抽样限制、置信区间和负结果见 [扩大评测报告](../../docs/expanded-evaluation.md)。

- `retrieval-summary.json`：T2 的 160 个测试查询、八个方案、配对 bootstrap。
- `retrieval-development.json`：在测试前用于选择方案的 40 查询开发集。
- `dialogue-summary.json`：MTRAG 六个测试会话、36 轮、六个核心方案，会话级 bootstrap。
- `dialogue-no-rerank*.json`：额外关闭重排的开发集与测试集对照。
- `blind-review-notes.json` / `blind-review-key.json`：先匿名审读、后揭晓的 18 份回答定性记录；不是独立人工实验。
- `chinese-review.json`：自编中文六轮真实模型回归的定性审读。
- `ui-followup-regression.json`：正式评测完成后的三例定向修复验证，不作为保留集成绩。
- `resource-revisions.json` / `frozen-*-sha256.json`：资源版本与历史冻结源码指纹；旧路径是当时布局。

逐题排名、完整回答、索引、失败迭代和对应源码快照保存在本地及服务器 `data/experiments/`，不提交较大的原始数据或模型。资源可按仓库中的下载器与复现指南重新获取。正式指标以冻结源码为准，不能直接归因于之后的所有交付改动。
