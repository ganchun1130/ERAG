# 知识库构建工具使用指南 📚

## 简介 🌟

此文件夹包含用于构建本地向量知识库的工具。基于 CSV 文件创建 FAISS 向量索引，支持多知识库管理和双层检索策略。

## 文件说明 📋

- `build_knowledge_base.py` - 主要工具，用于构建向量知识库
- `query_knowledge_base.py` - 知识库查询工具（如果有）

## 前提条件 🔧

- Python 3.8+
- 必要的依赖包:
  ```bash
  pip install openai faiss-cpu tqdm numpy xinference
  ```
- 本地 embedding 模型 (如 bge-m3)
- 启动 Xinference 服务:
  ```bash 
  xinference-local
  ```

## 使用方法 🚀

### 1. 构建所有 CSV 文件的知识库

```bash
python build_knowledge_base.py --all
```

系统会自动读取配置的 CSV 目录中的所有文件，并为每个 CSV 文件创建一个以文件名命名的知识库。

### 2. 构建特定 CSV 文件的知识库

```bash
python build_knowledge_base.py --csv 文件路径.csv --name 知识库名称
```

例如:

```bash
python build_knowledge_base.py --csv F:\数据\电力系统.csv --name 电力知识库
```

## 知识库结构 🗂️

每个知识库将创建以下文件:

- `summary_index.faiss` - 总结内容的向量索引
- `tag_index.faiss` - 标签的向量索引 
- `metadata.pkl` - 知识库元数据
- `info.json` - 知识库基本信息

根目录下会生成 `kb_mapping.json` 文件，记录所有知识库的映射关系。

## 检索逻辑 🔍

知识库采用双层检索策略:

1. **第一层**: 将查询向量与文档总结向量计算相似度
2. **第二层**: 将关键查询与标签向量计算相似度
3. 设定阈值过滤，返回匹配结果

## 参数设置 ⚙️

主要参数在 `build_knowledge_base.py` 文件顶部:

```python
EMBEDDING_API_URL = "http://localhost:9997/v1"
KNOWLEDGE_BASE_ROOT = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\knowledge_base"
CSV_ROOT = r"F:\StrivingRendersMeCozy\DeepLearning\ERAG\Data\Knowledge_Docs\final"
```

根据需要修改这些路径。

## 注意事项 ⚠️

- CSV 文件必须包含 `内容`、`总结`、`标签` 这三个字段
- 标签字段可以是 JSON 列表格式或逗号分隔的字符串
- 确保 Xinference 服务已启动并正确加载 embedding 模型