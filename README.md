# 🚀 ERAG - 智能教学问答助手

<div align="center">

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Gradio](https://img.shields.io/badge/Gradio-3.32+-orange)
![许可证](https://img.shields.io/badge/许可证-MIT-brightgreen)
![状态](https://img.shields.io/badge/状态-开发中-yellow)

</div>

<div align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/2621/2621230.png" alt="ERAG Logo" width="150">
</div>

## 📖 项目简介

ERAG是一个为教育领域定制的基于知识库的智能问答系统。它能够根据不同学科知识库提供精准的问答服务，支持知识库的动态构建与切换，实现智能化教学辅助，为师生提供高效的学习和教学体验。

### ✨ 核心功能

- 🤖 **智能问答**：基于知识库提供准确的问题回答
- 📚 **多知识库支持**：包含计算教育学、基础数学理论、机器学习基础等多个学科
- 📤 **知识库构建**：支持PDF/TXT文件上传，动态扩充知识库
- 🔍 **智能检索**：精准定位相关信息，提供参考来源
- 💬 **自然对话**：流畅的对话体验，支持连续提问

## 🛠️ 技术架构

<div align="center">
  <img src="https://mermaid.ink/img/pako:eNptkc1qwzAQhF9F7CmF5AfYoYcUemkpPbQXX4xYbAVsyUhygyn57o1_ctrCHsTOzszuaqUDtF4hpPDm-o4fTr1FhZAXF0FjR3t0UeGbqdu6fWQXU1a4MgK3ZA2OJPu5CctETG2cVci9Pn1wcfp-OlcVzkoJMkGZFapm8JfCxw7l2fmRLD6UREXn-c23btPeOosxnufDBGunsUv-O_G1O5qBtr9OS2sAww8T07287_BloachYB_JCAVnj5F4QkiL3ls0CC22iSJdKJDyCXPFS4TMTc92pICWmxQhH8Loot1OQmaK0IaquCiKxdP9_RP9epbaRsSA_XVFm5sWF9zJTGYHvMZsZ63C7B8zpLew" alt="ERAG 架构" width="600">
</div>

- **前端**：基于Gradio构建的交互式Web界面
- **后端**：Python实现的知识检索与问答处理
- **UI设计**：自定义CSS样式，提供美观的用户界面
- **多线程处理**：支持文件异步处理，避免阻塞主界面

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 相关依赖包

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/ganchun1130/ERAG.git
cd ERAG

# 安装依赖
pip install -r requirements.txt

# 启动应用
python ui.py
```

### 🌐 在线体验

访问我们的在线演示: [ERAG演示](https://huggingface.co/spaces/example/erag-demo) (示例链接)

## 📱 使用指南

<div align="center">
  <img src="https://via.placeholder.com/800x400.png?text=ERAG+使用流程图" alt="使用流程" width="700">
</div>

1. **选择知识库**：从左侧面板选择要使用的知识库
2. **提问**：在输入框中输入问题，点击发送或按Enter键
3. **查看回答**：系统会基于选定知识库提供答案
4. **参考来源**：点击"显示参考来源"查看答案的知识依据
5. **上传文件**：点击"选择文件上传"扩充知识库
6. **新建对话**：点击"新建对话"开始全新会话

## 📂 项目结构

```
ERAG/
├── ui.py                  # 主界面实现
├── css.py                 # CSS样式定义
├── functions.py           # 核心功能函数
├── knowledge_base/        # 知识库文件夹
│   ├── 计算教育学/
│   ├── 基础数学理论/
│   └── 机器学习基础/
├── assets/                # 静态资源文件
├── tests/                 # 测试文件
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明文档
```

## 🧩 功能截图

<div align="center">
  <img src="https://via.placeholder.com/800x450.png?text=ERAG+智能问答界面" alt="智能问答界面" width="700">
  <p>ERAG智能问答界面</p>
</div>

## 🔧 配置说明

系统支持多种配置选项，包括：

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| 知识库路径 | 指定知识库存储位置 | `./knowledge_base` |
| 检索参数 | 控制检索精确度与范围 | k=3, threshold=0.7 |
| 界面主题 | 自定义UI界面风格 | `default` |
| 文件处理 | 文件处理相关参数 | chunk_size=500 |

## 📈 未来计划

- [ ] 支持更多文件格式（Markdown、Word、PPT等）
- [ ] 添加用户权限管理和多用户支持
- [ ] 优化检索算法，提高响应速度和准确性
- [ ] 增加数据分析功能和使用统计报告
- [ ] 开发移动端应用，实现跨平台访问

## 👨‍💻 开发者

- **开发者**: ganchun
- **联系方式**: ganchun1130@github.com
- **贡献指南**: [如何贡献](CONTRIBUTING.md)

## 🙏 致谢

感谢以下开源项目的支持：
- [Gradio](https://gradio.app/)
- [LangChain](https://langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📄 许可证

本项目采用MIT许可证

---

<div align="center">
  ⭐ 如果您觉得这个项目有用，请给它一个star！
</div>
