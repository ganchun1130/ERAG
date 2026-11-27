"""Compact settings editor; values are published only by Save, never by typing."""

import gradio as gr
from pydantic import ValidationError

from .configuration import WebProfile

CHAT_TYPES = [
    ("沿用启动配置", "inherit"),
    ("OpenAI 兼容 API", "openai"),
    ("vLLM", "vllm"),
    ("Ollama", "ollama"),
    ("Xinference", "xinference"),
    ("火山方舟", "ark"),
]


class SettingsPanel:
    def __init__(self):
        self.fields, self.groups = {}, {}
        with gr.Accordion("模型与构建", open=False, elem_classes=["sidebar-section", "settings-panel"]):
            gr.Markdown("为当前工作空间设置，保存后生效。", elem_classes="settings-hint")
            with gr.Tabs():
                with gr.Tab("聊天"):
                    self.connection("chat", CHAT_TYPES, "聊天服务", "https://api.example.com/v1")
                    gr.Markdown(
                        "API 地址填服务根地址；兼容接口通常以 /v1 结尾，Ollama 不加 /v1。",
                        elem_classes="settings-hint",
                    )
                    self.test_chat = gr.Button("测试聊天连接", size="sm")
                with gr.Tab("知识库"):
                    self.fields["method"] = gr.Dropdown(
                        [
                            ("沿用项目方式", "inherit"),
                            ("自动解析与分块", "documents"),
                            ("导入已分块 JSONL", "jsonl"),
                            ("外部检索服务", "external"),
                        ],
                        value="inherit",
                        label="构建方式",
                    )
                    self.method_note = gr.Markdown(
                        "沿用项目预设，也支持管理员安装的自定义扩展。", elem_classes="settings-hint"
                    )
                    with gr.Column(visible=False, elem_classes="settings-group") as external:
                        self.fields["external_url"] = gr.Textbox(
                            label="检索 API 地址", placeholder="https://knowledge.example.com/retrieval"
                        )
                        self.fields["external_name"] = gr.Textbox(label="显示名称", value="我的外部知识库")
                        self.fields["external_id"] = gr.Textbox(label="远程知识库 ID")
                        self.fields["external_key"] = gr.Textbox(
                            label="知识库 API 密钥", type="password", placeholder="选填"
                        )
                        self.fields["external_clear"] = gr.Checkbox(label="清除已保存密钥", value=False)
                    with gr.Column(elem_classes="settings-group") as embedding:
                        self.connection(
                            "embedding",
                            [
                                ("沿用启动配置", "inherit"),
                                ("OpenAI 兼容 API", "openai"),
                                ("Ollama", "ollama"),
                                ("方舟多模态嵌入", "ark_multimodal"),
                            ],
                            "嵌入服务",
                            "http://127.0.0.1:8001/v1",
                        )
                    self.test_knowledge = gr.Button("测试知识库连接", size="sm")
                    with gr.Accordion("分块与向量选项", open=False, visible=False) as advanced:
                        self.fields["chunk_size"] = gr.Number(
                            value=900, precision=0, label="分块长度（字符）", minimum=100, maximum=20000
                        )
                        self.fields["chunk_overlap"] = gr.Number(
                            value=120, precision=0, label="重叠长度（字符）", minimum=0
                        )
                        self.fields["enrich"] = gr.Checkbox(value=False, label="用聊天模型生成摘要和标签")
                        self.fields["embedding.revision"] = gr.Textbox(
                            label="嵌入模型版本标记", placeholder="可选；更换权重后更新此标记"
                        )
                        self.fields["embedding.query_template"] = gr.Textbox(
                            label="查询模板", placeholder="留空自动选择；自定义时须包含 {text}"
                        )

                    def method_changed(method, protocol):
                        notes = {
                            "inherit": "沿用项目预设，也支持管理员安装的自定义扩展。",
                            "documents": "上传 PDF、TXT、Markdown 或 CSV，自动解析、分块并建库。",
                            "jsonl": "上传自己的知识块，每行包含 id、content、source，可附 metadata。",
                            "external": "资料在外部服务中维护；ERAG 连接其检索接口，继续提供多轮问答与引用。",
                        }
                        return (
                            gr.update(visible=method == "external"),
                            gr.update(visible=method != "external"),
                            gr.update(
                                visible=method != "external"
                                and (method != "inherit" or protocol != "inherit")
                            ),
                            notes[method],
                            gr.update(visible=method == "documents"),
                            gr.update(visible=method == "documents"),
                            gr.update(visible=method in {"documents", "jsonl"}),
                            gr.update(visible=protocol != "inherit"),
                            gr.update(visible=protocol != "inherit"),
                        )

                    for component in (self.fields["method"], self.fields["embedding.protocol"]):
                        component.change(
                            method_changed,
                            [self.fields["method"], self.fields["embedding.protocol"]],
                            [
                                external,
                                embedding,
                                advanced,
                                self.method_note,
                                self.fields["chunk_size"],
                                self.fields["chunk_overlap"],
                                self.fields["enrich"],
                                self.fields["embedding.revision"],
                                self.fields["embedding.query_template"],
                            ],
                            api_visibility="private",
                        )
                with gr.Tab("重排"):
                    self.connection(
                        "rerank",
                        [
                            ("沿用启动配置", "inherit"),
                            ("自定义重排 API", "openai"),
                            ("不使用重排", "disabled"),
                        ],
                        "重排服务",
                        "http://127.0.0.1:8002",
                    )
                    gr.Markdown(
                        "可选。用于调整检索结果的顺序，接口需支持 /rerank。", elem_classes="settings-hint"
                    )
                    self.test_rerank = gr.Button("测试重排连接", size="sm")
            self.status = gr.Markdown("", elem_classes="settings-status")
            self.save = gr.Button("保存并应用", variant="primary", size="sm")
            self.reset = gr.Button("恢复启动配置", size="sm")
            gr.Markdown(
                "密钥仅保存在服务端，不回填页面。更换 API 地址后，请重新填写密钥。",
                elem_classes="settings-hint",
            )

    def connection(self, role, choices, label, placeholder):
        protocol = self.fields[f"{role}.protocol"] = gr.Dropdown(choices, value="inherit", label=label)
        with gr.Column(visible=False, elem_classes="settings-group") as group:
            self.fields[f"{role}.url"] = gr.Textbox(label="API 地址", placeholder=placeholder, max_lines=1)
            self.fields[f"{role}.model"] = gr.Textbox(
                label="模型名称", placeholder="填写服务中的模型名或部署别名", max_lines=1
            )
            self.fields[f"{role}.key"] = gr.Textbox(
                label="API 密钥", type="password", placeholder="选填；本地服务通常可留空"
            )
            self.fields[f"{role}.clear"] = gr.Checkbox(value=False, label="清除已保存密钥")
        self.groups[role] = group
        protocol.change(
            lambda value: gr.update(visible=value not in {"inherit", "disabled"}),
            [protocol],
            [group],
            api_visibility="private",
        )

    @property
    def inputs(self):
        return list(self.fields.values())

    def draft(self, runtime, owner, values):
        data, clear = {}, []
        for name, value in zip(self.fields, values, strict=True):
            if name.endswith(".clear") or name == "external_clear":
                if value:
                    clear.append("external" if name == "external_clear" else name.split(".")[0])
            elif "." in name:
                role, field = name.split(".")
                data.setdefault(role, {})[field] = value
            else:
                data[name] = value
        try:
            return runtime.profiles.prepare(owner, data, clear)
        except ValidationError as exc:
            # Avoid Pydantic's input dump, which can contain credentials.
            message = exc.errors(include_input=False, include_url=False)[0]["msg"]
            raise gr.Error("请检查配置：" + message.removeprefix("Value error, ")) from None

    def values(self, profile: WebProfile):
        data = profile.model_dump()
        updates = []
        for name in self.fields:
            if name.endswith(".clear") or name == "external_clear":
                updates.append(gr.update(value=False))
                continue
            if "." in name:
                role, field = name.split(".")
                value = data[role][field]
            else:
                value = data[name]
            if name.endswith(".key") or name == "external_key":
                updates.append(
                    gr.update(
                        value="", placeholder="已保存；留空保留" if value.get_secret_value() else "选填"
                    )
                )
            else:
                updates.append(gr.update(value=value))
        return updates
