import gradio as gr
import time

history_style = """
/* 历史记录容器 */
.history-container {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 12px;
    height: 320px;
    overflow-y: auto;
}

/* 新增按钮样式 */
.new-chat-btn {
    width: 100%;
    margin: 10px 0;
    background: #3B82F6 !important;
    color: white !important;
}

.new-chat-btn:hover {
    background: #2563eb !important;
}

/* 调整历史容器高度 */
.history-container {
    height: 280px !important;  /* 原高度320px调整为280px */
}


/* 单个历史条目 */
.history-item {
    background: white;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: all 0.2s;
    border-left: 4px solid #3B82F6; /* 左侧装饰条 */
}

.history-item:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* 条目内容 */
.history-content {
    flex-grow: 1;
    margin-left: 12px;
}

.history-title {
    font-weight: 500;
    color: #1f2937;
    font-size: 14px;
}

.history-date {
    color: #6b7280;
    font-size: 12px;
    margin-top: 4px;
}

/* 操作按钮 */
.history-actions {
    opacity: 0;
    transition: opacity 0.2s;
}

.history-item:hover .history-actions {
    opacity: 0.6;
}

.history-actions img {
    width: 10px;
    height: 10px;
    cursor: pointer;
    margin-left: 5px;
}
"""

custom_css = """

/* 新增模态弹窗样式 */
.modal-mask {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: 9999;
    display: none;
    justify-content: center;
    align-items: center;
}

.modal-mask.show {
    display: flex !important;
}

.modal-content {
    background: white;
    padding: 30px 40px;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    text-align: center;
    animation: modalIn 0.3s ease;
}

@keyframes modalIn {
    from { transform: scale(0.9); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}

.loading-spinner {
    width: 50px;
    height: 50px;
    margin: 0 auto 15px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3B82F6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 智能体列表 */
.agent-list .agent-card {
    padding: 10px;
    margin: 8px 0;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* 文件上传区 */
.upload-section {
    position: relative;  /* 新增定位上下文 */
    border: 2px dashed #e5e7eb;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin: 15px 0;
    background: #f8f9fa;
    min-height: 160px;  /* 确保有足够空间 */
}

/* 上传状态提示 */
.upload-status {
    position: absolute;
    bottom: -40px;
    left: 0;
    right: 0;
    padding: 12px;
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border-top: 3px solid #3B82F6;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.3s ease;
    z-index: 100;
}

/* 紧凑历史记录 */
.compact-history .message {
    padding: 8px 12px !important;
    margin: 4px 0 !important;
    min-height: 32px !important;
    font-size: 13px !important;
}

/* 知识库列表样式 */
.knowledge-list {
    background: white;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.knowledge-item {
    padding: 8px 12px;
    margin: 6px 0;
    background: #f8f9fa;
    border-radius: 6px;
    display: flex;
    align-items: center;
}

.knowledge-item::before {
    content: "📚";
    margin-right: 8px;
}



/* 知识库选中样式 */
.knowledge-item.selected {
    background: #e0f2fe !important;
    border: 2px solid #38bdf8 !important;
}

/* 知识库提示 */
.knowledge-hint {
    margin: 10px 0;
    padding: 8px 12px;
    background: #f0f9ff;
    border-radius: 6px;
    border: 1px solid #7dd3fc;
    color: #0369a1;
    font-size: 13px;
}


.upload-status.show {
    opacity: 1;
    transform: translateY(0);
}

.loading-dots {
    display: inline-flex;
    gap: 6px;
    margin-left: 12px;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    background: #3B82F6;
    border-radius: 50%;
    animation: bounce 1.4s infinite;
}

@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-4px); }
}

"""

def create_history_item(title, date):
    return f"""
    <div class="history-item">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 12V5H4V19H13" stroke="#4B5563" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M18 14V17M18 17V20M18 17H15M18 17H21" stroke="#3B82F6" stroke-width="1.5"/>
        </svg>
        <div class="history-content">
            <div class="history-title">{title}</div>
            <div class="history-date">{date}</div>
        </div>
        <div class="history-actions">
            <img  title="删除">
            <img  title="导出">
        </div>
    </div>
    """

def create_interface():
    with gr.Blocks(title="智能问答", css=custom_css) as demo:
        # 添加模态弹窗
        modal = gr.HTML(visible=False)
        
        # 修改知识块相关样式，默认显示知识块
        knowledge_sources_css = """
        /* 知识块展示相关样式 - 修改为默认显示 */
        .knowledge-sources {
            margin-top: 10px;
            padding: 10px;
            border-radius: 8px;
            background: #f8fafc;
            border-left: 3px solid #3B82F6;
            font-size: 14px;
            /* 默认显示，不设置display:none */
        }

        .source-item {
            margin: 8px 0;
            padding: 10px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        }

        .source-title {
            font-weight: 500;
            color: #3B82F6;
            margin-bottom: 5px;
        }

        .source-text {
            color: #4b5563;
            font-size: 13px;
            line-height: 1.5;
        }

        .toggle-sources {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            color: #0369a1;
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            margin-top: 5px;
        }

        .toggle-sources:hover {
            background: #e0f2fe;
        }

        .message-buttons {
            display: flex;
            justify-content: flex-end;
            margin-top: 8px;
        }
        """
        
        # 将知识块CSS添加到主CSS中
        demo.css = custom_css + knowledge_sources_css
        
        # 修改JavaScript，使用向上箭头隐藏、向下箭头显示
        gr.HTML("""
        <script>
        function toggleSources(messageId) {
            const sourcesEl = document.getElementById('sources-' + messageId);
            const buttonEl = document.getElementById('toggle-' + messageId);
            
            if (sourcesEl.style.display === 'none') {
                sourcesEl.style.display = 'block';
                buttonEl.innerHTML = '↑ 隐藏参考来源';
            } else {
                sourcesEl.style.display = 'none';
                buttonEl.innerHTML = '↓ 显示参考来源';
            }
        }
        </script>
        """)
        
        with gr.Row():
            # 左侧边栏
            with gr.Column(scale=1, min_width=300):
                # 智能体列表
                gr.Markdown("### 🤖 智能体列表")
                with gr.Column(elem_classes="agent-list"):
                    gr.HTML("""
                        <div class="agent-card">🔄 动态检索路由 智能体</div>
                        <div class="agent-card">🔍 检索 智能体</div>
                        <div class="agent-card">✨ 生成 智能体</div>
                    """)
                    
                # 知识库列表
                gr.Markdown("### 🗃️ 知识库列表")
                knowledge_radio = gr.Radio(
                    choices=["计算教育学", "基础数学理论", "机器学习基础"],
                    label="可用知识库",
                    elem_classes="knowledge-list",
                    interactive=True,
                    show_label=False
                )
                    
                # 文件上传
                gr.Markdown("### 📤 知识库构建")
                with gr.Column(elem_classes="upload-section"):
                    upload_btn = gr.UploadButton(
                        "点击上传知识文件",
                        file_types=[".txt", ".pdf"],
                        file_count="multiple"
                    )
                    gr.Markdown("支持TXT/PDF格式", elem_classes="format-hint")
                    # 集成状态提示
                    status_display = gr.HTML("""
                        <div class="upload-status">
                            <span>正在解析知识文件...</span>
                            <div class="loading-dots">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    """, visible=False)
                
                # 新建对话按钮
                new_chat_btn = gr.Button(
                    "➕ 新建对话",
                    variant="primary",
                    elem_classes="new-chat-btn"
                )

            # 主聊天区
            with gr.Column(scale=3):
                # 重新添加预设对话，并默认显示知识来源
                initial_chat = [
                    ["什么是计算教育学？",
                     """计算教育学是信息时代教育科学领域的一次范式革新，其核心在于运用大数据、人工智能等技术手段对教育现象进行量化分析与建模，从而揭示教学规律并优化教育实践。
                     
具体来说，计算教育学是教育学、计算机科学、认知科学等多学科深度交叉的产物。狭义上，它是以大数据为基础，以复杂性算法为核心，通过量化建模揭示教育规律的研究领域；广义而言，其研究对象涵盖教育过程中的网络化组织机制、教育质量量化测度及改进方法。相较于传统教育学依赖经验总结的特点，计算教育学强调通过数据驱动的方式实现教育研究的科学化转型，从定性经验转向定量精准分析。

<div class="message-buttons">
    <button id="toggle-1" class="toggle-sources" onclick="toggleSources(1)">
        ↑ 隐藏参考来源
    </button>
</div>

<div id="sources-1" class="knowledge-sources">
    <div class="source-item">
        <div class="source-title">《计算教育学导论》</div>
        <div class="source-text">计算教育学是信息时代教育科学领域的一次范式革新，其核心在于运用大数据、人工智能等技术手段对教育现象进行量化分析与建模。计算教育学是教育学、计算机科学、认知科学等多学科深度交叉的产物，以大数据为基础，以复杂性算法为核心，通过量化建模揭示教育规律的研究领域。</div>
    </div>
    <div class="source-item">
        <div class="source-title">《计算教育学导论》</div>
        <div class="source-text">相较于传统教育学依赖经验总结的特点，计算教育学强调通过数据驱动的方式实现教育研究的科学化转型，从定性经验转向定量精准分析。其研究对象涵盖教育过程中的网络化组织机制、教育质量量化测度及改进方法。</div>
    </div>
</div>"""
                    ],
                    ["它有哪些应用？",
                     """计算教育学的应用覆盖教育全场景，通过数据与算法的深度融合，正在重塑教育生态。以下是其在六大核心领域的典型应用场景：

1. **学习者画像**：通过多维数据分析构建学习者认知、情感和行为特征模型，实现精准化教学干预。

2. **学习资源智能推荐**：基于知识图谱和用户学习行为，自适应推荐个性化学习资源。

3. **智能评测**：利用自然语言处理技术实现作文自动评分、编程作业自动评测等。

4. **学习过程分析**：挖掘学习行为数据，揭示学习规律，预测学习结果，及时干预。

5. **教学质量评估**：构建多元教学评价模型，实现对教学过程的自动化、精准化评价。

6. **教育政策决策支持**：基于大规模教育数据分析，为教育管理者提供科学决策依据。

<div class="message-buttons">
    <button id="toggle-2" class="toggle-sources" onclick="toggleSources(2)">
        ↑ 隐藏参考来源
    </button>
</div>

<div id="sources-2" class="knowledge-sources">
    <div class="source-item">
        <div class="source-title">《计算教育学导论》</div>
        <div class="source-text">计算教育学应用覆盖了从学习者建模、资源推荐、智能评测到学习分析、教学评估和决策支持等多个领域。其核心价值在于通过数据驱动和算法支持，实现教育全场景的智能化、个性化和精准化，从而提升教育效能和学习体验。</div>
    </div>
    <div class="source-item">
        <div class="source-title">《计算教育学导论》</div>
        <div class="source-text">计算教育学在实践中已形成六大核心应用场景：1)学习者认知、情感和行为特征建模；2)基于知识图谱的个性化资源推荐；3)自动化学习评测；4)学习过程挖掘与预测；5)多维度教学质量评估；6)数据驱动的教育决策支持。</div>
    </div>
</div>"""
                ]
]
                chatbot = gr.Chatbot(value=initial_chat, height=600)
                with gr.Group():
                    input_box = gr.Textbox(placeholder="输入问题...")
                    hint_area = gr.HTML(
                        visible=False,
                        elem_classes="knowledge-hint"
                    )
                send_btn = gr.Button("发送", variant="primary")
            
        # 其余代码保持不变
        def show_loading(files):
            return gr.HTML("""
                <div class="modal-mask show">
                    <div class="modal-content">
                        <div class="loading-spinner"></div>
                        <h3 style="margin:0;color:#1f2937">正在解析文件中...</h3>
                        <p style="color:#6b7280;margin-top:8px">请稍候，这可能需要一些时间</p>
                    </div>
                </div>
            """, visible=True)

        def hide_loading():
            time.sleep(5)
            return gr.HTML(visible=False)
            
        # 新建对话的处理函数
        def create_new_chat():
            return []  # 返回空对话记录
            
        # 聊天函数，添加带知识块的回复
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            
            # 根据问题返回不同的回复和知识来源
            if "机器学习" in message or "人工智能" in message:
                msg_id = len(chat_history) + 1
                bot_message = f"""机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并做出预测的算法和模型。

机器学习系统通过对大量数据的训练，能够识别模式并在没有明确编程的情况下做出决策。常见的机器学习类型包括监督学习、非监督学习和强化学习，每种类型适用于不同的问题场景。

<div class="message-buttons">
    <button id="toggle-{msg_id}" class="toggle-sources" onclick="toggleSources({msg_id})">
        ↑ 隐藏参考来源
    </button>
</div>

<div id="sources-{msg_id}" class="knowledge-sources">
    <div class="source-item">
        <div class="source-title">《机器学习基础》第一章</div>
        <div class="source-text">机器学习是人工智能的一个分支，其核心是让计算机从数据中自动学习。机器学习系统通过识别数据中的模式，可以在没有明确编程指令的情况下做出推断和决策。</div>
    </div>
    <div class="source-item">
        <div class="source-title">《人工智能导论》2022版</div>
        <div class="source-text">机器学习可分为监督学习、非监督学习和强化学习。监督学习使用带标签的数据训练模型，非监督学习从无标签数据中发现结构，强化学习则通过奖惩机制引导代理做出最优决策。</div>
    </div>
</div>"""
            else:
                # 默认回复
                msg_id = len(chat_history) + 1
                bot_message = f"""这是一个示例回复，真实系统中会根据知识库内容生成回答。

<div class="message-buttons">
    <button id="toggle-{msg_id}" class="toggle-sources" onclick="toggleSources({msg_id})">
        ↑ 隐藏参考来源
    </button>
</div>

<div id="sources-{msg_id}" class="knowledge-sources">
    <div class="source-item">
        <div class="source-title">示例知识来源1</div>
        <div class="source-text">这是一个示例知识块，展示了回答中参考的信息来源。在实际应用中，这里会显示检索到的真实文档片段。</div>
    </div>
    <div class="source-item">
        <div class="source-title">示例知识来源2</div>
        <div class="source-text">第二个知识来源的示例内容，用于展示多个参考来源的情况。RAG系统通常会检索多个相关文档片段作为生成回答的依据。</div>
    </div>
</div>"""
            
            chat_history.append([message, bot_message])
            return "", chat_history

        # 事件绑定
        upload_btn.upload(
            show_loading,
            inputs=[upload_btn],
            outputs=[modal]
        ).then(
            hide_loading,
            outputs=[modal]
        )
        
        # 知识库选择交互
        def update_hint(knowledge):
            if knowledge:
                return gr.HTML(
                    f"🧠 教学助手正在基于 <strong>{knowledge}</strong> 知识库和您聊天",
                    visible=True
                )
            return gr.HTML(visible=False)
      
        knowledge_radio.change(
            update_hint,
            inputs=[knowledge_radio],
            outputs=[hint_area]
        )
        
        # 新建对话按钮事件绑定
        new_chat_btn.click(
            create_new_chat,
            outputs=[chatbot]
        )
        
        # 添加发送按钮功能
        send_btn.click(
            respond,
            inputs=[input_box, chatbot],
            outputs=[input_box, chatbot]
        )
        
        # 添加回车发送功能
        input_box.submit(
            respond,
            inputs=[input_box, chatbot],
            outputs=[input_box, chatbot]
        )

    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()