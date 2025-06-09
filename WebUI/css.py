# coding:utf-8
# @File  : css.py
# @Author: ganchun
# @Date  :  2025/04/22
# @Description: 存放前端UI的所有CSS样式

# 历史记录样式
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

# 主要样式
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

# 知识块样式
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

# 合并所有CSS
complete_css = custom_css + knowledge_sources_css