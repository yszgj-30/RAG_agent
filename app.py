"""
企业知识库 RAG 问答 Agent —— 主程序入口
技术栈：通义千问 API + LangChain + Chroma + Streamlit
启动方式：streamlit run app.py
"""
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone

import streamlit as st
from langchain_core.messages import HumanMessage

from agent.core import RAGAgent
from document.processor import DocumentProcessor
from memory.manager import ChatMemoryManager
from models.tongyi_llm import TongyiLLM
from operations.store import OperationsStore
from ui.layout import render_chat_area, render_sidebar, show_welcome
from ui.workspaces import (
    render_evaluation_workspace,
    render_knowledge_workspace,
    render_retrieval_workspace,
)
from vectordb.store import VectorStore


# ── 页面配置 ────────────────────────────────
st.set_page_config(
    page_title="企业知识库问答",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="auto",
)


# ── 文案配置：简约商务中文 ──────────────────
UI_COPY = {
    "sidebar": {
        "brand_title": "知识库",
        "brand_subtitle": "企业智能问答工作台",
        "api_title": "模型连接",
        "api_placeholder": "请输入 API Key",
        "api_help": "用于连接大模型服务，仅当前会话有效",
        "api_button": "连接模型",
        "api_hint": "连接后即可导入文档并开始问答",
        "api_ready": "模型已连接",
        "api_idle": "尚未连接",
        "upload_title": "文档导入",
        "upload_help": "支持 TXT、PDF、MD，可多选",
        "upload_hint": "建议导入制度、流程、产品资料等业务文档",
        "upload_selected": "已选 {count} 个文件",
        "upload_build": "导入知识库",
        "upload_clear": "清空知识库",
        "store_title": "知识库概况",
        "store_label": "知识片段",
        "store_empty": "暂无文档",
        "store_collection": "集合",
        "chat_title": "会话管理",
        "chat_button": "清空当前会话",
        "chat_hint": "仅清空会话记录，不影响知识库",
        "footer_line1": "企业知识库",
        "footer_line2": "检索增强生成 · 来源可追溯",
    },
    "main": {
        "eyebrow": "ENTERPRISE KNOWLEDGE",
        "title": "企业知识助手",
        "subtitle": "连接散落的业务资料，让每次回答都有依据、可继续追问。",
        "capabilities": ["问题分类", "知识库检索", "工具调用", "多轮记忆"],
        "summary_title": "系统状态",
        "summary_items": [
            ("模型服务", "通义千问"),
            ("知识引擎", "Chroma 向量检索"),
            ("会话能力", "多轮上下文问答"),
        ],
    },
    "status": {
        "main_idle": "请先连接模型，再开始构建知识库",
        "main_ready_no_docs": "模型已连接，请导入业务文档以开始问答",
        "main_ready": "知识库就绪，请输入业务问题",
        "metrics_ready": "就绪",
        "metrics_pending": "待连接",
    },
    "chat": {
        "input_placeholder": "输入问题，支持普通对话与知识库检索",
        "spinner": "检索中…",
        "expander_thinking": "处理过程",
        "expander_sources": "参考来源",
        "no_sources": "本次回答未引用知识库内容",
        "error_prefix": "问答异常",
    },
    "welcome": {
        "eyebrow": "快速开始",
        "title": "从一份业务文档开始",
        "subtitle": "三步完成知识导入与可追溯问答，无需预先配置复杂流程。",
        "steps_title": "使用流程",
        "steps": [
            ("连接模型", "配置 API Key，建立大模型服务连接"),
            ("导入知识", "上传业务文档，自动解析并写入知识库"),
            ("开始检索", "输入问题，获取基于知识库的准确回答"),
        ],
        "abilities_title": "适用场景",
        "abilities": ["制度手册", "产品资料", "流程规范", "培训文档"],
        "tip": "回答会区分普通对话与知识库问题；引用知识时可展开查看来源。",
    },
    "toast": {
        "init_success": "模型连接成功",
        "init_error": "模型连接失败",
        "upload_empty": "请先选择文档",
        "upload_success": "已导入 {file_count} 个文件，共 {chunk_count} 个片段",
        "upload_info": "文件：{files}",
        "upload_no_text": "未提取到有效文本，请检查文件内容",
        "upload_file_error": "「{file_name}」解析失败：{error}",
        "clear_kb_success": "知识库已清空",
        "clear_chat_success": "会话已清空",
        "query_error": "处理异常：{error}",
    },
    "metrics": {
        "kb_count": "知识片段",
        "message_count": "会话消息",
        "agent_status": "系统状态",
    },
}


# ── 全局样式 ────────────────────────────────
st.markdown("""
<style>
    :root {
        --app-bg: #F8FAFC;
        --surface: #FFFFFF;
        --surface-muted: #F1F5F9;
        --text: #0F172A;
        --text-soft: #475569;
        --muted: #64748B;
        --border: #E2E8F0;
        --primary: #2563EB;
        --primary-hover: #1D4ED8;
        --primary-soft: #EFF6FF;
        --success: #047857;
        --success-soft: #ECFDF5;
        --warning: #B45309;
        --warning-soft: #FFFBEB;
        --error: #B91C1C;
        --error-soft: #FEF2F2;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
        --shadow-md: 0 8px 24px rgba(15, 23, 42, 0.07);
        --transition: 200ms ease;
    }

    html, body, .stApp {
        font-family: Inter, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        color: var(--text);
        background: var(--app-bg);
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--app-bg);
    }

    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.94);
        border-bottom: 1px solid var(--border);
    }

    [data-testid="stToolbar"] { background: transparent; }

    [data-testid="stMainBlockContainer"] {
        max-width: 1080px;
        padding: 2.25rem 2rem 7rem;
    }

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] {
        color: var(--text-soft);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #F1F5F9;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div { background: transparent; }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.4rem 0.65rem 1.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
    }

    .brand-block {
        padding: 1rem 0.2rem 1.1rem;
        margin-bottom: 0.15rem;
    }

    .brand-inner {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-mark {
        width: 40px;
        height: 40px;
        border-radius: 11px;
        display: grid;
        place-items: center;
        color: #FFFFFF;
        background: var(--primary);
        font-size: 1rem;
        font-weight: 750;
        box-shadow: 0 5px 14px rgba(37, 99, 235, 0.2);
    }

    .brand-title,
    .brand-subtitle,
    .section-title,
    .soft-note,
    .count-value,
    .count-label {
        margin: 0;
    }

    .brand-title {
        color: var(--text);
        font-size: 0.98rem;
        font-weight: 720;
    }

    .brand-subtitle,
    .soft-note {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.55;
    }

    .brand-subtitle { margin-top: 0.12rem; }

    .section-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.3rem;
    }

    .section-title {
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 680;
    }

    .section-tag {
        color: var(--primary);
        background: var(--primary-soft);
        border: 1px solid #DBEAFE;
        border-radius: 999px;
        padding: 0.18rem 0.48rem;
        font-size: 0.64rem;
        font-weight: 750;
        letter-spacing: 0.06em;
    }

    .status-chip {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 0.55rem;
        padding: 0.5rem 0.65rem;
        border-radius: var(--radius-sm);
        font-size: 0.76rem;
        font-weight: 650;
    }

    .status-chip.success {
        color: var(--success);
        background: var(--success-soft);
        border: 1px solid #A7F3D0;
    }

    .status-chip.warning {
        color: var(--warning);
        background: var(--warning-soft);
        border: 1px solid #FDE68A;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        flex: 0 0 7px;
        border-radius: 50%;
        background: currentColor;
    }

    .knowledge-stat {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-top: 0.75rem;
        padding-top: 0.7rem;
        border-top: 1px solid var(--border);
    }

    .count-value {
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 750;
    }

    .count-label {
        color: var(--muted);
        font-size: 0.73rem;
    }

    .kb-state {
        color: var(--text-soft);
        background: var(--surface-muted);
        border-radius: 999px;
        padding: 0.25rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 650;
    }

    .sidebar-footer {
        margin-top: 1.15rem;
        padding-top: 0.85rem;
        border-top: 1px solid var(--border);
        text-align: center;
    }

    .sidebar-footer p {
        margin: 0.1rem 0;
        color: #94A3B8;
        font-size: 0.7rem;
    }

    /* Workspace header */
    .workspace-hero {
        padding: 0.35rem 0 1.1rem;
    }

    .hero-eyebrow,
    .welcome-eyebrow {
        margin: 0 0 0.55rem;
        color: var(--primary);
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .stMarkdown .hero-eyebrow,
    .stMarkdown .welcome-eyebrow {
        color: var(--primary);
    }

    .workspace-title {
        margin: 0;
        color: var(--text);
        font-size: clamp(2rem, 4vw, 2.85rem);
        line-height: 1.12;
        font-weight: 760;
        letter-spacing: -0.045em;
    }

    .workspace-subtitle {
        max-width: 680px;
        margin: 0.7rem 0 0;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.7;
    }

    .capability-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 1rem;
    }

    .capability-chip {
        display: inline-flex;
        align-items: center;
        color: var(--text-soft);
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.34rem 0.68rem;
        font-size: 0.75rem;
        font-weight: 580;
    }

    .workspace-status {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 0.2rem 0 1.1rem;
        padding: 0.78rem 0.9rem;
        border: 1px solid;
        border-radius: var(--radius-md);
        font-size: 0.82rem;
        font-weight: 600;
    }

    .workspace-status.idle {
        color: var(--warning);
        background: var(--warning-soft);
        border-color: #FDE68A;
    }

    .workspace-status.ready {
        color: var(--success);
        background: var(--success-soft);
        border-color: #A7F3D0;
    }

    .workspace-status .status-dot {
        box-shadow: 0 0 0 4px currentColor;
        opacity: 0.8;
        transform: scale(0.45);
    }

    /* Welcome */
    .welcome-panel {
        margin: 0.3rem 0 1.3rem;
        padding: 1.5rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-sm);
    }

    .welcome-title {
        margin: 0;
        color: var(--text);
        font-size: 1.4rem;
        font-weight: 720;
        letter-spacing: -0.025em;
    }

    .welcome-subtitle {
        margin: 0.5rem 0 0;
        max-width: 680px;
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.65;
    }

    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1.25rem;
    }

    .workflow-card {
        min-height: 128px;
        padding: 1rem;
        background: var(--app-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
    }

    .workflow-number {
        display: grid;
        place-items: center;
        width: 28px;
        height: 28px;
        margin-bottom: 0.75rem;
        color: var(--primary);
        background: var(--primary-soft);
        border: 1px solid #DBEAFE;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 750;
    }

    .workflow-card h3 {
        margin: 0;
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 680;
        letter-spacing: 0;
    }

    .workflow-card p {
        margin: 0.35rem 0 0;
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.55;
    }

    .welcome-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 1.1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }

    .welcome-tip {
        margin: 0;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.55;
    }

    .scene-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.35rem;
    }

    .scene-chip {
        color: var(--text-soft);
        background: var(--surface-muted);
        border-radius: 999px;
        padding: 0.28rem 0.55rem;
        font-size: 0.7rem;
        white-space: nowrap;
    }

    /* Inputs and buttons */
    .stButton > button {
        min-height: 2.45rem;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        color: var(--text);
        background: var(--surface);
        box-shadow: var(--shadow-sm);
        font-size: 0.82rem;
        font-weight: 650;
        transition: border-color var(--transition), color var(--transition),
                    background var(--transition), box-shadow var(--transition);
    }

    .stButton > button:hover:not(:disabled) {
        color: var(--primary);
        background: var(--primary-soft);
        border-color: #93C5FD;
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.1);
    }

    .stButton > button[kind="primary"] {
        color: #FFFFFF;
        background: var(--primary);
        border-color: var(--primary);
    }

    .stButton > button[kind="primary"]:hover:not(:disabled) {
        color: #FFFFFF;
        background: var(--primary-hover);
        border-color: var(--primary-hover);
    }

    .stButton > button:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    button:focus-visible {
        outline: 3px solid rgba(37, 99, 235, 0.22) !important;
        outline-offset: 2px;
    }

    .stButton > button:disabled {
        color: #94A3B8;
        background: #F8FAFC;
        border-color: var(--border);
        box-shadow: none;
    }

    .stTextInput input {
        color: var(--text) !important;
        background: var(--surface) !important;
        border-color: var(--border) !important;
    }

    .stTextInput input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
        opacity: 1;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 108px;
        background: var(--app-bg);
        border: 1px dashed #CBD5E1;
        border-radius: var(--radius-sm);
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background: var(--primary-soft);
        border-color: #93C5FD;
    }

    [data-testid="stBottomBlockContainer"] {
        background: var(--app-bg);
        border-top: 1px solid var(--border);
    }

    [data-testid="stChatInput"] {
        background: var(--surface) !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        box-shadow: var(--shadow-md);
    }

    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        background: var(--surface) !important;
        caret-color: var(--primary);
    }

    [data-testid="stChatInput"] button {
        color: #FFFFFF !important;
        background: var(--primary) !important;
        border-radius: 9px !important;
    }

    /* Conversation and citations */
    div[data-testid="stChatMessage"] {
        margin-bottom: 0.75rem;
        padding: 0.7rem 0.85rem;
        color: var(--text);
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-sm);
    }

    div[data-testid="stChatMessage"] p { color: var(--text-soft); }

    [data-testid="stExpander"] {
        overflow: hidden;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: none;
    }

    [data-testid="stExpander"] summary {
        color: var(--text-soft);
        font-size: 0.8rem;
        font-weight: 650;
    }

    .source-list {
        display: grid;
        gap: 0.55rem;
    }

    .source-card {
        display: grid;
        grid-template-columns: 30px minmax(0, 1fr) auto;
        align-items: center;
        gap: 0.75rem;
        padding: 0.72rem;
        background: var(--app-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
    }

    .source-index {
        display: grid;
        place-items: center;
        width: 30px;
        height: 30px;
        color: var(--primary);
        background: var(--primary-soft);
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 750;
    }

    .source-name {
        overflow-wrap: anywhere;
        color: var(--text);
        font-size: 0.8rem;
        font-weight: 620;
    }

    .source-score {
        color: var(--success);
        background: var(--success-soft);
        border-radius: 999px;
        padding: 0.28rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 680;
        white-space: nowrap;
    }

    /* Workspace navigation and operations views */
    [data-baseweb="tab-list"] {
        gap: 0.35rem;
        margin-bottom: 1rem;
        padding: 0.3rem;
        background: #EEF2F7;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
    }

    [data-baseweb="tab"] {
        min-height: 44px;
        padding: 0 1rem;
        color: var(--muted);
        border-radius: var(--radius-sm);
        font-size: 0.82rem;
        font-weight: 650;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: var(--text);
        background: var(--surface);
        box-shadow: var(--shadow-sm);
    }

    [data-baseweb="tab-highlight"] { display: none; }

    .ops-heading {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.45rem 0 1.1rem;
    }

    .ops-heading h2 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 740;
    }

    .ops-heading p {
        margin: 0.42rem 0 0;
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .ops-heading .ops-eyebrow {
        margin: 0 0 0.4rem;
        color: var(--primary);
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.1em;
    }

    .ops-empty {
        margin: 0.7rem 0 1rem;
        padding: 1.4rem;
        text-align: center;
        background: var(--surface);
        border: 1px dashed #CBD5E1;
        border-radius: var(--radius-md);
    }

    .ops-empty-title {
        margin: 0;
        color: var(--text);
        font-size: 0.92rem;
        font-weight: 680;
    }

    .ops-empty-description {
        margin: 0.35rem 0 0;
        color: var(--muted);
        font-size: 0.78rem;
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
    }

    .retrieval-card {
        margin-bottom: 0.65rem;
        padding: 0.9rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
    }

    .retrieval-card-head {
        display: grid;
        grid-template-columns: 30px minmax(0, 1fr) auto;
        align-items: center;
        gap: 0.7rem;
    }

    .retrieval-rank,
    .trace-index {
        display: grid;
        place-items: center;
        width: 30px;
        height: 30px;
        color: var(--primary);
        background: var(--primary-soft);
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 750;
        font-variant-numeric: tabular-nums;
    }

    .retrieval-score {
        color: var(--success);
        background: var(--success-soft);
        border-radius: 999px;
        padding: 0.25rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }

    .retrieval-card p {
        margin: 0.7rem 0 0;
        color: var(--text-soft);
        font-size: 0.8rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    .trace-list {
        display: grid;
        gap: 0.5rem;
    }

    .trace-item {
        display: grid;
        grid-template-columns: 30px minmax(0, 1fr) auto;
        align-items: center;
        gap: 0.7rem;
        padding: 0.65rem;
        background: var(--app-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
    }

    .trace-item strong {
        color: var(--text);
        font-size: 0.78rem;
    }

    .trace-item p {
        margin: 0.15rem 0 0 !important;
        color: var(--muted) !important;
        font-size: 0.72rem;
    }

    .trace-meta {
        color: var(--muted);
        font-size: 0.68rem;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }

    /* Footer summary */
    .summary-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: 1.5rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
    }

    .summary-item {
        padding: 0.8rem 1rem;
        border-right: 1px solid var(--border);
    }

    .summary-item:last-child { border-right: 0; }

    .summary-label {
        display: block;
        color: var(--muted);
        font-size: 0.69rem;
    }

    .summary-value {
        display: block;
        margin-top: 0.18rem;
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 700;
    }

    [data-testid="stAlert"] {
        color: var(--text-soft);
        border-radius: var(--radius-sm);
        font-size: 0.82rem;
    }

    hr, [data-testid="stDivider"] { border-color: var(--border); }

    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; }
    ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 2.75rem 1rem 6.5rem;
        }

        .workspace-title { font-size: 2rem; }
        .workspace-subtitle { font-size: 0.9rem; }
        .workflow-grid { grid-template-columns: 1fr; }
        .workflow-card { min-height: 0; }
        .welcome-footer { align-items: flex-start; flex-direction: column; }
        .scene-row { justify-content: flex-start; }
        .summary-strip { grid-template-columns: 1fr; }
        .summary-item {
            border-right: 0;
            border-bottom: 1px solid var(--border);
        }
        .summary-item:last-child { border-bottom: 0; }
        [data-baseweb="tab-list"] {
            overflow-x: auto;
            justify-content: flex-start;
        }
        [data-baseweb="tab"] {
            flex: 0 0 auto;
            padding: 0 0.72rem;
        }
        .ops-heading h2 { font-size: 1.45rem; }
        .trace-item {
            grid-template-columns: 28px minmax(0, 1fr);
        }
        .trace-meta {
            grid-column: 2;
            justify-self: start;
        }
    }

    @media (max-width: 480px) {
        .welcome-panel { padding: 1.1rem; }
        .capability-row { gap: 0.35rem; }
        .capability-chip { font-size: 0.69rem; }
        .source-card { grid-template-columns: 28px minmax(0, 1fr); }
        .source-score { grid-column: 2; justify-self: start; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ── 会话状态初始化 ──────────────────────────
def init_session_state():
    """初始化 Streamlit 会话状态中的持久化变量"""
    defaults = {
        "agent_ready": False,
        "llm": None,
        "vector_store": None,
        "memory": None,
        "agent": None,
        "processor": None,
        "messages": [],
        "kb_stats": {"doc_count": 0},
        "retrieval_debug": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if "operations_store" not in st.session_state:
        st.session_state.operations_store = OperationsStore()
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex


# ── 回调：模型连接 ──────────────────────────
def on_init(api_key: str):
    """初始化模型连接回调"""
    try:
        llm = TongyiLLM(api_key=api_key, model_name="qwen3.7-plus", temperature=0.1)
        _ = llm.invoke([HumanMessage(content="你好，请回复: 连接成功")])

        vector_store = VectorStore(api_key=api_key)
        memory = ChatMemoryManager(max_turns=10)
        agent = RAGAgent(llm=llm, vector_store=vector_store, memory=memory)
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)

        st.session_state.llm = llm
        st.session_state.vector_store = vector_store
        st.session_state.memory = memory
        st.session_state.agent = agent
        st.session_state.processor = processor
        st.session_state.agent_ready = True
        st.session_state.kb_stats = vector_store.get_stats()
        st.success(UI_COPY["toast"]["init_success"])
        st.rerun()
    except Exception as e:
        st.error(f"{UI_COPY['toast']['init_error']}：{str(e)}")


# ── 回调：文档导入 ──────────────────────────
def on_upload(uploaded_files):
    """文档上传与知识库构建回调"""
    if not uploaded_files:
        st.warning(UI_COPY["toast"]["upload_empty"])
        return

    processor = st.session_state.processor
    vector_store = st.session_state.vector_store
    all_chunks = []
    file_names = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(tmpdir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                chunks = processor.process_file(file_path)
                ingested_at = datetime.now(timezone.utc).isoformat()
                extension = os.path.splitext(uploaded_file.name)[1].lstrip(".")
                for index, chunk in enumerate(chunks):
                    chunk.metadata["source"] = uploaded_file.name
                    chunk.metadata.pop("file_path", None)
                    chunk.metadata["file_type"] = extension
                    chunk.metadata["ingested_at"] = ingested_at
                    chunk.metadata["chunk_index"] = index
                all_chunks.extend(chunks)
                file_names.append(uploaded_file.name)
            except Exception as e:
                st.warning(
                    UI_COPY["toast"]["upload_file_error"].format(
                        file_name=uploaded_file.name,
                        error=str(e),
                    )
                )
                continue

        if all_chunks:
            count = vector_store.add_documents(all_chunks)
            st.session_state.kb_stats = vector_store.get_stats()
            st.success(
                UI_COPY["toast"]["upload_success"].format(
                    file_count=len(file_names),
                    chunk_count=count,
                )
            )
            st.info(
                UI_COPY["toast"]["upload_info"].format(
                    files="、".join(file_names),
                )
            )
        else:
            st.warning(UI_COPY["toast"]["upload_no_text"])


# ── 回调：清空知识库 ────────────────────────
def on_clear_kb():
    """清空知识库回调"""
    if st.session_state.vector_store:
        st.session_state.vector_store.clear()
        st.session_state.kb_stats = st.session_state.vector_store.get_stats()
        st.success(UI_COPY["toast"]["clear_kb_success"])
        st.rerun()


# ── 回调：清空会话 ──────────────────────────
def on_clear_chat():
    """清空聊天记录回调"""
    st.session_state.messages = []
    if st.session_state.memory:
        st.session_state.memory.clear()
    st.success(UI_COPY["toast"]["clear_chat_success"])
    st.rerun()


def on_delete_document(source: str) -> int:
    """删除指定文档并刷新知识库状态。"""
    deleted = st.session_state.vector_store.delete_document(source)
    st.session_state.kb_stats = st.session_state.vector_store.get_stats()
    return deleted


def on_feedback(
    *,
    question: str,
    answer: str,
    rating: int,
    comment: str,
    intent: str,
) -> dict:
    """保存回答反馈。"""
    return st.session_state.operations_store.add_feedback(
        session_id=st.session_state.session_id,
        question=question,
        answer=answer,
        rating=rating,
        comment=comment,
        intent=intent,
    )


def on_run_evaluation() -> dict:
    """使用独立短期记忆运行全部评测用例。"""
    cases = st.session_state.operations_store.list_evaluation_cases()
    if not cases:
        raise ValueError("没有可运行的评测用例")

    items = []
    for case in cases:
        agent = RAGAgent(
            llm=st.session_state.llm,
            vector_store=st.session_state.vector_store,
            memory=ChatMemoryManager(max_turns=2),
        )
        started = time.perf_counter()
        result = agent.query(case["question"])
        latency_ms = (time.perf_counter() - started) * 1000
        sources = [
            str(item.get("source", ""))
            for item in result.get("sources", [])
        ]
        answer_ok = (
            not case["expected_answer"]
            or case["expected_answer"].lower() in result["answer"].lower()
        )
        source_ok = (
            not case["expected_source"]
            or any(
                case["expected_source"].lower() in source.lower()
                for source in sources
            )
        )
        items.append(
            st.session_state.operations_store.record_evaluation_run(
                case_id=case["id"],
                answer=result["answer"],
                intent=result.get("intent", "unknown"),
                sources=sources,
                passed=answer_ok and source_ok and not result.get("error"),
                latency_ms=latency_ms,
            )
        )
    passed = sum(1 for item in items if item["passed"])
    return {
        "run_count": len(items),
        "passed": passed,
        "pass_rate": round(passed / len(items), 4),
        "items": items,
    }


# ── 主函数 ──────────────────────────────────
def main():
    init_session_state()

    render_sidebar(
        on_init=on_init,
        on_upload=on_upload,
        on_clear_kb=on_clear_kb,
        on_clear_chat=on_clear_chat,
        kb_stats=st.session_state.kb_stats,
        agent_ready=st.session_state.agent_ready,
        ui_text=UI_COPY,
    )

    tab_chat, tab_knowledge, tab_retrieval, tab_evaluation = st.tabs([
        "智能问答",
        "知识库管理",
        "检索调试",
        "评测与反馈",
    ])

    with tab_chat:
        render_chat_area(
            agent_ready=st.session_state.agent_ready,
            kb_stats=st.session_state.kb_stats,
            ui_text=UI_COPY,
            on_feedback=on_feedback,
        )

        if not st.session_state.messages:
            show_welcome(UI_COPY)

        if prompt := st.chat_input(
            UI_COPY["chat"]["input_placeholder"],
            disabled=not st.session_state.agent_ready,
        ):
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat(),
            })
            with st.spinner(UI_COPY["chat"]["spinner"]):
                try:
                    result = st.session_state.agent.query(prompt)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "thinking": result["thinking"],
                        "trace": result.get("trace", []),
                        "intent": result.get("intent", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                    })
                except Exception as e:
                    error_msg = UI_COPY["toast"]["query_error"].format(error=str(e))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "thinking": [],
                        "trace": [],
                        "intent": "error",
                        "timestamp": datetime.now().isoformat(),
                    })
            st.rerun()

    with tab_knowledge:
        render_knowledge_workspace(
            vector_store=st.session_state.vector_store,
            agent_ready=st.session_state.agent_ready,
            on_delete_document=on_delete_document,
        )

    with tab_retrieval:
        render_retrieval_workspace(
            vector_store=st.session_state.vector_store,
            agent_ready=st.session_state.agent_ready,
        )

    with tab_evaluation:
        render_evaluation_workspace(
            operations_store=st.session_state.operations_store,
            agent_ready=st.session_state.agent_ready,
            on_run_evaluation=on_run_evaluation,
        )

    # ── 底部状态摘要 ──
    kb_count = st.session_state.kb_stats.get("doc_count", 0)
    msg_count = len(st.session_state.messages)
    agent_status = (
        UI_COPY["status"]["metrics_ready"]
        if st.session_state.agent_ready
        else UI_COPY["status"]["metrics_pending"]
    )
    st.markdown(
        f"""
        <section class="summary-strip" aria-label="{UI_COPY["main"]["summary_title"]}">
            <div class="summary-item">
                <span class="summary-label">{UI_COPY["metrics"]["kb_count"]}</span>
                <span class="summary-value">{kb_count}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">{UI_COPY["metrics"]["message_count"]}</span>
                <span class="summary-value">{msg_count}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">{UI_COPY["metrics"]["agent_status"]}</span>
                <span class="summary-value">{agent_status}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
