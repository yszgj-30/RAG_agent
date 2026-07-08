"""
企业知识库 RAG 问答 Agent —— 主程序入口
技术栈：通义千问 API + LangChain + Chroma + Streamlit
启动方式：streamlit run app.py
"""
import os
import tempfile
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage

from agent.core import RAGAgent
from document.processor import DocumentProcessor
from memory.manager import ChatMemoryManager
from models.tongyi_llm import TongyiLLM
from ui.layout import render_chat_area, render_sidebar, render_sources, show_welcome
from vectordb.store import VectorStore


# ── 页面配置 ────────────────────────────────
st.set_page_config(
    page_title="企业知识库问答",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
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
        "title": "知识库问答工作台",
        "subtitle": "基于企业文档的智能检索与问答",
        "capabilities": ["知识检索", "业务问答", "来源追溯", "多轮对话"],
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
        "input_placeholder": "输入业务问题，系统将基于知识库作答",
        "spinner": "检索中…",
        "expander_thinking": "处理过程",
        "expander_sources": "参考来源",
        "no_sources": "本次回答未引用知识库内容",
        "error_prefix": "问答异常",
    },
    "welcome": {
        "title": "欢迎使用企业知识库工作台",
        "subtitle": "将制度、流程、产品资料沉淀为可检索、可追溯的企业知识资产",
        "steps_title": "使用流程",
        "steps": [
            ("连接模型", "配置 API Key，建立大模型服务连接"),
            ("导入知识", "上传业务文档，自动解析并写入知识库"),
            ("开始检索", "输入问题，获取基于知识库的准确回答"),
        ],
        "abilities_title": "核心能力",
        "abilities": ["语义检索", "多轮对话", "来源引用", "结果追溯"],
        "tip": "适用于制度手册、产品资料、流程规范、培训文档等企业内容",
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
        --bg-color: #F4F6F9;
        --surface-color: #FFFFFF;
        --surface-soft: #F7F8FA;
        --sidebar-bg: #F5F6F9;
        --text-color: #1A1F2E;
        --text-secondary: #3A3F4E;
        --muted-color: #7B8296;
        --border-color: #E5E8ED;
        --accent-color: #3366CC;
        --accent-hover: #2952A3;
        --accent-soft: #EDF2FB;
        --success-color: #10A868;
        --success-soft: #ECF9F4;
        --warning-color: #E88A20;
        --warning-soft: #FEF7EE;
        --error-color: #DC3A3A;
        --error-soft: #FEF3F3;
        --radius-xs: 6px;
        --radius-sm: 8px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.04);
        --transition: 160ms ease;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                     "Microsoft YaHei", "Helvetica Neue", sans-serif;
        color: var(--text-color);
    }

    body { background: var(--bg-color); }

    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
                     "Microsoft YaHei", sans-serif;
        color: var(--text-color);
        letter-spacing: 0;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--bg-color);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 246, 249, 0.9);
        border-bottom: 1px solid var(--border-color);
        backdrop-filter: blur(8px);
    }

    [data-testid="stToolbar"] { background: transparent; }

    [data-testid="stMainBlockContainer"] {
        max-width: 1140px;
        padding: 1.75rem 1.5rem 1rem;
    }

    /* ── 侧边栏 ── */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] > div { background: transparent; }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        gap: 0.75rem;
    }

    /* ── 通用卡片 ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stMetric"] {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-xs);
        padding: 0.75rem 0.9rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted-color); font-weight: 500; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: var(--text-color); font-weight: 700; font-size: 1.35rem; }

    /* ── 聊天消息 ── */
    div[data-testid="stChatMessage"] {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-xs);
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.75rem;
    }

    /* ── 折叠面板 ── */
    [data-testid="stExpander"] {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        box-shadow: none;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }

    /* ── 按钮 ── */
    .stButton > button {
        height: 2.5rem;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-color);
        background: var(--surface-color);
        color: var(--text-color);
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0 1.1rem;
        transition: all var(--transition);
        box-shadow: var(--shadow-xs);
    }
    .stButton > button:hover:not(:disabled) {
        border-color: var(--accent-color);
        color: var(--accent-color);
        background: var(--accent-soft);
        box-shadow: var(--shadow-sm);
        transform: none;
    }
    .stButton > button:active:not(:disabled) {
        transform: scale(0.985);
    }
    .stButton > button:disabled {
        background: #F5F6F8;
        color: #B0B7C3;
        border-color: #E8EBF0;
    }

    /* 侧边栏主按钮加 accent */
    [data-testid="stSidebar"] .stButton > button {
        border-color: var(--accent-color);
        color: var(--accent-color);
        background: var(--accent-soft);
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
        background: var(--accent-color);
        color: #FFFFFF;
    }

    /* ── 输入框 ── */
    .stTextInput > div > div > input,
    [data-testid="stChatInput"] textarea {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-color);
        background: var(--surface-color);
        color: var(--text-color);
        padding: 0.6rem 0.8rem;
        font-size: 0.9rem;
    }
    .stTextInput > div > div > input:focus,
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(51, 102, 204, 0.1);
    }

    .stChatInputContainer,
    [data-testid="stChatInput"] { background: transparent; }

    /* ── 文件上传 ── */
    .stFileUploader > div {
        border-radius: var(--radius-md);
        border: 1px dashed #CDD2DB;
        background: var(--surface-soft);
        transition: all var(--transition);
    }
    .stFileUploader > div:hover {
        border-color: var(--accent-color);
        background: #F5F7FC;
    }

    /* ── 提示 / 通知 ── */
    [data-testid="stAlert"] {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-color);
        font-size: 0.85rem;
    }
    .stSuccess { background: var(--success-soft); border-color: #C6EDDB; }
    .stWarning { background: var(--warning-soft); border-color: #FDDCB5; }
    .stError   { background: var(--error-soft);   border-color: #F9C9C9; }
    .stInfo    { background: var(--accent-soft);   border-color: #CDDDF5; }

    .stSpinner > div:first-child {
        border-top-color: var(--accent-color);
        border-right-color: var(--accent-color);
    }

    /* ── 品牌标识 ── */
    .brand-block {
        padding: 1rem 0 0.85rem;
        margin-bottom: 0.25rem;
        border-bottom: 1px solid var(--border-color);
    }
    .brand-inner {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: var(--accent-color);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(51, 102, 204, 0.2);
    }
    .brand-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-color);
        margin: 0;
    }
    .brand-subtitle,
    .soft-note {
        font-size: 0.78rem;
        color: var(--muted-color);
        margin: 0.15rem 0 0;
        line-height: 1.45;
    }

    /* ── 区块标题 ── */
    .section-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 0.55rem;
    }
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 0;
    }
    .section-tag {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 24px;
        height: 24px;
        padding: 0 6px;
        border-radius: 6px;
        background: var(--accent-soft);
        color: var(--accent-color);
        font-size: 0.7rem;
        font-weight: 700;
    }

    /* ── 状态标签 ── */
    .status-chip {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 0.5rem 0.7rem;
        margin-top: 0.45rem;
        border-radius: var(--radius-sm);
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid transparent;
    }
    .status-chip.success {
        background: var(--success-soft);
        color: var(--success-color);
        border-color: #C6EDDB;
    }
    .status-chip.warning {
        background: var(--warning-soft);
        color: var(--warning-color);
        border-color: #FDDCB5;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        display: inline-block;
        background: currentColor;
        opacity: 0.85;
    }

    /* ── 统计卡片 ── */
    .count-card {
        padding: 0.65rem 0.8rem;
        border-radius: var(--radius-sm);
        background: var(--surface-soft);
        border: 1px solid var(--border-color);
    }
    .count-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-color);
        margin: 0;
    }
    .count-label {
        font-size: 0.72rem;
        color: var(--muted-color);
        margin: 0.1rem 0 0;
    }

    /* ── 能力标签 ── */
    .capability-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.35rem 0 0.85rem;
    }
    .capability-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* ── 分割线 ── */
    hr, [data-testid="stDivider"] {
        border-color: var(--border-color);
        margin: 0.75rem 0;
    }

    /* ── 滚动条 ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #EDF0F5; border-radius: 999px; }
    ::-webkit-scrollbar-thumb { background: #CBD0D8; border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: #A8AFBB; }
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
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


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

    render_chat_area(
        agent_ready=st.session_state.agent_ready,
        kb_stats=st.session_state.kb_stats,
        ui_text=UI_COPY,
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

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(UI_COPY["chat"]["spinner"]):
                try:
                    result = st.session_state.agent.query(prompt)

                    with st.expander(UI_COPY["chat"]["expander_thinking"], expanded=False):
                        for step in result["thinking"]:
                            st.caption(step)

                    if result["sources"]:
                        with st.expander(UI_COPY["chat"]["expander_sources"], expanded=False):
                            render_sources(result["sources"], UI_COPY)

                    st.markdown(result["answer"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "thinking": result["thinking"],
                        "timestamp": datetime.now().isoformat(),
                    })

                except Exception as e:
                    error_msg = UI_COPY["toast"]["query_error"].format(error=str(e))
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "thinking": [],
                        "timestamp": datetime.now().isoformat(),
                    })

    # ── 底部指标 ──
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        kb_count = st.session_state.kb_stats.get("doc_count", 0)
        st.metric(UI_COPY["metrics"]["kb_count"], kb_count)
    with col_f2:
        msg_count = len(st.session_state.messages)
        st.metric(UI_COPY["metrics"]["message_count"], msg_count)
    with col_f3:
        agent_status = (
            UI_COPY["status"]["metrics_ready"]
            if st.session_state.agent_ready
            else UI_COPY["status"]["metrics_pending"]
        )
        st.metric(UI_COPY["metrics"]["agent_status"], agent_status)


if __name__ == "__main__":
    main()
