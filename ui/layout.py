"""
UI 交互界面层 —— Streamlit 页面布局与组件渲染
提供侧边栏配置、聊天区域、资料来源展示等标准化组件
"""
import html
from typing import Callable, Dict, List, Optional

import streamlit as st

from ui.workspaces import render_feedback_form, render_trace


def render_sidebar(
    on_init: callable,
    on_upload: callable,
    on_clear_kb: callable,
    on_clear_chat: callable,
    kb_stats: Dict,
    agent_ready: bool,
    ui_text: Dict,
) -> None:
    """渲染侧边栏：模型连接、文档导入、知识库状态、会话管理"""
    sidebar = ui_text["sidebar"]

    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-block">
                <div class="brand-inner">
                    <div class="brand-mark">R</div>
                    <div>
                        <p class="brand-title">{sidebar["brand_title"]}</p>
                        <p class="brand-subtitle">{sidebar["brand_subtitle"]}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="section-head">
                    <p class="section-title">{sidebar["api_title"]}</p>
                    <span class="section-tag">API</span>
                </div>
                <p class="soft-note">{sidebar["api_hint"]}</p>
                """,
                unsafe_allow_html=True,
            )

            api_key = st.text_input(
                sidebar["api_title"],
                type="password",
                placeholder=sidebar["api_placeholder"],
                help=sidebar["api_help"],
                label_visibility="collapsed",
            )

            if st.button(
                sidebar["api_button"],
                use_container_width=True,
                disabled=not api_key,
                type="primary",
            ):
                on_init(api_key)

            status_class = "success" if agent_ready else "warning"
            status_text = sidebar["api_ready"] if agent_ready else sidebar["api_idle"]
            st.markdown(
                f"""
                <div class="status-chip {status_class}">
                    <span class="status-dot"></span>
                    <span>{status_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="section-head">
                    <p class="section-title">{sidebar["upload_title"]}</p>
                    <span class="section-tag">DOC</span>
                </div>
                <p class="soft-note">{sidebar["upload_hint"]}</p>
                """,
                unsafe_allow_html=True,
            )

            uploaded_files = st.file_uploader(
                sidebar["upload_title"],
                type=["txt", "pdf", "md"],
                accept_multiple_files=True,
                help=sidebar["upload_help"],
                label_visibility="collapsed",
            )

            if uploaded_files:
                st.caption(sidebar["upload_selected"].format(count=len(uploaded_files)))

            col_left, col_right = st.columns(2)
            with col_left:
                if st.button(
                    sidebar["upload_build"],
                    use_container_width=True,
                    disabled=not agent_ready or not uploaded_files,
                    type="primary",
                ):
                    on_upload(uploaded_files)
            with col_right:
                if st.button(
                    sidebar["upload_clear"],
                    use_container_width=True,
                    disabled=not agent_ready,
                ):
                    on_clear_kb()

            kb_state = (
                sidebar["store_empty"]
                if kb_stats.get("doc_count", 0) == 0
                else sidebar["api_ready"]
            )
            st.markdown(
                f"""
                <div class="knowledge-stat">
                    <div>
                        <p class="count-value">{kb_stats.get("doc_count", 0)}</p>
                        <p class="count-label">{sidebar["store_label"]}</p>
                    </div>
                    <span class="kb-state">{kb_state}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if kb_stats.get("doc_count", 0) > 0:
                st.caption(
                    f'{sidebar["store_collection"]}：{kb_stats.get("collection", "N/A")}'
                )

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="section-head">
                    <p class="section-title">{sidebar["chat_title"]}</p>
                    <span class="section-tag">CHAT</span>
                </div>
                <p class="soft-note">{sidebar["chat_hint"]}</p>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                sidebar["chat_button"],
                use_container_width=True,
                disabled=len(st.session_state.get("messages", [])) == 0,
            ):
                on_clear_chat()

        st.markdown(
            f"""
            <div class="sidebar-footer">
                <p>{sidebar["footer_line1"]}</p>
                <p>{sidebar["footer_line2"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_area(
    agent_ready: bool,
    kb_stats: Dict,
    ui_text: Dict,
    on_feedback: Optional[Callable] = None,
) -> None:
    """渲染主区域聊天消息与顶部状态提示"""
    main = ui_text["main"]
    status = ui_text["status"]
    chat = ui_text["chat"]

    badges = "".join(
        f'<span class="capability-chip">{item}</span>' for item in main["capabilities"]
    )
    st.markdown(
        f"""
        <section class="workspace-hero">
            <p class="hero-eyebrow">{main["eyebrow"]}</p>
            <h1 class="workspace-title">{main["title"]}</h1>
            <p class="workspace-subtitle">{main["subtitle"]}</p>
            <div class="capability-row">{badges}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not agent_ready:
        status_class = "idle"
        status_text = status["main_idle"]
    elif kb_stats.get("doc_count", 0) == 0:
        status_class = "idle"
        status_text = status["main_ready_no_docs"]
    else:
        status_class = "ready"
        status_text = status["main_ready"]

    st.markdown(
        f"""
        <div class="workspace-status {status_class}" role="status">
            <span class="status-dot"></span>
            <span>{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    previous_question = ""
    for index, msg in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(html.escape(msg["content"]))
                previous_question = msg["content"]
            else:
                st.markdown(msg["content"])
                if msg.get("trace"):
                    with st.expander(chat["expander_thinking"], expanded=False):
                        render_trace(msg["trace"])
                elif msg.get("thinking"):
                    with st.expander(chat["expander_thinking"], expanded=False):
                        for i, step in enumerate(msg["thinking"], 1):
                            st.markdown(f"**{i}.** {step}")
                if msg.get("sources"):
                    with st.expander(chat["expander_sources"], expanded=False):
                        render_sources(msg["sources"], ui_text)
                if on_feedback:
                    render_feedback_form(
                        message=msg,
                        question=previous_question,
                        message_index=index,
                        on_feedback=on_feedback,
                    )


def render_sources(sources: List[Dict], ui_text: Dict) -> None:
    """渲染检索资料来源卡片"""
    if not sources:
        st.warning(ui_text["chat"]["no_sources"])
        return

    cards = []
    for i, src in enumerate(sources, 1):
        source_name = html.escape(str(src.get("source", "未知来源")))
        score = float(src.get("score", 0))
        cards.append(
            f"""
            <div class="source-card">
                <span class="source-index">{i:02d}</span>
                <span class="source-name">{source_name}</span>
                <span class="source-score">匹配 {score:.1%}</span>
            </div>
            """
        )
    st.markdown(
        f'<div class="source-list">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def show_welcome(ui_text: Dict) -> None:
    """显示欢迎引导信息"""
    welcome = ui_text["welcome"]
    workflow_cards = "".join(
        f"""
        <div class="workflow-card">
            <span class="workflow-number">{i:02d}</span>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """
        for i, (title, desc) in enumerate(welcome["steps"], 1)
    )
    scene_chips = "".join(
        f'<span class="scene-chip">{item}</span>' for item in welcome["abilities"]
    )

    st.markdown(
        f"""
        <section class="welcome-panel">
            <p class="welcome-eyebrow">{welcome["eyebrow"]}</p>
            <h2 class="welcome-title">{welcome["title"]}</h2>
            <p class="welcome-subtitle">{welcome["subtitle"]}</p>
            <div class="workflow-grid">{workflow_cards}</div>
            <div class="welcome-footer">
                <p class="welcome-tip">{welcome["tip"]}</p>
                <div class="scene-row">{scene_chips}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
