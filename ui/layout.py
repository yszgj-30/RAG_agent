"""
UI 交互界面层 —— Streamlit 页面布局与组件渲染
提供侧边栏配置、聊天区域、资料来源展示等标准化组件
"""
import html
from typing import Dict, List

import streamlit as st


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
                "",
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
                "",
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

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="section-head">
                    <p class="section-title">{sidebar["store_title"]}</p>
                    <span class="section-tag">KB</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="count-card">
                    <p class="count-value">{kb_stats.get("doc_count", 0)}</p>
                    <p class="count-label">{sidebar["store_label"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if kb_stats.get("doc_count", 0) > 0:
                st.caption(
                    f'{sidebar["store_collection"]}：{kb_stats.get("collection", "N/A")}'
                )
            else:
                st.caption(sidebar["store_empty"])

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
            <div style="margin-top: 1.25rem; padding-top: 0.9rem; border-top: 1px solid #E2E8F0;">
                <p style="font-size: 0.76rem; color: #94A3B8; text-align: center; margin: 0;">
                    {sidebar["footer_line1"]}
                </p>
                <p style="font-size: 0.76rem; color: #94A3B8; text-align: center; margin: 0.15rem 0 0;">
                    {sidebar["footer_line2"]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_area(agent_ready: bool, kb_stats: Dict, ui_text: Dict) -> None:
    """渲染主区域聊天消息与顶部状态提示"""
    main = ui_text["main"]
    status = ui_text["status"]
    chat = ui_text["chat"]

    st.title(main["title"])
    st.caption(main["subtitle"])

    badges = "".join(
        f'<span class="capability-chip">{item}</span>' for item in main["capabilities"]
    )
    st.markdown(f'<div class="capability-row">{badges}</div>', unsafe_allow_html=True)

    if not agent_ready:
        st.info(status["main_idle"])
    elif kb_stats.get("doc_count", 0) == 0:
        st.info(status["main_ready_no_docs"])
    else:
        st.success(status["main_ready"])

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(html.escape(msg["content"]))
            else:
                st.markdown(msg["content"])
                if msg.get("thinking"):
                    with st.expander(chat["expander_thinking"], expanded=False):
                        for i, step in enumerate(msg["thinking"], 1):
                            st.markdown(f"**{i}.** {step}")
                if msg.get("sources"):
                    with st.expander(chat["expander_sources"], expanded=False):
                        render_sources(msg["sources"], ui_text)


def render_sources(sources: List[Dict], ui_text: Dict) -> None:
    """渲染检索资料来源卡片"""
    if not sources:
        st.warning(ui_text["chat"]["no_sources"])
        return

    cols = st.columns(min(len(sources), 3))
    for i, src in enumerate(sources):
        with cols[i % 3]:
            score_color = "green" if src.get("score", 0) >= 0.7 else "orange"
            with st.container(border=True):
                st.markdown(f'**{src["source"]}**')
                st.markdown(f'匹配度：:{score_color}[**{src["score"]:.1%}**]')


def show_welcome(ui_text: Dict) -> None:
    """显示欢迎引导信息"""
    welcome = ui_text["welcome"]
    _, center_col, _ = st.columns([1, 3, 1])

    with center_col:
        st.markdown(f"#### {welcome['title']}")
        st.caption(welcome["subtitle"])
        st.divider()
        st.markdown(f"##### {welcome['steps_title']}")

        for i, (title, desc) in enumerate(welcome["steps"], 1):
            col_left, col_right = st.columns([1, 12])
            with col_left:
                st.markdown(f"**{i}**")
            with col_right:
                st.markdown(f"**{title}**  \n{desc}")

        st.divider()
        st.markdown(f"##### {welcome['abilities_title']}")
        ability_badges = "".join(
            f'<span class="capability-chip">{item}</span>'
            for item in welcome["abilities"]
        )
        st.markdown(f'<div class="capability-row">{ability_badges}</div>', unsafe_allow_html=True)
        st.info(welcome["tip"])
