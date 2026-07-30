"""知识运营工作台组件。"""
import html
import time
from typing import Callable, Dict, Optional

import streamlit as st


def _empty_state(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="ops-empty">
            <p class="ops-empty-title">{html.escape(title)}</p>
            <p class="ops-empty-description">{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_knowledge_workspace(
    *,
    vector_store,
    agent_ready: bool,
    on_delete_document: Callable[[str], int],
) -> None:
    """文档级知识库管理与切块预览。"""
    st.markdown(
        """
        <div class="ops-heading">
            <div>
                <p class="ops-eyebrow">KNOWLEDGE OPERATIONS</p>
                <h2>知识库管理</h2>
                <p>查看文档处理结果、片段规模和实际切块内容。</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not agent_ready:
        _empty_state("模型尚未连接", "请先在侧栏连接模型，然后导入企业文档。")
        return

    documents = vector_store.list_documents()
    if not documents:
        _empty_state("知识库暂无文档", "从侧栏上传 TXT、PDF 或 Markdown 文档开始构建知识库。")
        return

    total_chunks = sum(item["chunk_count"] for item in documents)
    total_chars = sum(item["char_count"] for item in documents)
    col1, col2, col3 = st.columns(3)
    col1.metric("文档数", len(documents))
    col2.metric("知识片段", total_chunks)
    col3.metric("文本字符", f"{total_chars:,}")

    rows = [{
        "文档": item["source"],
        "类型": (item["file_type"] or "未知").upper(),
        "片段数": item["chunk_count"],
        "字符数": item["char_count"],
        "导入时间": item["ingested_at"][:19].replace("T", " ")
        if item["ingested_at"] else "历史数据",
    } for item in documents]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("#### 切块预览")
    selected_source = st.selectbox(
        "选择文档",
        options=[item["source"] for item in documents],
        help="查看该文档实际写入向量库的文本片段。",
    )
    chunks = vector_store.get_document_chunks(selected_source, limit=100)
    for index, chunk in enumerate(chunks, 1):
        label = f"片段 {index:02d} · {chunk['char_count']} 字符"
        with st.expander(label):
            st.text(chunk["content"])

    with st.expander("危险操作"):
        st.caption("删除后需要重新上传文档才能恢复。")
        confirmed = st.checkbox(
            f"确认删除《{selected_source}》的全部片段",
            key=f"confirm_delete_{selected_source}",
        )
        if st.button(
            "删除当前文档",
            type="secondary",
            disabled=not confirmed,
            key=f"delete_{selected_source}",
        ):
            deleted = on_delete_document(selected_source)
            if deleted:
                st.success(f"已删除 {deleted} 个片段")
                st.rerun()
            st.error("未找到可删除的文档片段")


def render_retrieval_workspace(*, vector_store, agent_ready: bool) -> None:
    """Top-K 检索调试工作台。"""
    st.markdown(
        """
        <div class="ops-heading">
            <div>
                <p class="ops-eyebrow">RETRIEVAL LAB</p>
                <h2>检索调试台</h2>
                <p>绕过回答生成，直接检查向量检索的召回片段、分数和耗时。</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not agent_ready:
        _empty_state("检索服务尚未连接", "连接模型并导入文档后即可进行检索测试。")
        return

    with st.form("retrieval_debug_form"):
        query = st.text_input(
            "测试问题",
            placeholder="例如：员工差旅费应该如何报销？",
        )
        top_k = st.slider("召回数量 Top-K", min_value=1, max_value=12, value=6)
        submitted = st.form_submit_button(
            "运行检索",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not query.strip():
            st.warning("请输入测试问题")
        else:
            with st.spinner("正在检索向量库…"):
                started = time.perf_counter()
                try:
                    results = vector_store.similarity_search(query.strip(), k=top_k)
                    st.session_state.retrieval_debug = {
                        "query": query.strip(),
                        "results": results,
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000,
                            2,
                        ),
                    }
                except Exception as exc:
                    st.error(f"检索失败：{exc}")

    debug_result: Optional[Dict] = st.session_state.get("retrieval_debug")
    if not debug_result:
        _empty_state("等待测试问题", "运行一次检索后，这里会展示实际召回结果。")
        return

    results = debug_result["results"]
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("召回片段", len(results))
    metric2.metric("检索耗时", f"{debug_result['latency_ms']:.2f} ms")
    metric3.metric(
        "最高匹配度",
        f"{results[0]['score']:.1%}" if results else "0.0%",
    )
    st.caption(f"测试问题：{debug_result['query']}")

    if not results:
        _empty_state("未召回相关片段", "尝试修改问题表达，或检查知识库中是否存在相关内容。")
        return

    for index, result in enumerate(results, 1):
        source = html.escape(str(result["metadata"].get("source", "未知来源")))
        score = float(result.get("score", 0))
        st.markdown(
            f"""
            <article class="retrieval-card">
                <div class="retrieval-card-head">
                    <span class="retrieval-rank">{index:02d}</span>
                    <strong>{source}</strong>
                    <span class="retrieval-score">{score:.1%}</span>
                </div>
                <p>{html.escape(result["content"])}</p>
            </article>
            """,
            unsafe_allow_html=True,
        )


def render_evaluation_workspace(
    *,
    operations_store,
    agent_ready: bool,
    on_run_evaluation: Callable[[], Dict],
) -> None:
    """评测集、运行结果和用户反馈仪表盘。"""
    st.markdown(
        """
        <div class="ops-heading">
            <div>
                <p class="ops-eyebrow">QUALITY LOOP</p>
                <h2>评测与反馈</h2>
                <p>把“感觉回答不错”变成可重复执行的测试结果和用户反馈证据。</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_summary = operations_store.evaluation_summary()
    feedback_summary = operations_store.feedback_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("评测用例", eval_summary["case_count"])
    col2.metric("历史运行", eval_summary["run_count"])
    col3.metric("通过率", f"{eval_summary['pass_rate']:.1%}")
    col4.metric("反馈满意度", f"{feedback_summary['positive_rate']:.1%}")

    tab_cases, tab_runs, tab_feedback = st.tabs(["评测用例", "运行记录", "用户反馈"])

    with tab_cases:
        with st.form("add_evaluation_case"):
            question = st.text_input(
                "评测问题",
                placeholder="例如：员工每年有多少天年假？",
            )
            expected_answer = st.text_input(
                "期望答案关键词",
                placeholder="例如：5天",
                help="只校验关键词是否出现在回答中，可留空。",
            )
            expected_source = st.text_input(
                "期望来源",
                placeholder="例如：企业员工手册.txt",
                help="只校验来源名称是否包含该文本，可留空。",
            )
            add_case = st.form_submit_button("添加评测用例", use_container_width=True)

        if add_case:
            try:
                operations_store.add_evaluation_case(
                    question=question,
                    expected_answer=expected_answer,
                    expected_source=expected_source,
                )
                st.success("评测用例已添加")
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))

        cases = operations_store.list_evaluation_cases()
        if cases:
            st.dataframe(
                [{
                    "问题": item["question"],
                    "答案关键词": item["expected_answer"] or "不校验",
                    "期望来源": item["expected_source"] or "不校验",
                    "创建时间": item["created_at"][:19].replace("T", " "),
                } for item in cases],
                use_container_width=True,
                hide_index=True,
            )
            if st.button(
                "运行全部评测",
                type="primary",
                use_container_width=True,
                disabled=not agent_ready,
            ):
                with st.spinner("正在逐条运行评测用例…"):
                    result = on_run_evaluation()
                st.success(
                    f"评测完成：{result['passed']}/{result['run_count']} 通过"
                )
                st.rerun()
        else:
            _empty_state("尚无评测用例", "至少添加一条问题，并设置答案关键词或期望来源。")

    with tab_runs:
        runs = operations_store.list_evaluation_runs()
        if runs:
            st.dataframe(
                [{
                    "结果": "通过" if item["passed"] else "未通过",
                    "问题": item["question"],
                    "来源": "、".join(item["sources"]) or "无",
                    "耗时(ms)": item["latency_ms"],
                    "运行时间": item["created_at"][:19].replace("T", " "),
                } for item in runs],
                use_container_width=True,
                hide_index=True,
            )
        else:
            _empty_state("暂无运行记录", "添加评测用例并运行后，这里会保留结果和耗时。")

    with tab_feedback:
        feedback = operations_store.list_feedback()
        if feedback:
            st.dataframe(
                [{
                    "评价": "有帮助" if item["rating"] == 1 else "需改进",
                    "问题": item["question"],
                    "意见": item["comment"] or "未填写",
                    "类型": item["intent"],
                    "时间": item["created_at"][:19].replace("T", " "),
                } for item in feedback],
                use_container_width=True,
                hide_index=True,
            )
        else:
            _empty_state("暂无用户反馈", "用户在回答下方提交评价后，这里会自动汇总。")


def render_trace(trace: list[Dict]) -> None:
    """展示结构化 Agent 运行轨迹。"""
    if not trace:
        st.caption("本次运行没有可用轨迹。")
        return
    items = []
    for index, event in enumerate(trace, 1):
        status = html.escape(str(event.get("status", "unknown")))
        label = html.escape(str(event.get("label", event.get("step", "步骤"))))
        detail = html.escape(str(event.get("detail", "")))
        duration = float(event.get("duration_ms", 0))
        items.append(
            f"""
            <div class="trace-item">
                <span class="trace-index">{index:02d}</span>
                <div>
                    <strong>{label}</strong>
                    <p>{detail}</p>
                </div>
                <span class="trace-meta">{status} · {duration:.2f} ms</span>
            </div>
            """
        )
    st.markdown(
        f'<div class="trace-list">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


def render_feedback_form(
    *,
    message: Dict,
    question: str,
    message_index: int,
    on_feedback: Callable[..., Dict],
) -> None:
    """回答级反馈表单。"""
    if not question or message.get("feedback_saved"):
        if message.get("feedback_saved"):
            st.caption("已记录本次回答评价")
        return

    with st.expander("评价这次回答"):
        with st.form(f"feedback_form_{message_index}"):
            rating_label = st.radio(
                "回答是否有帮助？",
                ["有帮助", "需要改进"],
                horizontal=True,
            )
            comment = st.text_input(
                "补充意见（可选）",
                placeholder="例如：来源正确，但回答缺少审批时限。",
            )
            submitted = st.form_submit_button("提交评价")
        if submitted:
            on_feedback(
                question=question,
                answer=message["content"],
                rating=1 if rating_label == "有帮助" else -1,
                comment=comment,
                intent=message.get("intent", "unknown"),
            )
            message["feedback_saved"] = True
            st.success("感谢反馈，已进入质量看板")
            st.rerun()
