"""Run a reproducible retrieval and answer-quality benchmark.

The benchmark uses the repository's real document processor, DashScope
embeddings, Chroma retrieval, and an OpenAI-compatible chat endpoint. It keeps
the evaluation collection isolated in a temporary directory and exports both
raw case results and a Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import httpx

from document.processor import DocumentProcessor
from embeddings.ollama_embedding import OllamaEmbeddings
from vectordb.store import VectorStore


DEFAULT_DOCUMENTS = [
    "企业员工手册.txt",
    "报销流程规范.txt",
    "考勤管理制度.txt",
    "智能扫地机器人使用指南.md",
    "智能摄像头产品手册.pdf",
]
REFUSAL_MARKERS = (
    "资料不足",
    "未提及",
    "没有提及",
    "未提供",
    "无法确定",
    "无法根据",
    "没有相关",
    "未找到",
    "知识库中没有",
    "文档中没有",
    "无法回答",
)
SYSTEM_PROMPT = """你是企业知识库评测助手，只能根据给定参考资料回答。
要求：
1. 不得使用参考资料之外的知识补充答案。
2. 如果资料中没有问题所需的明确信息，只回答“资料不足，无法根据知识库回答。”
3. 如果资料足够，给出简洁答案，最后一行必须严格写成【来源：文件名】。
4. 不要解释你的推理过程。"""


def load_dataset(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测数据集必须包含非空 cases 列表")

    ids = [str(case.get("id", "")).strip() for case in cases]
    if any(not case_id for case_id in ids):
        raise ValueError("每条评测用例都必须包含 id")
    if len(ids) != len(set(ids)):
        raise ValueError("评测用例 id 不能重复")

    for case in cases:
        if not str(case.get("question", "")).strip():
            raise ValueError(f"{case['id']} 缺少 question")
        if case.get("answerable"):
            if not str(case.get("expected_source", "")).strip():
                raise ValueError(f"{case['id']} 缺少 expected_source")
            if not case.get("expected_terms"):
                raise ValueError(f"{case['id']} 缺少 expected_terms")
    return payload


def normalize_text(value: str) -> str:
    punctuation = " \t\r\n，。！？；：、,.!?;:（）()【】[]《》<>“”\"'`"
    normalized = str(value).lower()
    for char in punctuation:
        normalized = normalized.replace(char, "")
    return normalized


def matches_expected_terms(
    answer: str,
    expected_groups: Sequence[Sequence[str]],
) -> bool:
    normalized = normalize_text(answer)
    return all(
        any(normalize_text(alternative) in normalized for alternative in group)
        for group in expected_groups
    )


def answer_context_rank(
    retrieved: Sequence[Dict[str, Any]],
    expected_source: str | None,
    expected_groups: Sequence[Sequence[str]],
    limit: int,
) -> int | None:
    """查找同时满足正确来源和答案关键词的首个上下文排名。"""
    if not expected_source:
        return None
    normalized_source = normalize_text(expected_source)
    for rank, item in enumerate(retrieved[:limit], 1):
        source = item.get("metadata", {}).get("source", "")
        if normalize_text(source) != normalized_source:
            continue
        if matches_expected_terms(item.get("content", ""), expected_groups):
            return rank
    return None


def is_refusal(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(normalize_text(marker) in normalized for marker in REFUSAL_MARKERS)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 2)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize_results(results: Sequence[Dict[str, Any]], hit_k: int) -> Dict:
    answerable = [item for item in results if item["answerable"]]
    unanswerable = [item for item in results if not item["answerable"]]
    generated = [item for item in results if item.get("generation_attempted")]
    generated_answerable = [item for item in generated if item["answerable"]]
    generated_unanswerable = [item for item in generated if not item["answerable"]]

    category_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in answerable:
        category_items[item["category"]].append(item)

    category_metrics = {}
    for category, items in sorted(category_items.items()):
        category_metrics[category] = {
            "case_count": len(items),
            f"retrieval_hit_at_{hit_k}": _rate(
                sum(bool(item["retrieval_hit"]) for item in items),
                len(items),
            ),
            f"answer_context_hit_at_{hit_k}": _rate(
                sum(bool(item["answer_context_hit"]) for item in items),
                len(items),
            ),
            "answer_keyword_pass_rate": _rate(
                sum(bool(item.get("answer_pass")) for item in items),
                sum(bool(item.get("generation_attempted")) for item in items),
            ),
        }

    retrieval_latencies = [item["retrieval_latency_ms"] for item in results]
    generation_latencies = [
        item["generation_latency_ms"] for item in generated
        if item.get("generation_latency_ms") is not None
    ]
    total_latencies = [
        item["total_latency_ms"] for item in generated
        if item.get("total_latency_ms") is not None
    ]
    reciprocal_ranks = [
        1 / item["source_rank"] if item.get("source_rank") else 0.0
        for item in answerable
    ]
    answer_context_reciprocal_ranks = [
        1 / item["answer_context_rank"]
        if item.get("answer_context_rank")
        else 0.0
        for item in answerable
    ]

    return {
        "total_cases": len(results),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        f"retrieval_hit_at_{hit_k}": _rate(
            sum(bool(item["retrieval_hit"]) for item in answerable),
            len(answerable),
        ),
        f"retrieval_mrr_at_{hit_k}": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 4
        ) if reciprocal_ranks else 0.0,
        f"answer_context_hit_at_{hit_k}": _rate(
            sum(bool(item["answer_context_hit"]) for item in answerable),
            len(answerable),
        ),
        f"answer_context_mrr_at_{hit_k}": round(
            sum(answer_context_reciprocal_ranks)
            / len(answer_context_reciprocal_ranks),
            4,
        ) if answer_context_reciprocal_ranks else 0.0,
        "answer_keyword_pass_rate": _rate(
            sum(bool(item.get("answer_pass")) for item in generated_answerable),
            len(generated_answerable),
        ),
        "source_citation_rate": _rate(
            sum(bool(item.get("source_cited")) for item in generated_answerable),
            len(generated_answerable),
        ),
        "refusal_accuracy": _rate(
            sum(bool(item.get("refusal_pass")) for item in generated_unanswerable),
            len(generated_unanswerable),
        ),
        "generation_error_rate": _rate(
            sum(bool(item.get("generation_error")) for item in generated),
            len(generated),
        ),
        "retrieval_latency_ms": {
            "p50": percentile(retrieval_latencies, 0.50),
            "p95": percentile(retrieval_latencies, 0.95),
        },
        "generation_latency_ms": {
            "p50": percentile(generation_latencies, 0.50),
            "p95": percentile(generation_latencies, 0.95),
        },
        "total_latency_ms": {
            "p50": percentile(total_latencies, 0.50),
            "p95": percentile(total_latencies, 0.95),
        },
        "category_metrics": category_metrics,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_dirty(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else True


def ingest_documents(
    root: Path,
    store: VectorStore,
    processor: DocumentProcessor,
    document_names: Sequence[str],
) -> Dict[str, Any]:
    all_chunks = []
    documents = []
    for name in document_names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"评测文档不存在: {path}")
        chunks = processor.process_file(str(path))
        for index, chunk in enumerate(chunks):
            chunk.metadata = {
                "source": name,
                "file_type": path.suffix.lstrip("."),
                "chunk_index": index,
            }
        all_chunks.extend(chunks)
        documents.append({
            "source": name,
            "chunk_count": len(chunks),
            "sha256": _sha256(path),
        })

    started = time.perf_counter()
    chunk_count = store.add_documents(all_chunks)
    return {
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "ingestion_latency_ms": round(
            (time.perf_counter() - started) * 1000,
            2,
        ),
        "documents": documents,
    }


def build_context(results: Sequence[Dict[str, Any]]) -> str:
    sections = []
    for index, item in enumerate(results, 1):
        source = item.get("metadata", {}).get("source", "未知")
        sections.append(
            f"【参考资料{index}｜来源：{source}】\n{item['content']}"
        )
    return "\n\n".join(sections)


def generate_answer(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    context: str,
    max_tokens: int,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = client.post(
        endpoint,
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"【参考资料】\n{context}\n\n【问题】\n{question}\n\n"
                        "有答案时不要遗漏最后一行的【来源：文件名】。"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"].get("content") or "").strip()


def evaluate_case(
    case: Dict[str, Any],
    *,
    store: VectorStore,
    client: httpx.Client | None,
    base_url: str,
    api_key: str,
    model: str,
    retrieval_strategy: str,
    retrieval_k: int,
    candidate_k: int,
    hit_k: int,
    max_tokens: int,
) -> Dict[str, Any]:
    total_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    if retrieval_strategy == "hybrid":
        retrieval_results = store.hybrid_search(
            case["question"],
            k=retrieval_k,
            candidate_k=candidate_k,
        )
    else:
        retrieval_results = store.similarity_search(
            case["question"],
            k=retrieval_k,
        )
    retrieval_latency_ms = round(
        (time.perf_counter() - retrieval_started) * 1000,
        2,
    )

    expected_source = case.get("expected_source")
    top_sources = [
        str(item.get("metadata", {}).get("source", "未知"))
        for item in retrieval_results
    ]
    source_rank = None
    if expected_source:
        for index, source in enumerate(top_sources[:hit_k], 1):
            if source == expected_source:
                source_rank = index
                break
    context_rank = answer_context_rank(
        retrieval_results,
        expected_source,
        case.get("expected_terms", []),
        hit_k,
    )

    result = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answerable": bool(case["answerable"]),
        "expected_source": expected_source,
        "expected_terms": case.get("expected_terms", []),
        "retrieval_hit": source_rank is not None if expected_source else None,
        "source_rank": source_rank,
        "answer_context_hit": (
            context_rank is not None if expected_source else None
        ),
        "answer_context_rank": context_rank,
        "retrieval_latency_ms": retrieval_latency_ms,
        "retrieved": [
            {
                "rank": index,
                "source": source,
                "score": retrieval_results[index - 1]["score"],
                "content": retrieval_results[index - 1]["content"],
            }
            for index, source in enumerate(top_sources, 1)
        ],
        "generation_attempted": client is not None,
        "answer": "",
        "answer_pass": None,
        "source_cited": None,
        "refusal_pass": None,
        "generation_latency_ms": None,
        "generation_error": None,
        "total_latency_ms": None,
    }
    if client is None:
        return result

    generation_started = time.perf_counter()
    try:
        answer = generate_answer(
            client,
            base_url=base_url,
            api_key=api_key,
            model=model,
            question=case["question"],
            context=build_context(retrieval_results),
            max_tokens=max_tokens,
        )
        result["answer"] = answer
        if case["answerable"]:
            result["answer_pass"] = matches_expected_terms(
                answer,
                case["expected_terms"],
            )
            source_stem = Path(str(expected_source)).stem
            result["source_cited"] = (
                normalize_text(str(expected_source)) in normalize_text(answer)
                or normalize_text(source_stem) in normalize_text(answer)
            )
        else:
            result["refusal_pass"] = is_refusal(answer)
    except Exception as exc:
        result["generation_error"] = f"{type(exc).__name__}: {exc}"

    result["generation_latency_ms"] = round(
        (time.perf_counter() - generation_started) * 1000,
        2,
    )
    result["total_latency_ms"] = round(
        (time.perf_counter() - total_started) * 1000,
        2,
    )
    return result


def render_report(payload: Dict[str, Any]) -> str:
    metadata = payload["metadata"]
    metrics = payload["metrics"]
    hit_key = f"retrieval_hit_at_{metadata['hit_k']}"
    mrr_key = f"retrieval_mrr_at_{metadata['hit_k']}"
    context_hit_key = f"answer_context_hit_at_{metadata['hit_k']}"
    context_mrr_key = f"answer_context_mrr_at_{metadata['hit_k']}"
    if metadata["retrieval_strategy"] == "hybrid":
        retrieval_description = (
            f"`hybrid`，候选 Top-{metadata['candidate_k']}，"
            f"输出 Top-{metadata['retrieval_k']}，"
            f"Hit@{metadata['hit_k']} 评估"
        )
    else:
        retrieval_description = (
            f"`vector`，输出 Top-{metadata['retrieval_k']}，"
            f"Hit@{metadata['hit_k']} 评估"
        )
    generation_ran = metadata["mode"] == "full"
    answer_rate = (
        f"{metrics['answer_keyword_pass_rate']:.1%}"
        if generation_ran else "未运行"
    )
    citation_rate = (
        f"{metrics['source_citation_rate']:.1%}"
        if generation_ran else "未运行"
    )
    refusal_rate = (
        f"{metrics['refusal_accuracy']:.1%}"
        if generation_ran else "未运行"
    )
    total_latency = (
        f"{metrics['total_latency_ms']['p50']:.0f} / "
        f"{metrics['total_latency_ms']['p95']:.0f} ms"
        if generation_ran else "未运行"
    )
    lines = [
        "# KnowledgeOps Agent 评测报告",
        "",
        f"- 运行时间：{metadata['created_at']}",
        f"- Git 基准提交：`{metadata['git_commit']}`"
        f"（运行时工作区{'有' if metadata.get('git_dirty') else '无'}未提交改动）",
        f"- 数据集：`{metadata['dataset_path']}`（SHA256 `{metadata['dataset_sha256']}`）",
        f"- 生成模型：`{metadata['llm_model']}`",
        f"- 嵌入模型：`{metadata['embedding_model']}`",
        f"- 切分参数：`chunk_size={metadata['chunk_size']}`，`chunk_overlap={metadata['chunk_overlap']}`",
        f"- 检索参数：{retrieval_description}",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 评测用例 | {metrics['total_cases']} |",
        f"| 有答案 / 无答案 | {metrics['answerable_cases']} / {metrics['unanswerable_cases']} |",
        f"| Top-{metadata['hit_k']} 来源命中率 | {metrics[hit_key]:.1%} |",
        f"| MRR@{metadata['hit_k']} | {metrics[mrr_key]:.3f} |",
        f"| 答案上下文命中率@{metadata['hit_k']} | {metrics[context_hit_key]:.1%} |",
        f"| 答案上下文 MRR@{metadata['hit_k']} | {metrics[context_mrr_key]:.3f} |",
        f"| 答案关键词通过率 | {answer_rate} |",
        f"| 来源标注率 | {citation_rate} |",
        f"| 无答案拒答准确率 | {refusal_rate} |",
        f"| 生成错误率 | {metrics['generation_error_rate']:.1%} |",
        f"| 检索延迟 P50 / P95 | {metrics['retrieval_latency_ms']['p50']:.0f} / {metrics['retrieval_latency_ms']['p95']:.0f} ms |",
        f"| 端到端延迟 P50 / P95 | {total_latency} |",
        "",
        "## 分类结果",
        "",
        "| 类别 | 用例数 | 来源命中率 | 答案上下文命中率 | 答案通过率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, item in metrics["category_metrics"].items():
        category_answer_rate = (
            f"{item['answer_keyword_pass_rate']:.1%}"
            if generation_ran else "未运行"
        )
        lines.append(
            f"| {category} | {item['case_count']} | "
            f"{item[hit_key]:.1%} | {item[context_hit_key]:.1%} | "
            f"{category_answer_rate} |"
        )

    retrieval_failures = [
        item for item in payload["cases"]
        if item["answerable"] and not item["retrieval_hit"]
    ]
    context_failures = [
        item for item in payload["cases"]
        if item["answerable"] and not item["answer_context_hit"]
    ]
    answer_failures = [
        item for item in payload["cases"]
        if item["answerable"]
        and item["generation_attempted"]
        and not item["answer_pass"]
    ]
    refusal_failures = [
        item for item in payload["cases"]
        if not item["answerable"]
        and item["generation_attempted"]
        and not item["refusal_pass"]
    ]
    lines.extend([
        "",
        "## 失败用例",
        "",
        f"- 来源未命中：{len(retrieval_failures)} 条",
        f"- 答案上下文未命中：{len(context_failures)} 条",
        f"- 答案关键词未通过：{len(answer_failures)} 条",
        f"- 无答案未正确拒答：{len(refusal_failures)} 条",
        "",
    ])
    for title, items in (
        ("来源未命中", retrieval_failures),
        ("答案上下文未命中", context_failures),
        ("答案未通过", answer_failures),
        ("拒答未通过", refusal_failures),
    ):
        if not items:
            continue
        lines.append(f"### {title}")
        lines.append("")
        for item in items:
            top_source = (
                item["retrieved"][0]["source"] if item["retrieved"] else "无"
            )
            lines.append(
                f"- `{item['id']}` {item['question']}（Top-1：{top_source}）"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="evaluation/dataset.json",
        help="评测数据集 JSON 路径",
    )
    parser.add_argument(
        "--documents-root",
        default=".",
        help="五份业务文档所在目录",
    )
    parser.add_argument(
        "--mode",
        choices=("retrieval", "full"),
        default="full",
        help="retrieval 只评估检索；full 同时调用生成模型",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retrieval-k", type=int, default=6)
    parser.add_argument(
        "--retrieval-strategy",
        choices=("vector", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--candidate-k", type=int, default=6)
    parser.add_argument("--hit-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "dashscope"),
        default="ollama",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv(
            "EVAL_EMBEDDING_MODEL",
            "qwen3-embedding:0.6b",
        ),
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv(
            "EVAL_EMBEDDING_BASE_URL",
            "http://127.0.0.1:11434",
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("EVAL_LLM_MODEL", "qwen3:4b-instruct"),
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv(
            "EVAL_LLM_BASE_URL",
            "http://127.0.0.1:11434/v1",
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        default="evaluation/results/latest.json",
    )
    parser.add_argument(
        "--report",
        default="evaluation/results/latest.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.documents_root).resolve()
    dataset_path = Path(args.dataset).resolve()
    dataset = load_dataset(dataset_path)
    cases = dataset["cases"][: args.limit or None]
    embedding_api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or ""
    )
    if args.embedding_provider == "dashscope" and not embedding_api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法运行真实向量检索评测")
    if args.hit_k > args.retrieval_k:
        raise ValueError("hit-k 不能大于 retrieval-k")
    if args.candidate_k < args.retrieval_k:
        raise ValueError("candidate-k 不能小于 retrieval-k")

    llm_api_key = os.getenv("EVAL_LLM_API_KEY", "ollama")
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    embedding_function = None
    if args.embedding_provider == "ollama":
        embedding_function = OllamaEmbeddings(
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            timeout=args.timeout,
        )
    with tempfile.TemporaryDirectory(prefix="knowledgeops-eval-") as tmpdir:
        store = VectorStore(
            api_key=embedding_api_key,
            collection_name="knowledgeops_evaluation",
            persist_dir=tmpdir,
            embedding_function=embedding_function,
        )
        print("正在解析并写入 5 份评测文档...", flush=True)
        ingestion = ingest_documents(
            root,
            store,
            processor,
            DEFAULT_DOCUMENTS,
        )
        print(
            f"入库完成：{ingestion['document_count']} 份文档，"
            f"{ingestion['chunk_count']} 个片段",
            flush=True,
        )

        client = None
        if args.mode == "full":
            is_local = args.llm_base_url.startswith(
                ("http://127.0.0.1", "http://localhost")
            )
            client = httpx.Client(
                trust_env=not is_local,
                timeout=args.timeout,
            )

        results = []
        try:
            for index, case in enumerate(cases, 1):
                item = evaluate_case(
                    case,
                    store=store,
                    client=client,
                    base_url=args.llm_base_url,
                    api_key=llm_api_key,
                    model=args.llm_model,
                    retrieval_strategy=args.retrieval_strategy,
                    retrieval_k=args.retrieval_k,
                    candidate_k=args.candidate_k,
                    hit_k=args.hit_k,
                    max_tokens=args.max_tokens,
                )
                results.append(item)
                status = (
                    "PASS" if (
                        item.get("answer_pass")
                        or item.get("refusal_pass")
                        or (
                            args.mode == "retrieval"
                            and (
                                item.get("answer_context_hit")
                                or not item["answerable"]
                            )
                        )
                    ) else "FAIL"
                )
                print(
                    f"[{index:02d}/{len(cases)}] {item['id']} {status} "
                    f"retrieval={item['retrieval_latency_ms']:.0f}ms "
                    f"generation={item.get('generation_latency_ms') or 0:.0f}ms",
                    flush=True,
                )
        finally:
            if client is not None:
                client.close()
            store.close()

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "git_dirty": _git_dirty(root),
        "dataset_path": str(dataset_path.relative_to(root)),
        "dataset_sha256": _sha256(dataset_path),
        "llm_model": args.llm_model if args.mode == "full" else "not-run",
        "llm_base_url": args.llm_base_url if args.mode == "full" else "not-run",
        "embedding_provider": args.embedding_provider,
        "embedding_model": (
            args.embedding_model
            if args.embedding_provider == "ollama"
            else "text-embedding-v2"
        ),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "retrieval_strategy": args.retrieval_strategy,
        "retrieval_k": args.retrieval_k,
        "candidate_k": args.candidate_k,
        "hit_k": args.hit_k,
        "mode": args.mode,
        "ingestion": ingestion,
    }
    payload = {
        "metadata": metadata,
        "metrics": summarize_results(results, args.hit_k),
        "cases": results,
    }
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(f"原始结果：{output_path}", flush=True)
    print(f"评测报告：{report_path}", flush=True)
    print(
        json.dumps(payload["metrics"], ensure_ascii=False, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
