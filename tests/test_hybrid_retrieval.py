from vectordb.store import VectorStore


def test_bm25_rrf_promotes_exact_answer_from_vector_candidates():
    candidates = [
        {
            "content": "设备支持远程查看与云存储。",
            "metadata": {"source": "产品手册.pdf"},
            "score": 0.91,
        },
        {
            "content": "支持多人分享设备和权限管理。",
            "metadata": {"source": "产品手册.pdf"},
            "score": 0.89,
        },
        {
            "content": "同一账号最多允许3台设备同时在线观看，"
            "单个设备最多支持5个用户同时访问。",
            "metadata": {"source": "产品手册.pdf"},
            "score": 0.72,
        },
    ]
    store = object.__new__(VectorStore)
    store.similarity_search = lambda query, k: candidates[:k]

    results = store.hybrid_search(
        "同一账号和单个设备分别允许多少人并发访问？",
        k=2,
        candidate_k=3,
    )

    promoted = next(
        item for item in results
        if "3台设备" in item["content"]
    )
    assert promoted["vector_rank"] == 3
    assert promoted["lexical_rank"] == 1
    assert promoted["retrieval_strategy"] == "bm25_rrf"


def test_hybrid_search_keeps_empty_result_contract():
    store = object.__new__(VectorStore)
    store.similarity_search = lambda query, k: []

    assert store.hybrid_search("任意问题", k=3, candidate_k=6) == []
