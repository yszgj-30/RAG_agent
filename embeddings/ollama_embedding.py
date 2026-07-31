"""Ollama text embeddings adapted to the LangChain Embeddings interface."""

from __future__ import annotations

from typing import List, Optional

import httpx
from langchain_core.embeddings import Embeddings


class OllamaEmbeddings(Embeddings):
    """Generate local embeddings through Ollama's native ``/api/embed`` API."""

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://127.0.0.1:11434",
        batch_size: int = 20,
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            trust_env=False,
            timeout=timeout,
        )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self._client.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
            },
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Ollama 返回的向量数量与输入文本数量不一致"
            )
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [str(text).strip() for text in texts]
        all_embeddings: List[List[float]] = []
        for index in range(0, len(clean_texts), self.batch_size):
            all_embeddings.extend(
                self._embed_batch(clean_texts[index:index + self.batch_size])
            )
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([str(text).strip()])[0]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
