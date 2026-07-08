"""
通义千问 DashScope 文本嵌入模型 —— LangChain Embeddings 接口实现
完全基于 DashScope 原生 API，不依赖 langchain_openai
"""
from typing import List
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings


class DashScopeEmbeddings(Embeddings):
    """通义千问 DashScope 文本嵌入模型

    继承 LangChain Embeddings 基类，实现 embed_documents / embed_query，
    可直接作为 langchain_chroma.Chroma 的 embedding_function 参数。

    API 规格:
        - 模型: text-embedding-v1（1536维）
        - 单次最大输入: 25 条文本
        - 支持 text_type 参数（document / query）以优化检索精度
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v2",
        text_type: str = "document",
        batch_size: int = 20,
    ):
        """
        Args:
            api_key: DashScope API 密钥
            model: 嵌入模型名称，默认 text-embedding-v1
            text_type: 文本类型，document 或 query
            batch_size: 单次 API 调用最大文本数（上限 25）
        """
        self.api_key = api_key
        self.model = model
        self.text_type = text_type
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入向量（LangChain 标准接口）

        Args:
            texts: 纯文本字符串列表 List[str]

        Returns:
            按输入顺序排列的嵌入向量列表 List[List[float]]
        """
        if not texts:
            return []

        # 强制确保每个元素都是纯字符串
        clean_texts: List[str] = []
        for t in texts:
            if not isinstance(t, str):
                t = str(t)
            clean_texts.append(t.strip())

        all_embeddings: List[List[float]] = []

        # 分批调用 DashScope TextEmbedding API
        for i in range(0, len(clean_texts), self.batch_size):
            batch = clean_texts[i:i + self.batch_size]

            response = TextEmbedding.call(
                model=self.model,
                input=batch,
                api_key=self.api_key,
                text_type="document",
            )

            if response["status_code"] != 200:
                raise RuntimeError(
                    f"DashScope 嵌入生成失败: "
                    f"code={response['status_code']}, message={response['message']}"
                )

            # 按 text_index 排序，保证输出顺序与输入一致
            emb_items = response["output"]["embeddings"]
            sorted_items = sorted(
                emb_items,
                key=lambda item: item["text_index"],
            )
            for item in sorted_items:
                all_embeddings.append(item["embedding"])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """生成单条查询文本的嵌入向量（LangChain 标准接口）

        Args:
            text: 查询文本字符串

        Returns:
            嵌入向量 List[float]
        """
        if not isinstance(text, str):
            text = str(text)

        response = TextEmbedding.call(
            model=self.model,
            input=text.strip(),
            api_key=self.api_key,
            text_type="query",
        )

        if response["status_code"] != 200:
            raise RuntimeError(
                f"DashScope 查询嵌入失败: "
                f"code={response['status_code']}, message={response['message']}"
            )

        emb = response["output"]["embeddings"][0]
        return emb["embedding"]
