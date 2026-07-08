"""
文档处理层 —— 文件加载 / 文本清洗 / 内容切片
支持 TXT、PDF、Markdown 三种格式，输出 LangChain Document 对象列表
"""
import os
import re
from typing import List, Dict, Optional
from pypdf import PdfReader
from langchain_core.documents import Document


class DocumentProcessor:
    """企业文档处理器
    负责从原始文件中提取文本、清洗噪声、按语义边界切分为 LangChain Document 列表。
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── 文件加载 ──────────────────────────────

    def load_file(self, file_path: str) -> Optional[str]:
        """根据文件扩展名分发至对应加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"暂不支持的文件格式: {ext}，当前仅支持 {self.SUPPORTED_EXTENSIONS}"
            )
        if ext == ".txt":
            return self._load_txt(file_path)
        elif ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".md":
            return self._load_md(file_path)

    @staticmethod
    def _load_txt(file_path: str) -> str:
        """加载 TXT 文件，自动检测编码"""
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise UnicodeDecodeError(f"无法以任何已知编码读取文件: {file_path}")

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """加载 PDF 文件，提取所有页面文本"""
        reader = PdfReader(file_path)
        pages = [page.extract_text() for page in reader.pages]
        return "\n\n".join(p for p in pages if p)

    @staticmethod
    def _load_md(file_path: str) -> str:
        """加载 Markdown 文件"""
        return DocumentProcessor._load_txt(file_path)

    # ── 文本清洗 ──────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        """对原始文本执行基础清洗"""
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"[ \t]{3,}", "  ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    # ── 文本切片 ──────────────────────────────

    def split_text(
        self, text: str, metadata: Optional[Dict] = None
    ) -> List[Document]:
        """将长文本切分为 LangChain Document 列表

        策略：段落优先 → 句子兜底 → 固定窗口强制
        """
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = (
                    (current_chunk + "\n\n" + para).strip()
                    if current_chunk else para
                )
            else:
                if len(para) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    chunks.extend(self._force_split(para))
                else:
                    chunks.append(current_chunk)
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # 构建 LangChain Document 列表
        meta = metadata or {}
        return [
            Document(page_content=c, metadata=dict(meta))
            for c in chunks if c.strip()
        ]

    def _force_split(self, text: str) -> List[str]:
        """对超长段落按句子边界 + 滑动窗口强制切分"""
        sentences = re.split(r"(?<=[。！？；\n])\s*", text)
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                if len(sent) > self.chunk_size:
                    step = self.chunk_size - self.chunk_overlap
                    for i in range(0, len(sent), step):
                        chunks.append(sent[i:i + self.chunk_size])
                else:
                    current = sent
        if current:
            chunks.append(current)
        return chunks

    # ── 批量处理入口 ──────────────────────────

    def process_file(self, file_path: str) -> List[Document]:
        """完整处理流水线：加载 → 清洗 → 切片 → List[Document]"""
        raw = self.load_file(file_path)
        cleaned = self.clean_text(raw)
        file_name = os.path.basename(file_path)
        return self.split_text(cleaned, metadata={
            "source": file_name,
            "file_path": file_path,
        })
