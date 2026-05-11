"""LangChain 向量检索最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


class LocalEmbeddings:
    """用本地 SentenceTransformer 生成 embedding。"""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把多个文本转换成向量。"""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """把查询文本转换成向量。"""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def main() -> None:
    """把文档向量化后做一次相似度检索。"""
    load_dotenv()

    documents = [
        Document(
            page_content="LangChain 可以帮助开发者组织提示词、模型调用和工作流。",
            metadata={"id": 1, "topic": "langchain"},
        ),
        Document(
            page_content="向量检索的核心是把文本转换成 embedding，再按相似度找相关内容。",
            metadata={"id": 2, "topic": "retrieval"},
        ),
        Document(
            page_content="结构化输出适合把模型结果转换成固定字段，便于程序继续处理。",
            metadata={"id": 3, "topic": "structured_output"},
        ),
    ]

    embeddings = LocalEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    # 这里直接做相似度检索，目的是让你看清 RAG 在回答前先“找资料”。
    results = vector_store.similarity_search("LangChain 怎么帮助组织复杂的大模型应用？", k=2)

    for index, document in enumerate(results, start=1):
        print(f"result {index}")
        print(f"content: {document.page_content}")
        print(f"metadata: {document.metadata}")
        print("-" * 20)


if __name__ == "__main__":
    main()
