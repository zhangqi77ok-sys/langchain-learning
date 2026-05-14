"""基础巩固 07：向量检索最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    """用本地 SentenceTransformer 生成 embedding。

    这里继续沿用你已经跑通过的本地 embedding 路线，
    是为了让重点放在“检索逻辑”本身，而不是第三方 embedding 接口兼容问题。
    """

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
    """演示最小语义检索流程。

    这一组的重点是：
    还没有生成最终答案，
    只是先把“最相关的资料块”找出来。
    """
    # 先加载 .env。
    # 这里主要是为了拿 LOCAL_EMBEDDING_MODEL。
    load_dotenv()

    # 先手工准备几段文档。
    # 在真实项目里，这些内容通常来自文件、数据库或知识库。
    documents = [
        Document(
            page_content="LangChain 可以帮助开发者组织提示词、模型调用和工作流。",
            metadata={"id": 1, "topic": "langchain"},
        ),
        Document(
            page_content="RAG 的核心是先检索相关资料，再让模型基于资料生成答案。",
            metadata={"id": 2, "topic": "rag"},
        ),
        Document(
            page_content="Tool Calling 的核心是让模型在回答前决定是否调用外部工具。",
            metadata={"id": 3, "topic": "tool_calling"},
        ),
    ]

    # 创建本地 embedding 模型。
    # 这里的作用不是生成答案，而是把文本映射成可比较相似度的向量。
    embeddings = LocalEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # 创建内存向量存储。
    # 这里适合学习和最小示例，不适合大规模正式数据。
    vector_store = InMemoryVectorStore(embeddings)

    # 先把文档加入向量存储。
    # 本质上就是先让文档完成向量化，并进入可检索状态。
    vector_store.add_documents(documents)

    # 这里定义一个查询问题。
    # 注意：这还不是最终回答给用户的结果，
    # 而只是“拿这个问题去找相关文档”。
    query = "RAG 的基本思路是什么？"

    # k=2 的意思是：返回最相似的前 2 条结果。
    # 这就是你之前问过的 top-k 检索。
    results = vector_store.similarity_search(query, k=2)

    print("检索结果：")
    for index, document in enumerate(results, start=1):
        # 这里逐条打印，目的是让你看清：
        # “检索”阶段的输出不是最终答案，而是候选资料。
        print(f"result {index}")
        print(f"content: {document.page_content}")
        print(f"metadata: {document.metadata}")
        print("-" * 20)


if __name__ == "__main__":
    main()
