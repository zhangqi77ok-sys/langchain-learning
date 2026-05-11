"""LangChain 最小 RAG 问答示例。"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
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


def build_context(documents: list[Document]) -> str:
    """把检索到的文档片段拼成上下文字符串。"""
    return "\n\n".join(document.page_content for document in documents)


def main() -> None:
    """先检索资料，再基于资料回答问题。"""
    load_dotenv()

    documents = [
        Document(
            page_content="LangChain 可以帮助开发者组织提示词、模型调用和工作流。",
            metadata={"id": 1, "topic": "langchain"},
        ),
        Document(
            page_content="RAG 的常见流程是先检索相关资料，再把资料放进提示词让大模型回答。",
            metadata={"id": 2, "topic": "rag"},
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

    question = "LangChain 怎么帮助构建复杂的大模型应用？"
    retrieved_docs = vector_store.similarity_search(question, k=2)
    context = build_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名简洁清晰的 AI 助手。请严格基于提供的上下文回答，不要编造上下文中没有的信息。",
            ),
            (
                "human",
                "上下文：\n{context}\n\n问题：\n{question}",
            ),
        ]
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    chain = prompt | model

    # 这里把检索结果塞进提示词，目的是让你看清 RAG 回答链的最小结构。
    response = chain.invoke({"context": context, "question": question})
    print("retrieved context:")
    print(context)
    print("-" * 20)
    print("answer:")
    print(response.content)


if __name__ == "__main__":
    main()
