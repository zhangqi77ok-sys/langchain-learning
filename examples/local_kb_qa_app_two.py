"""基于多文件知识库目录的最小问答示例。"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def load_knowledge_base(directory_path: Path) -> list[Document]:
    """读取知识库目录下的多个文本文件。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=40,
    )
    documents: list[Document] = []

    for file_path in sorted(directory_path.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        document = Document(
            page_content=content,
            metadata={"source": file_path.name},
        )
        documents.extend(splitter.split_documents([document]))

    return documents


def build_context(documents: list[Document]) -> str:
    """把检索到的文档块拼成上下文字符串。"""
    parts: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        parts.append(f"[{source}]\n{document.page_content}")
    return "\n\n".join(parts)


def main() -> None:
    """执行一次多文件知识库问答。"""
    load_dotenv()

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入你的问题：").strip()
    if not question:
        raise ValueError("问题不能为空")

    kb_path = Path(__file__).resolve().parent.parent / "notes" / "kb"
    print("step 1: loading knowledge base...")
    documents = load_knowledge_base(kb_path)
    print(f"loaded chunks: {len(documents)}")

    print("step 2: loading local embeddings model...")
    embeddings = LocalEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    print("local embeddings model ready")
    vector_store = InMemoryVectorStore(embeddings)
    print("step 3: adding documents to vector store...")
    vector_store.add_documents(documents)
    print("documents added to vector store")

    print("step 4: running similarity search...")
    retrieved_docs = vector_store.similarity_search(question, k=4)
    print(f"retrieved docs: {len(retrieved_docs)}")
    context = build_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名简洁清晰的知识库助手。请严格依据上下文回答，并尽量指出信息来自哪个文件。",
            ),
            ("human", "上下文：\n{context}\n\n问题：\n{question}"),
        ]
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    chain = prompt | model

    print("step 5: generating final answer...")
    response = chain.invoke({"context": context, "question": question})
    print("question:")
    print(question)
    print("-" * 20)
    print("retrieved context:")
    print(context)
    print("-" * 20)
    print("answer:")
    print(response.content)


if __name__ == "__main__":
    main()
