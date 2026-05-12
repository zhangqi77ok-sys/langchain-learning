"""基于本地文本文件的最小知识库问答示例。"""

import os
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


def load_knowledge_base(file_path: Path) -> list[Document]:
    """读取本地知识库文本并切分成多个文档块。"""
    content = file_path.read_text(encoding="utf-8")
    document = Document(
        page_content=content,
        metadata={"source": str(file_path.name)},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )
    return splitter.split_documents([document])


def build_context(documents: list[Document]) -> str:
    """把召回到的文档块拼成回答上下文。"""
    return "\n\n".join(document.page_content for document in documents)


def main() -> None:
    """执行一次本地知识库问答。"""
    load_dotenv()

    file_path = Path(__file__).resolve().parent.parent / "notes" / "langchain_intro.txt"
    documents = load_knowledge_base(file_path)

    embeddings = LocalEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
    """ 改 question 为 Tool Calling 的核心思想是什么？
        改 question 为 Memory 是怎么实现的？"""
    question = input("请输入你的问题：").strip()
    if not question:
        raise ValueError("问题不能为空")
    retrieved_docs = vector_store.similarity_search(question, k=3)
    context = build_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名简洁清晰的知识库助手。请严格依据上下文回答，不要编造上下文里没有的信息。",
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
    print("question:")
    print(question)
    print("-" * 20)
    print("retrieved context:")
    print(context)
    print("-" * 20)
    print("answer:")
    # 这里改成流式输出，目的是让你看到知识库回答如何边生成边展示。
    for chunk in chain.stream({"context": context, "question": question}):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
