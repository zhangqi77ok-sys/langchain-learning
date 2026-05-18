"""Chapter 03: 最小 RAG 闭环示例。"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def get_required_env(name: str) -> str:
    """读取必须存在的环境变量。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def load_knowledge() -> str:
    """读取本章的本地知识文本。"""
    data_path = Path(__file__).resolve().parent / "data" / "knowledge.txt"
    return data_path.read_text(encoding="utf-8")


def build_chunks() -> list[Document]:
    """把长文本切成更适合检索的小片段。"""
    text = load_knowledge()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=20,
    )
    return splitter.create_documents([text])


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """把文本编码成向量，方便后续做相似度检索。"""
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings)


def retrieve_top_k(query: str, chunks: list[Document], model: SentenceTransformer, top_k: int = 2) -> list[Document]:
    """从切分后的知识块中找出最相关的片段。"""
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_vectors = embed_texts(model, chunk_texts)
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = chunk_vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[index] for index in top_indices]


def build_prompt(question: str, context: str) -> str:
    """把检索结果拼进最终提示词。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个严谨的中文助手，只能依据给定资料回答。"),
            (
                "human",
                "资料如下：\n{context}\n\n问题：{question}\n请基于资料回答，尽量简洁。",
            ),
        ]
    )
    return prompt.format(context=context, question=question)


def main() -> None:
    """运行一次最小 RAG 闭环。"""
    root_env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env_path)

    question = "什么是 RAG，为什么要先检索再生成？"
    embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer(embedding_model_name)

    print("第 1 步：读取本地知识文件")
    chunks = build_chunks()
    print(f"切分后得到 {len(chunks)} 个片段")
    for index, chunk in enumerate(chunks, start=1):
        print(f"片段 {index}: {chunk.page_content}")
    print("-" * 40)

    print("第 2 步：对问题做检索")
    top_chunks = retrieve_top_k(question, chunks, embedding_model, top_k=2)
    context = "\n\n".join(chunk.page_content for chunk in top_chunks)
    print("召回到的片段：")
    for index, chunk in enumerate(top_chunks, start=1):
        print(f"{index}. {chunk.page_content}")
    print("-" * 40)

    print("第 3 步：把检索结果拼进提示词")
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    prompt = build_prompt(question, context)
    print("最终提示词：")
    print(prompt)
    print("-" * 40)

    print("第 4 步：把提示词交给模型")
    response = model.invoke(prompt)

    print("问题：")
    print(question)
    print("-" * 40)
    print("检索到的上下文：")
    print(context)
    print("-" * 40)
    print("模型回答：")
    print(response.content)


if __name__ == "__main__":
    main()
