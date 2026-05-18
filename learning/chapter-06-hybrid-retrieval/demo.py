"""Chapter 06: 混合检索最小示例。"""

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def load_documents() -> list[str]:
    """读取本章的本地文档，并按空行切成多条文档。"""
    data_path = Path(__file__).resolve().parent / "data" / "knowledge.txt"
    text = data_path.read_text(encoding="utf-8")
    return [item.strip() for item in text.split("\n\n") if item.strip()]


def tokenize(text: str) -> list[str]:
    """把文本切成适合 BM25 的最小词元。"""
    cleaned = text.replace("，", " ").replace("。", " ").replace("、", " ")
    return [token for token in cleaned.split() if token]


def normalize_scores(scores: list[float]) -> list[float]:
    """把不同检索器的分数拉到同一范围，方便后续融合。"""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def run_bm25(query: str, documents: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """执行一次 BM25 检索。"""
    corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[index], float(scores[index])) for index in top_indices]


def run_vector_search(query: str, documents: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """执行一次向量检索。"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_vectors = np.asarray(model.encode(documents, normalize_embeddings=True))
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[index], float(scores[index])) for index in top_indices]


def fuse_results(
    bm25_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """把 BM25 和向量检索的结果做最小融合。"""
    bm25_docs = [doc for doc, _ in bm25_results]
    bm25_scores = [score for _, score in bm25_results]
    vector_docs = [doc for doc, _ in vector_results]
    vector_scores = [score for _, score in vector_results]

    normalized_bm25 = dict(zip(bm25_docs, normalize_scores(bm25_scores)))
    normalized_vector = dict(zip(vector_docs, normalize_scores(vector_scores)))

    merged_docs = []
    for doc in bm25_docs + vector_docs:
        if doc not in merged_docs:
            merged_docs.append(doc)

    fused = []
    for doc in merged_docs:
        score = (
            normalized_bm25.get(doc, 0.0) * bm25_weight
            + normalized_vector.get(doc, 0.0) * vector_weight
        )
        fused.append((doc, score))

    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def print_results(title: str, results: list[tuple[str, float]]) -> None:
    """打印检索结果，方便逐步对比。"""
    print(title)
    for index, (doc, score) in enumerate(results, start=1):
        print(f"{index}. score={score:.4f}")
        print(doc)
        print("-" * 20)
    print("=" * 40)


def main() -> None:
    """依次演示 BM25、向量检索和融合结果。"""
    query = "为什么混合检索比单一路径更稳"
    documents = load_documents()

    print("第 1 步：读取文档")
    for index, doc in enumerate(documents, start=1):
        print(f"文档 {index}: {doc}")
    print("=" * 40)

    print(f"第 2 步：准备查询\n查询：{query}")
    print("=" * 40)

    bm25_results = run_bm25(query, documents, top_k=3)
    print_results("第 3 步：BM25 检索结果", bm25_results)

    vector_results = run_vector_search(query, documents, top_k=3)
    print_results("第 4 步：向量检索结果", vector_results)

    fused_results = fuse_results(bm25_results, vector_results)
    print_results("第 5 步：融合后的结果", fused_results)

    print("结论：")
    print("混合检索的价值，是让关键词匹配和语义匹配互相补短。")


if __name__ == "__main__":
    main()
