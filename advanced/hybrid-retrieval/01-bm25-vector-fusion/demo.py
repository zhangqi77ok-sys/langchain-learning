"""混合检索最小示例：BM25 + 向量检索融合。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Document:
    """最小文档结构。"""

    doc_id: str
    content: str


def tokenize(text: str) -> list[str]:
    """把文本切成适合 BM25 的最小词元。"""
    return [token for token in text.lower().replace("，", " ").replace("。", " ").split() if token]


def min_max_normalize(scores: list[float]) -> list[float]:
    """把不同检索器的分数拉到同一尺度，方便融合。"""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def build_documents() -> list[Document]:
    """准备一小组本地演示文档。"""
    return [
        Document(
            doc_id="doc-1",
            content="BM25 适合关键词明确的检索，尤其是专有名词和精确短语。",
        ),
        Document(
            doc_id="doc-2",
            content="向量检索更擅长语义相似，适合用户说法和文档措辞不同的情况。",
        ),
        Document(
            doc_id="doc-3",
            content="混合检索通常会把 BM25 和向量检索的结果融合起来，提高召回稳定性。",
        ),
        Document(
            doc_id="doc-4",
            content="RAG 系统里常见的做法是先检索，再把候选资料交给生成模型。",
        ),
        Document(
            doc_id="doc-5",
            content="如果查询里出现明确关键词，BM25 的优势通常会更明显。",
        ),
        Document(
            doc_id="doc-6",
            content="如果问题表达比较口语化，向量模型往往更容易找到语义接近的内容。",
        ),
    ]


def build_bm25_index(documents: list[Document]) -> BM25Okapi:
    """为文档集构建 BM25 索引。"""
    tokenized_corpus = [tokenize(doc.content) for doc in documents]
    return BM25Okapi(tokenized_corpus)


def build_vector_index(documents: list[Document], model: SentenceTransformer) -> np.ndarray:
    """把文档编码成向量，后续做语义相似度检索。"""
    embeddings = model.encode([doc.content for doc in documents], normalize_embeddings=True)
    return np.asarray(embeddings)


def retrieve_bm25(query: str, documents: list[Document], bm25: BM25Okapi, top_k: int) -> list[tuple[Document, float]]:
    """按 BM25 分数召回候选文档。"""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[index], float(scores[index])) for index in ranked_indices]


def retrieve_vector(query: str, documents: list[Document], model: SentenceTransformer, doc_embeddings: np.ndarray, top_k: int) -> list[tuple[Document, float]]:
    """按向量余弦相似度召回候选文档。"""
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = doc_embeddings @ query_embedding
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return [(documents[index], float(similarities[index])) for index in ranked_indices]


def fuse_results(
    bm25_results: list[tuple[Document, float]],
    vector_results: list[tuple[Document, float]],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[tuple[Document, float]]:
    """把两路召回结果归一化后合并成一个最终排序。"""
    combined_docs: dict[str, Document] = {}
    bm25_score_map = {doc.doc_id: score for doc, score in bm25_results}
    vector_score_map = {doc.doc_id: score for doc, score in vector_results}

    for doc, _ in bm25_results + vector_results:
        combined_docs[doc.doc_id] = doc

    normalized_bm25 = dict(zip(bm25_score_map.keys(), min_max_normalize(list(bm25_score_map.values()))))
    normalized_vector = dict(zip(vector_score_map.keys(), min_max_normalize(list(vector_score_map.values()))))

    fused_scores: list[tuple[Document, float]] = []
    for doc_id, doc in combined_docs.items():
        bm25_score = normalized_bm25.get(doc_id, 0.0)
        vector_score = normalized_vector.get(doc_id, 0.0)
        fused_score = bm25_weight * bm25_score + vector_weight * vector_score
        fused_scores.append((doc, fused_score))

    fused_scores.sort(key=lambda item: item[1], reverse=True)
    return fused_scores


def print_results(title: str, results: Iterable[tuple[Document, float]]) -> None:
    """打印检索结果，便于对比两路召回和最终融合。"""
    print(title)
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. {doc.doc_id} | score={score:.4f} | {doc.content}")
    print("-" * 60)


def main() -> None:
    """运行混合检索最小 demo。"""
    documents = build_documents()
    query = "混合检索为什么比单独检索更稳"

    model_path = Path.home() / ".cache" / "sentence-transformers"
    _ = model_path  # 这里保留一个明确位置，方便你以后知道模型缓存会落在哪里。
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    bm25 = build_bm25_index(documents)
    doc_embeddings = build_vector_index(documents, model)

    bm25_results = retrieve_bm25(query, documents, bm25, top_k=3)
    vector_results = retrieve_vector(query, documents, model, doc_embeddings, top_k=3)
    fused_results = fuse_results(bm25_results, vector_results)

    print(f"查询: {query}")
    print("=" * 60)
    print_results("BM25 召回结果", bm25_results)
    print_results("向量召回结果", vector_results)
    print_results("融合后结果", fused_results)


if __name__ == "__main__":
    main()
