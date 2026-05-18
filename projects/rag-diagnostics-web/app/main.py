"""rag-diagnostics-web 的教学型 FastAPI 入口。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


app = FastAPI(title="rag-diagnostics-web")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
load_dotenv(Path(__file__).resolve().parents[3] / ".env")
REJECTION_MESSAGE = "当前知识库里没有找到和这个问题足够相关的内容。"


MENU_ITEMS = [
    {
        "key": "langchain-basics",
        "title": "LangChain 基础",
        "url": "/menus/langchain-basics",
        "source": "langchain_basics.txt",
    },
    {
        "key": "rag-basics",
        "title": "RAG 基础",
        "url": "/menus/rag-basics",
        "source": "rag_basics.txt",
    },
    {
        "key": "tool-agent-basics",
        "title": "Tool / Agent 基础",
        "url": "/menus/tool-agent-basics",
        "source": "tool_agent_basics.txt",
    },
]

MENU_BY_KEY = {item["key"]: item for item in MENU_ITEMS}
MENU_BY_SOURCE = {item["source"]: item for item in MENU_ITEMS}


class AskRequest(BaseModel):
    """前端问答请求体。"""

    question: str


class RecommendedMenu(BaseModel):
    """推荐菜单信息。"""

    key: str
    title: str
    url: str


class RetrievedChunk(BaseModel):
    """单个检索片段的诊断信息。"""

    source: str
    content: str
    score: float


class Diagnostics(BaseModel):
    """页面需要展示的 RAG 诊断信息。"""

    rejected: bool
    best_score: float
    threshold: float
    bm25_results: list[RetrievedChunk]
    vector_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    final_context: str


class AskResponse(BaseModel):
    """问答接口响应体。"""

    question: str
    answer: str
    sources: list[str]
    recommended_menu: RecommendedMenu | None
    recommended_reason: str | None
    diagnostics: Diagnostics


def get_required_env(name: str) -> str:
    """读取必须存在的环境变量。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


class LocalEmbeddings:
    """使用本地 sentence-transformers 模型生成向量。"""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把多个文本转成向量。"""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """把查询文本转成向量。"""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def load_knowledge_base(directory_path: Path) -> list[Document]:
    """读取本地知识文件，并切分成适合检索的小片段。"""
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


def tokenize(text: str) -> list[str]:
    """把文本切成适合 BM25 的最小词元。"""
    cleaned = text.replace("，", " ").replace("。", " ").replace("、", " ")
    return [token for token in cleaned.split() if token]


def normalize_scores(scores: list[float]) -> list[float]:
    """把分数拉到 0 到 1，方便后续融合。"""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def run_bm25_search(question: str, documents: list[Document], top_k: int) -> list[tuple[Document, float]]:
    """执行一次 BM25 检索。"""
    corpus = [tokenize(document.page_content) for document in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(question))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]
    return [(documents[index], float(scores[index])) for index in ranked_indices]


def run_vector_search(
    question: str,
    vector_store: InMemoryVectorStore,
    top_k: int,
) -> list[tuple[Document, float]]:
    """执行一次向量检索，并保留相似度分数。"""
    results = vector_store.similarity_search_with_score(question, k=top_k)
    return [(document, float(score)) for document, score in results]


def fuse_results(
    bm25_results: list[tuple[Document, float]],
    vector_results: list[tuple[Document, float]],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[tuple[Document, float]]:
    """把 BM25 和向量检索结果做最小融合。"""
    bm25_docs = [document for document, _ in bm25_results]
    bm25_scores = [score for _, score in bm25_results]
    vector_docs = [document for document, _ in vector_results]
    vector_scores = [score for _, score in vector_results]

    normalized_bm25 = dict(
        zip((document.page_content for document in bm25_docs), normalize_scores(bm25_scores))
    )
    normalized_vector = dict(
        zip((document.page_content for document in vector_docs), normalize_scores(vector_scores))
    )

    merged_docs: list[Document] = []
    seen_contents: set[str] = set()
    for document in bm25_docs + vector_docs:
        if document.page_content not in seen_contents:
            merged_docs.append(document)
            seen_contents.add(document.page_content)

    fused: list[tuple[Document, float]] = []
    for document in merged_docs:
        score = (
            normalized_bm25.get(document.page_content, 0.0) * bm25_weight
            + normalized_vector.get(document.page_content, 0.0) * vector_weight
        )
        fused.append((document, score))

    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def build_context(documents: list[Document]) -> str:
    """把最终送给模型的片段拼成上下文。"""
    parts: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        parts.append(f"[{source}]\n{document.page_content}")
    return "\n\n".join(parts)


def convert_to_chunks(results: list[tuple[Document, float]]) -> list[RetrievedChunk]:
    """把内部检索结果转换成前端可展示的结构。"""
    chunks: list[RetrievedChunk] = []
    for document, score in results:
        chunks.append(
            RetrievedChunk(
                source=document.metadata.get("source", "unknown"),
                content=document.page_content,
                score=score,
            )
        )
    return chunks


def choose_recommended_menu(question: str, sources: list[str]) -> RecommendedMenu:
    """按最小规则决定推荐哪个菜单。"""
    for source in sources:
        if source in MENU_BY_SOURCE:
            item = MENU_BY_SOURCE[source]
            return RecommendedMenu(key=item["key"], title=item["title"], url=item["url"])

    lowered_question = question.lower()
    if "rag" in lowered_question:
        item = MENU_BY_KEY["rag-basics"]
    elif "agent" in lowered_question or "tool" in lowered_question:
        item = MENU_BY_KEY["tool-agent-basics"]
    else:
        item = MENU_BY_KEY["langchain-basics"]

    return RecommendedMenu(key=item["key"], title=item["title"], url=item["url"])


def build_recommended_reason(question: str, sources: list[str], recommended_menu: RecommendedMenu) -> str:
    """解释为什么推荐这个菜单。"""
    if sources:
        first_source = sources[0]
        return f"因为这次回答主要命中了 {first_source}，所以推荐你继续查看 {recommended_menu.title}。"

    return f"因为当前没有命中足够相关的知识内容，所以先推荐你去 {recommended_menu.title} 看基础内容。"


@app.get("/")
def read_root(request: Request):
    """返回首页。"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "rag-diagnostics-web",
            "menus": MENU_ITEMS,
        },
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """执行问答，并把内部诊断信息一并返回给前端。"""
    question = payload.question.strip()
    if not question:
        empty_diagnostics = Diagnostics(
            rejected=False,
            best_score=0.0,
            threshold=0.35,
            bm25_results=[],
            vector_results=[],
            fused_results=[],
            final_context="",
        )
        return AskResponse(
            question="",
            answer="问题不能为空。",
            sources=[],
            recommended_menu=None,
            recommended_reason=None,
            diagnostics=empty_diagnostics,
        )

    kb_path = Path(__file__).resolve().parent.parent / "data"
    documents = load_knowledge_base(kb_path)
    embeddings = LocalEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    bm25_results = run_bm25_search(question, documents, top_k=4)
    vector_results = run_vector_search(question, vector_store, top_k=4)
    fused_results = fuse_results(bm25_results, vector_results)[:4]
    best_score = fused_results[0][1] if fused_results else 0.0
    threshold = 0.35

    retrieved_docs = [document for document, _ in fused_results]
    context = build_context(retrieved_docs)
    sources = list(
        dict.fromkeys(document.metadata.get("source", "unknown") for document in retrieved_docs)
    )

    if best_score < threshold:
        recommended_menu = choose_recommended_menu(question, [])
        diagnostics = Diagnostics(
            rejected=True,
            best_score=best_score,
            threshold=threshold,
            bm25_results=convert_to_chunks(bm25_results),
            vector_results=convert_to_chunks(vector_results),
            fused_results=convert_to_chunks(fused_results),
            final_context=context,
        )
        return AskResponse(
            question=question,
            answer=REJECTION_MESSAGE,
            sources=sources,
            recommended_menu=recommended_menu,
            recommended_reason=build_recommended_reason(question, [], recommended_menu),
            diagnostics=diagnostics,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名简洁清晰的知识库助手。请严格依据上下文回答，并尽量指出信息来自哪个文件。"
                "如果上下文和问题明显无关，或者上下文不足以支持回答，"
                "必须直接回答：当前知识库里没有找到和这个问题足够相关的内容。"
                "不要勉强回答，不要扩展到上下文之外的知识。",
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
    response = chain.invoke({"context": context, "question": question})
    answer_text = response.content
    rejected = answer_text.strip() == REJECTION_MESSAGE

    if rejected:
        recommended_menu = choose_recommended_menu(question, [])
        diagnostics = Diagnostics(
            rejected=True,
            best_score=best_score,
            threshold=threshold,
            bm25_results=convert_to_chunks(bm25_results),
            vector_results=convert_to_chunks(vector_results),
            fused_results=convert_to_chunks(fused_results),
            final_context=context,
        )
        return AskResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            recommended_menu=recommended_menu,
            recommended_reason=build_recommended_reason(question, [], recommended_menu),
            diagnostics=diagnostics,
        )

    recommended_menu = choose_recommended_menu(question, sources)
    diagnostics = Diagnostics(
        rejected=False,
        best_score=best_score,
        threshold=threshold,
        bm25_results=convert_to_chunks(bm25_results),
        vector_results=convert_to_chunks(vector_results),
        fused_results=convert_to_chunks(fused_results),
        final_context=context,
    )
    return AskResponse(
        question=question,
        answer=answer_text,
        sources=sources,
        recommended_menu=recommended_menu,
        recommended_reason=build_recommended_reason(question, sources, recommended_menu),
        diagnostics=diagnostics,
    )
