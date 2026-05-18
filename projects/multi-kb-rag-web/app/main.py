"""multi-kb-rag-web 的最小 FastAPI 入口。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


app = FastAPI(title="multi-kb-rag-web")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


MENU_ITEMS = [
    {
        "key": "langchain-basics",
        "title": "LangChain 基础",
        "url": "/menus/langchain-basics",
        "source": "langchain_basics.txt",
        "description": "看 LangChain 是什么、能解决什么问题。",
    },
    {
        "key": "rag-basics",
        "title": "RAG 基础",
        "url": "/menus/rag-basics",
        "source": "rag_basics.txt",
        "description": "看 RAG 的流程、价值和典型使用方式。",
    },
    {
        "key": "tool-agent-basics",
        "title": "Tool / Agent 基础",
        "url": "/menus/tool-agent-basics",
        "source": "tool_agent_basics.txt",
        "description": "看工具调用和 Agent 的最小核心概念。",
    },
]

MENU_BY_KEY = {item["key"]: item for item in MENU_ITEMS}
MENU_BY_SOURCE = {item["source"]: item for item in MENU_ITEMS}


class AskRequest(BaseModel):
    """定义问答接口请求体。"""

    question: str


class RecommendedMenu(BaseModel):
    """定义问答后的推荐菜单信息。"""

    key: str
    title: str
    url: str


class AskResponse(BaseModel):
    """定义问答接口响应体。"""

    question: str
    answer: str
    sources: list[str]
    recommended_menu: RecommendedMenu | None


def get_required_env(name: str) -> str:
    """读取必须存在的环境变量，缺失时直接报错。"""
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


def tokenize(text: str) -> list[str]:
    """把文本切成适合 BM25 的最小词元。"""
    cleaned = text.replace("，", " ").replace("。", " ").replace("、", " ")
    return [token for token in cleaned.split() if token]


def normalize_scores(scores: list[float]) -> list[float]:
    """把不同检索器的分数拉到同一范围，方便融合。"""
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
    """把两路检索结果融合后重新排序。"""
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
    """把检索到的文档块拼成上下文字符串。"""
    parts: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        parts.append(f"[{source}]\n{document.page_content}")
    return "\n\n".join(parts)


def choose_recommended_menu(question: str, sources: list[str]) -> RecommendedMenu:
    """根据来源文件和问题内容，给出最小推荐菜单。"""
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


@app.get("/")
def read_root(request: Request):
    """返回首页模板和固定菜单。"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "multi-kb-rag-web",
            "menus": MENU_ITEMS,
        },
    )


@app.get("/menus/{menu_key}")
def read_menu(request: Request, menu_key: str):
    """返回单个菜单详情页。"""
    menu = MENU_BY_KEY.get(menu_key)
    if not menu:
        raise HTTPException(status_code=404, detail="菜单不存在")

    data_path = Path(__file__).resolve().parent.parent / "data" / menu["source"]
    content = data_path.read_text(encoding="utf-8")
    return templates.TemplateResponse(
        request=request,
        name="menu.html",
        context={
            "title": menu["title"],
            "menu": menu,
            "content": content,
            "menus": MENU_ITEMS,
        },
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """基于多文件知识库返回问答结果，并给出推荐菜单。"""
    question = payload.question.strip()
    if not question:
        return AskResponse(
            question="",
            answer="问题不能为空。",
            sources=[],
            recommended_menu=None,
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
    if best_score < 0.35:
        recommended_menu = choose_recommended_menu(question, [])
        return AskResponse(
            question=question,
            answer="当前知识库里没有找到和这个问题足够相关的内容。",
            sources=[],
            recommended_menu=recommended_menu,
        )

    retrieved_docs = [document for document, _ in fused_results]
    context = build_context(retrieved_docs)
    sources = list(
        dict.fromkeys(document.metadata.get("source", "unknown") for document in retrieved_docs)
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

    return AskResponse(
        question=question,
        answer=response.content,
        sources=sources,
        recommended_menu=choose_recommended_menu(question, sources),
    )
