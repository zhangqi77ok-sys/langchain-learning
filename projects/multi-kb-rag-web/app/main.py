"""multi-kb-rag-web 的最小 FastAPI 入口。"""

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
from sentence_transformers import SentenceTransformer


app = FastAPI(title="multi-kb-rag-web")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


class AskRequest(BaseModel):
    """定义问答接口请求体。"""

    question: str


class AskResponse(BaseModel):
    """定义问答接口响应体。"""

    question: str
    answer: str
    sources: list[str]


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


@app.get("/")
def read_root(request: Request):
    """返回最小首页模板。"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "multi-kb-rag-web"},
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """基于多文件知识库返回真实问答响应。"""
    question = payload.question.strip()
    if not question:
        return AskResponse(
            question="",
            answer="问题不能为空。",
            sources=[],
        )

    kb_path = Path(__file__).resolve().parent.parent / "data"
    documents = load_knowledge_base(kb_path)
    embeddings = LocalEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    retrieved_docs = vector_store.similarity_search(question, k=4)
    context = build_context(retrieved_docs)
    sources = list(dict.fromkeys(document.metadata.get("source", "unknown") for document in retrieved_docs))

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
    response = chain.invoke({"context": context, "question": question})

    return AskResponse(
        question=question,
        answer=response.content,
        sources=sources,
    )
