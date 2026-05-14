"""基础巩固 08：最小 RAG 链路示例。"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    保持和前几组案例一致，
    这样你在比较不同知识点时，不会被环境读取方式的变化干扰。
    """
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
    """把检索到的文档块拼成模型要读取的上下文字符串。

    这一步很关键，因为 retriever 找回来的是多个文档块，
    但模型最终需要读的是一段组织好的上下文文本。
    """
    return "\n\n".join(document.page_content for document in documents)


def main() -> None:
    """演示最小 RAG 流程。

    这一组把前两组知识连起来：
    - 07 组负责“先找资料”
    - 这一组负责“找回资料后，再基于资料回答”
    """
    # 先加载 .env，保证模型和本地 embedding 配置可用。
    load_dotenv()

    # 先准备几段文档。
    # 在真实项目里，这些内容可能来自文件、数据库或企业知识库。
    documents = [
        Document(
            page_content="LangChain 可以帮助开发者组织提示词、模型调用和工作流。",
            metadata={"id": 1, "topic": "langchain"},
        ),
        Document(
            page_content="RAG 的基本流程是：先检索相关资料，再把资料作为上下文交给模型生成答案。",
            metadata={"id": 2, "topic": "rag"},
        ),
        Document(
            page_content="Tool Calling 的核心是让模型在回答前决定是否调用外部工具。",
            metadata={"id": 3, "topic": "tool_calling"},
        ),
    ]

    # 创建本地 embedding 模型。
    # 它负责把文档和问题都变成向量，方便做相似度检索。
    embeddings = LocalEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # 创建内存向量存储，并把文档加入其中。
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    # 这里定义用户问题。
    question = "RAG 的基本思路是什么？"

    # 第一步：检索。
    # 先从文档里找出最相关的前 2 条候选资料。
    retrieved_docs = vector_store.similarity_search(question, k=2)

    # 第二步：把检索结果拼成上下文。
    # 模型不能直接理解“这是 2 个 Document 对象”，
    # 所以要先整理成它能读的文本。
    context = build_context(retrieved_docs)

    # 第三步：定义最终问答提示词。
    # 这里把“上下文”和“问题”一起交给模型。
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名简洁清晰的知识助手。请严格依据上下文回答，不要编造上下文中没有的信息。",
            ),
            ("human", "上下文：\n{context}\n\n问题：\n{question}"),
        ]
    )

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 把提示词和模型串起来。
    chain = prompt | model

    # 第四步：生成最终答案。
    # 这一步才是真正的“生成”阶段。
    response = chain.invoke({"context": context, "question": question})

    print("检索到的上下文：")
    print(context)
    print("-" * 20)
    print("最终回答：")
    print(response.content)


if __name__ == "__main__":
    main()
