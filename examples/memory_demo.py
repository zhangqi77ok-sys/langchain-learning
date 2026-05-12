"""LangChain 多轮对话记忆最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """按会话 ID 获取对应的内存消息历史。"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def main() -> None:
    """演示同一会话里模型如何利用上一轮对话内容。"""
    load_dotenv()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名简洁清晰的 AI 助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    chain = prompt | model
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "demo-session"}}

    first_response = chain_with_history.invoke(
        {"question": "请记住：我最喜欢的编程语言是 Python。"},
        config=config,
    )
    print("first answer:")
    print(first_response.content)
    print("-" * 20)

    # 第二轮不再重复提供事实，目的是验证历史消息是否真的被带入了模型。
    second_response = chain_with_history.invoke(
        {"question": "我刚才最喜欢的编程语言是什么？"},
        config=config,
    )
    print("second answer:")
    print(second_response.content)


if __name__ == "__main__":
    main()
