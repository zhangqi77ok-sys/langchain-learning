"""LangChain 聊天消息结构最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """使用 system 和 human 消息发起一次聊天调用。"""
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    messages = [
        SystemMessage(content="你是一名耐心的 AI 助手，回答要简短清晰。"),
        HumanMessage(content="请解释 LangChain 和直接调用大模型 API 的区别。"),
    ]

    # 这里显式传消息列表，目的是让你看清聊天模型的真实输入结构。
    response = model.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    main()
