"""基础巩固 03：聊天消息结构最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    继续保留这一层，是为了让配置问题尽早暴露，
    保持每一组基础案例的错误模式一致，方便你形成稳定心智模型。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """执行一次最小聊天消息调用。

    这一组和上一组最大的区别在于：
    上一组重点是“模板变量”；
    这一组重点是“消息角色”。
    """
    # 先加载 .env，确保模型配置能够被正确读取。
    load_dotenv()

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 这里显式构造消息列表。
    # 你可以把它理解成：
    # “不是直接发一句话给模型，而是把对话结构整理清楚再发给模型”。
    messages = [
        # SystemMessage 用来定义角色、边界、回答风格。
        # 它不是用户问题，而是“系统级说明”。
        SystemMessage(content="你是一名简洁清晰的 AI 助手，回答尽量短一些。"),
        # HumanMessage 才是真正的用户提问。
        HumanMessage(content="请解释 LangChain 和直接调用大模型 API 的区别。"),
    ]

    # invoke(...) 这次接收的不再是单个字符串，而是消息列表。
    response = model.invoke(messages)

    # 返回值仍然是消息对象，真正文本在 content 中。
    print("模型回答：")
    print(response.content)


if __name__ == "__main__":
    main()
