"""基础巩固 11：Memory 最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    继续保持基础案例的统一写法，
    让你把注意力集中在 Memory 机制本身，而不是配置读取细节。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


# 这里用一个最小字典模拟“按会话 ID 存储历史消息”。
# 键是 session_id，值是该会话对应的消息历史。
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据会话 ID 获取对应的历史消息对象。

    如果这个会话第一次出现，就给它新建一个空的历史消息容器。
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def main() -> None:
    """演示多轮对话里的最小记忆闭环。

    这一组的重点不是“模型多聪明”，
    而是“同一个会话的历史消息如何再次进入模型输入”。
    """
    # 先加载 .env。
    load_dotenv()

    # 创建一个聊天提示词模板。
    # MessagesPlaceholder 的作用是：
    # 在这里预留一个“历史消息插槽”。
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名简洁清晰的 AI 助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 先把提示词模板和模型串起来。
    chain = prompt | model

    # 再给这条链包上一层“带消息历史”的能力。
    # 这样每次调用时，它都会根据 session_id 自动拿历史消息。
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 用同一个 session_id，表示这两轮属于同一段会话。
    config = {"configurable": {"session_id": "demo-session"}}

    # 第一轮：先告诉模型一个事实。
    first_response = chain_with_history.invoke(
        {"question": "请记住：我最喜欢的编程语言是 Python。"},
        config=config,
    )
    print("第一轮回答：")
    print(first_response.content)
    print("-" * 20)

    # 第二轮：不重复提供前文事实，只做追问。
    # 如果历史消息真的被带进去了，模型就能回答出来。
    second_response = chain_with_history.invoke(
        {"question": "我刚才最喜欢的编程语言是什么？"},
        config=config,
    )
    print("第二轮回答：")
    print(second_response.content)


if __name__ == "__main__":
    main()
