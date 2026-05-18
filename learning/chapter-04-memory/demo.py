"""Chapter 04: Memory 最小示例。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


def get_required_env(name: str) -> str:
    """读取必须存在的环境变量。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """演示“带历史消息”和“不带历史消息”的区别。"""
    root_env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env_path)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    print("第 1 步：先发第一轮消息")
    first_round = [HumanMessage(content="请记住：我最喜欢的编程语言是 Python。")]
    first_response = model.invoke(first_round)
    print("第一轮回答：")
    print(first_response.content)
    print("-" * 40)

    print("第 2 步：不带历史消息，直接问第二轮")
    second_round_without_history = [
        HumanMessage(content="我刚才最喜欢的编程语言是什么？")
    ]
    second_response_without_history = model.invoke(second_round_without_history)
    print("不带历史时的回答：")
    print(second_response_without_history.content)
    print("-" * 40)

    print("第 3 步：带上历史消息，再问第二轮")
    second_round_with_history = [
        HumanMessage(content="请记住：我最喜欢的编程语言是 Python。"),
        AIMessage(content=first_response.content),
        HumanMessage(content="我刚才最喜欢的编程语言是什么？"),
    ]
    second_response_with_history = model.invoke(second_round_with_history)
    print("带历史时的回答：")
    print(second_response_with_history.content)
    print("-" * 40)

    print("结论：")
    print("很多 memory 的基础做法，就是把前面的消息继续放回下一次输入。")


if __name__ == "__main__":
    main()
