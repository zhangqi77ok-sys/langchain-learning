"""LangChain 流式输出最小示例。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """演示模型如何一边生成一边输出内容。"""
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    print("streaming answer:")

    # 这里使用 stream，目的是让你看清输出不是一次性返回，而是按片段到达。
    for chunk in model.stream("请用三句话解释 LangChain 的作用。"):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
