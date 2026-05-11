"""LangChain 最小示例：调用聊天模型并输出结果。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_api_key() -> str:
    """读取 OpenAI API Key，缺失时直接报错。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("缺少环境变量 OPENAI_API_KEY")
    return api_key


def main() -> None:
    """执行一次最小 LangChain 调用。"""
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_api_key(),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 这里先固定提示词，目的是让你先跑通最小闭环。
    response = model.invoke("请用一句话解释什么是 LangChain。")
    print(response.content)


if __name__ == "__main__":
    main()
