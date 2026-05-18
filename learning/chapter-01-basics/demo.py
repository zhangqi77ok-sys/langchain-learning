"""Chapter 01: 最小模型调用示例。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必须存在的环境变量。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """执行一次最小模型调用。"""
    # 从仓库根目录读取 .env。
    root_env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env_path)

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 发起一次最小调用。
    response = model.invoke("请用一句话介绍 LangChain。")

    print("模型返回：")
    print(response.content)


if __name__ == "__main__":
    main()
