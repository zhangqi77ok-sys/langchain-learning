"""LangChain 结构化输出最小示例。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class LangChainSummary(BaseModel):
    """定义模型输出结构。"""

    concept: str = Field(description="LangChain 是什么")
    usage: str = Field(description="LangChain 的一个核心用途")
    advice: str = Field(description="给初学者的一句学习建议")


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """让模型按固定字段返回结构化结果。"""
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    structured_model = model.with_structured_output(LangChainSummary)

    # 这里要求固定字段输出，目的是让你看到结果如何直接变成对象。
    result = structured_model.invoke("请介绍一下 LangChain，面向初学者回答。")
    print(f"concept: {result.concept}")
    print(f"usage: {result.usage}")
    print(f"advice: {result.advice}")


if __name__ == "__main__":
    main()
