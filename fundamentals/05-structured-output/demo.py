"""基础巩固 05：结构化输出最小示例。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    保持与前几组案例一致，
    让你的注意力始终放在“当前这个知识点”本身，而不是环境配置细节变化。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


class LangChainSummary(BaseModel):
    """定义模型输出的数据结构。

    这里最关键的不是类名，而是“固定字段”这个概念。
    你是在提前告诉模型：
    “我后面不是只想看一段话，我要拿到这些明确的字段。”
    """

    concept: str = Field(description="LangChain 是什么")
    usage: str = Field(description="LangChain 的一个核心用途")
    advice: str = Field(description="给初学者的一句学习建议")


def main() -> None:
    """执行一次最小结构化输出调用。

    这一组和上一组的关键区别在于：
    上一组最终拿到的是字符串；
    这一组最终拿到的是带固定字段的对象。
    """
    # 先加载 .env，确保模型配置可用。
    load_dotenv()

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 这里不是普通 model 了，而是“结构化输出版本的 model”。
    # 你可以先把它理解成：
    # “同样还是调模型，但希望它最后按 LangChainSummary 这个结构回结果”。
    structured_model = model.with_structured_output(LangChainSummary)

    # 这里直接发一个问题，让模型按固定字段返回结果。
    result = structured_model.invoke("请面向初学者介绍一下 LangChain。")

    # 这时拿到的 result 不再是普通消息对象，
    # 而是一个 LangChainSummary 实例。
    print("结构化结果：")
    print(f"concept: {result.concept}")
    print(f"usage: {result.usage}")
    print(f"advice: {result.advice}")


if __name__ == "__main__":
    main()
