"""LangChain 聊天提示词模板最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """使用聊天提示词模板发起一次模型调用。"""
    load_dotenv()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名{role}，回答要{style}。"),
            ("human", "请向{target}解释什么是{topic}。"),
        ]
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    chain = prompt | model

    # 这里同时给 system 和 human 模板传参，目的是让你看清组合方式。
    response = chain.invoke(
        {
            "role": "AI 编程老师",
            "style": "简短清晰",
            "target": "刚接触大模型开发的初学者",
            "topic": "LangChain",
        }
    )
    print(response.content)


if __name__ == "__main__":
    main()
