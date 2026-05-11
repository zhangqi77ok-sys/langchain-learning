"""LangChain PromptTemplate 最小示例。"""

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
    """使用提示词模板生成一次模型调用。"""
    load_dotenv()

    prompt = ChatPromptTemplate.from_template(
        "请用{style}的方式，向{target}解释什么是{topic}。"
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    chain = prompt | model

    # 这里先固定参数，目的是让你看清模板变量是怎么传进去的。
    response = chain.invoke(
        {
            "style": "通俗易懂",
            "target": "初学者",
            "topic": "LangChain",
        }
    )
    print(response.content)


if __name__ == "__main__":
    main()
