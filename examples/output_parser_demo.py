"""LangChain 输出解析器最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """通过输出解析器拿到纯字符串结果。"""
    load_dotenv()

    prompt = ChatPromptTemplate.from_template(
        "请用一句话介绍 {topic}，要求语气 {style}。"
    )

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 这里加 parser，目的是让你看清模型消息对象如何转成纯文本。
    result = chain.invoke({"topic": "LangChain", "style": "简洁"})
    print(result)


if __name__ == "__main__":
    main()
