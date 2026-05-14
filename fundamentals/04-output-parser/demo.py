"""基础巩固 04：输出解析器最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    继续保持和前几组案例一致，
    这样你在学习不同知识点时，不会因为配置读取方式变化而分散注意力。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """执行一次带字符串输出解析器的最小调用。

    这一组的核心不是“怎么问”，
    而是“模型回答出来之后，程序怎么接住这个结果”。
    """
    # 先加载 .env，让模型配置可用。
    load_dotenv()

    # 先定义一个最小提示词模板。
    # 这里继续保留模板形式，是为了让链路更接近后面真实项目里的写法。
    prompt = ChatPromptTemplate.from_template(
        "请用一句话介绍 {topic}，语气要 {style}。"
    )

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 创建最简单的输出解析器。
    # StrOutputParser 的作用就是：
    # 把模型返回的消息对象，转换成普通字符串。
    parser = StrOutputParser()

    # 把“提示词 -> 模型 -> 输出解析器”串成一条链。
    # 你可以把它理解成三段流水线：
    # 1. 先生成最终提示词
    # 2. 再调用模型
    # 3. 最后把模型结果解析成字符串
    chain = prompt | model | parser

    # 传入模板变量并执行整条链。
    result = chain.invoke(
        {
            "topic": "LangChain",
            "style": "简洁",
        }
    )

    # 这里打印的 result 已经是普通字符串了，
    # 不再是消息对象。
    print("解析后的结果：")
    print(result)


if __name__ == "__main__":
    main()
