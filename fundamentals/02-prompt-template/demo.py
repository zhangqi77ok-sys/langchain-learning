"""基础巩固 02：PromptTemplate 最小示例。"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    这一层和上一组案例保持一致，
    目的是让“配置错误”尽早暴露，而不是把问题拖到后面的模型调用阶段。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """执行一次最小 PromptTemplate 调用。

    这一组和上一组最大的区别在于：
    上一组是直接把问题字符串交给模型；
    这一组是先定义模板，再把变量填进去。
    """
    # 先读取 .env，让模型配置可以从环境中拿到。
    load_dotenv()

    # 创建一个最小聊天提示词模板。
    # 这里的 {style}、{target}、{topic} 都是“模板变量”。
    # 它们不是现在就有值，而是等真正调用时再传进去。
    prompt = ChatPromptTemplate.from_template(
        "请用{style}的方式，向{target}解释什么是{topic}。"
    )

    # 创建聊天模型对象。
    # 这部分和最小模型调用示例基本一致。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 用 `prompt | model` 把“提示词模板”和“模型”串起来。
    # 你可以先把它粗略理解成：
    # “先生成最终提示词，再把提示词交给模型”。
    chain = prompt | model

    # 调用时再给模板变量传值。
    # 这里的字典键名必须和模板里的变量名一致。
    response = chain.invoke(
        {
            "style": "通俗易懂",
            "target": "刚接触大模型开发的初学者",
            "topic": "LangChain",
        }
    )

    # 返回值仍然是消息对象，真正答案在 content 里。
    print("模型回答：")
    print(response.content)


if __name__ == "__main__":
    main()
