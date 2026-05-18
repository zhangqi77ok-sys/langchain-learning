"""Chapter 02: PromptTemplate 最小示例。"""

from langchain_core.prompts import PromptTemplate


def render_prompt(role: str, task: str, language: str) -> str:
    """把固定模板和可变变量拼成最终提示词。"""
    template = PromptTemplate.from_template(
        "你是{role}。\n"
        "你的任务是：{task}。\n"
        "请用{language}回答。"
    )
    return template.format(role=role, task=task, language=language)


def main() -> None:
    """演示同一个模板在不同输入下如何变化。"""
    first_prompt = render_prompt(
        role="一名严谨的 Python 教学助手",
        task="解释什么是 PromptTemplate",
        language="中文",
    )
    second_prompt = render_prompt(
        role="一名简洁的 LangChain 教学助手",
        task="说明为什么 prompt 需要模板化",
        language="中文",
    )

    print("第一组提示词：")
    print(first_prompt)
    print("-" * 40)
    print("第二组提示词：")
    print(second_prompt)


if __name__ == "__main__":
    main()
