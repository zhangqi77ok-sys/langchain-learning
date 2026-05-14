"""基础巩固 12：Streaming 最小示例。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    继续保持和前几组基础案例一致，
    这样你可以专注在“流式返回方式”本身，而不是配置读取差异。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """演示最小流式输出过程。

    这一组的重点不是问题内容，
    而是观察模型结果是如何一小段一小段返回的。
    """
    # 先加载 .env。
    load_dotenv()

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    print("流式输出开始：")

    # 这里不用 invoke(...)，而改用 stream(...)。
    # invoke(...) 的特点是：等完整结果生成后一次性返回。
    # stream(...) 的特点是：模型每生成一小段，就先把这一小段交回来。
    for chunk in model.stream("请用三句话解释 LangChain 的作用。"):
        # chunk 仍然是消息片段对象，不一定每次都有可打印文本。
        if chunk.content:
            # 这里不用换行，是为了让你直观看到“边生成边显示”的过程。
            print(chunk.content, end="", flush=True)

    # 最后补一个换行，让终端输出更整齐。
    print()


if __name__ == "__main__":
    main()
