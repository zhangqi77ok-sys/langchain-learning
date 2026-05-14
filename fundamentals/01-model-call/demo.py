"""基础巩固 01：最小模型调用示例。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    这里单独封装的原因不是为了“炫技”，
    而是为了让“环境变量缺失”这个错误更早暴露出来。
    这样你一运行就知道问题出在配置，而不是后面的模型调用。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def main() -> None:
    """执行一次最小模型调用。

    这段代码就是最基础的 LangChain 使用方式：
    1. 先加载 .env
    2. 再创建模型对象
    3. 然后把一个问题发给模型
    4. 最后打印模型回答
    """
    # 让程序先读取仓库根目录下的 .env 文件。
    # 没有这一步的话，OPENAI_API_KEY 这类变量通常读不到。
    load_dotenv()

    # 创建一个聊天模型对象。
    # ChatOpenAI 是 LangChain 对“聊天模型”的封装。
    # 这里虽然你用的是第三方兼容接口，
    # 但对 LangChain 来说，调用方式仍然和 OpenAI-compatible 模型一致。
    model = ChatOpenAI(
        # 优先读取环境变量中的模型名。
        # 如果你后面切模型，只改 .env 就行，不用改代码。
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        # API Key 必须存在，否则模型根本无法调用。
        api_key=get_required_env("OPENAI_API_KEY"),
        # 这里支持第三方兼容接口地址。
        # 如果你用官方接口，这个值也可以不配。
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # invoke(...) 是最基础的调用方法。
    # 这里直接传一个字符串，意思就是：
    # “把这句话作为用户输入发给聊天模型”。
    response = model.invoke("请用一句话解释什么是 LangChain。")

    # response 不是普通字符串，而是一个消息对象。
    # 真正的回答文本在 response.content 里。
    print("模型回答：")
    print(response.content)


if __name__ == "__main__":
    main()
