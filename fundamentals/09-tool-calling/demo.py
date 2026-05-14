"""基础巩固 09：Tool Calling 最小闭环示例。"""

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    继续保持基础案例的统一写法，
    让你在不同主题间切换时，注意力只放在当前知识点本身。
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


@tool
def get_weather(city: str) -> str:
    """根据城市名返回演示用天气信息。

    这里的工具不是“真的连外部天气 API”，
    而是一个本地演示工具。
    这样你能先把注意力放在 Tool Calling 机制本身。
    """
    weather_map = {
        "上海": "上海今天多云，气温 26 到 31 度。",
        "北京": "北京今天晴，气温 22 到 29 度。",
        "深圳": "深圳今天小雨，气温 27 到 32 度。",
    }
    return weather_map.get(city, f"暂时没有 {city} 的天气信息。")


def main() -> None:
    """演示 Tool Calling 的最小完整闭环。

    这一组的关键点是：
    模型第一次响应，不一定直接给最终答案，
    它可能先说“我要调用哪个工具”。
    """
    # 先加载 .env。
    load_dotenv()

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 把本地工具绑定到模型。
    # 绑定之后，模型就知道“自己有哪些可调用工具”。
    llm_with_tools = model.bind_tools([get_weather])

    # 先发一个会触发工具的问题。
    query = "请告诉我上海今天天气怎么样。"

    # 第一次调用模型。
    # 这时模型可能直接回答，也可能先返回 tool_calls。
    first_response = llm_with_tools.invoke(query)

    print("第一次模型响应：")
    print(first_response)
    print("-" * 20)

    # 如果模型没有触发工具，那这个示例的闭环就不会继续往下走。
    # 这里加这个判断，是为了把分支逻辑明确展示出来。
    if not first_response.tool_calls:
        print("模型没有触发工具调用。")
        return

    # 这里取第一个工具调用。
    # 在更复杂场景里，模型可能一次返回多个 tool_calls。
    tool_call = first_response.tool_calls[0]

    # 这一步才是真正执行本地工具。
    # 注意：不是模型自己执行 Python，
    # 而是我们的程序根据模型意图去执行。
    tool_result = get_weather.invoke(tool_call["args"])

    # 把工具结果包装成 ToolMessage，再交回模型。
    # 这样模型才能基于工具结果生成最终回答。
    final_response = llm_with_tools.invoke(
        [
            first_response,
            ToolMessage(
                # 这里用 JSON 字符串包装，是为了让结果在消息里更清晰可读。
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=tool_call["id"],
            ),
        ]
    )

    print("工具执行结果：")
    print(tool_result)
    print("-" * 20)
    print("最终回答：")
    print(final_response.content)


if __name__ == "__main__":
    main()
