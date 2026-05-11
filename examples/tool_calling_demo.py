"""LangChain Tool Calling 最小示例。"""

import os
import json

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


@tool
def get_weather(city: str) -> str:
    """根据城市名返回演示用天气信息。"""
    weather_map = {
        "上海": "上海今天多云，气温 26 到 31 度。",
        "北京": "北京今天晴，气温 22 到 29 度。",
        "深圳": "深圳今天小雨，气温 27 到 32 度。",
    }
    return weather_map.get(city, f"暂时没有 {city} 的天气信息。")


def main() -> None:
    """让模型决定是否调用本地工具。"""
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    llm_with_tools = model.bind_tools([get_weather])

    query = "请告诉我上海今天天气怎么样。"
    first_response = llm_with_tools.invoke(query)

    print("first response:")
    print(first_response)
    print("-" * 20)

    if not first_response.tool_calls:
        print("模型没有触发工具调用。")
        return

    tool_call = first_response.tool_calls[0]
    tool_result = get_weather.invoke(tool_call["args"])

    # 这里补一条 ToolMessage，目的是让模型拿到工具结果后再组织最终回答。
    final_response = llm_with_tools.invoke(
        [
            first_response,
            ToolMessage(
                content=json.dumps(tool_result, ensure_ascii=False),
                tool_call_id=tool_call["id"],
            ),
        ]
    )

    print("tool result:")
    print(tool_result)
    print("-" * 20)
    print("final answer:")
    print(final_response.content)


if __name__ == "__main__":
    main()
