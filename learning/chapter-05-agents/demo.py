"""Chapter 05: 最小 Agent 示例。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必须存在的环境变量。"""
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


@tool
def get_travel_tip(city: str) -> str:
    """根据城市名返回演示用出行建议。"""
    tip_map = {
        "上海": "建议带一把折叠伞，优先地铁出行。",
        "北京": "建议注意早晚温差，备一件薄外套。",
        "深圳": "建议留意阵雨，随身带伞。",
    }
    return tip_map.get(city, f"暂时没有 {city} 的出行建议。")


def main() -> None:
    """运行最小 Agent，观察它如何借助工具回答问题。"""
    root_env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env_path)

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, get_travel_tip],
        system_prompt="你是一名简洁清楚的旅行助手，必要时先调用工具再回答。",
    )

    question = "我明天去上海，请告诉我天气情况，并给一个简短的出行建议。"
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print("问题：")
    print(question)
    print("-" * 40)
    print("最终回答：")
    print(result["messages"][-1].content)
    print("-" * 40)
    print("消息轨迹：")
    for index, message in enumerate(result["messages"], start=1):
        print(f"第 {index} 条消息类型：{message.__class__.__name__}")
        if getattr(message, "content", None):
            print(f"内容：{message.content}")
        if getattr(message, "tool_calls", None):
            print(f"工具调用：{message.tool_calls}")
        print("-" * 20)


if __name__ == "__main__":
    main()
