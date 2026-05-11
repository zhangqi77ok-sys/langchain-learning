"""LangChain Agent 最小示例。"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
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
    }
    return weather_map.get(city, f"暂时没有 {city} 的天气信息。")


@tool
def get_travel_tip(city: str) -> str:
    """根据城市名返回演示用出行建议。"""
    tip_map = {
        "上海": "建议带一把折叠伞，地铁出行更稳妥。",
        "北京": "建议注意早晚温差，出门备一件薄外套。",
    }
    return tip_map.get(city, f"暂时没有 {city} 的出行建议。")


def main() -> None:
    """让 Agent 自己决定调用哪些工具并整合结果。"""
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, get_travel_tip],
        system_prompt="你是一名简洁清晰的旅行助手，必要时调用工具获取信息。",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我明天去上海，请告诉我天气情况，并给一个出行建议。",
                }
            ]
        }
    )

    print("final messages:")
    for message in result["messages"]:
        print(message)
        print("-" * 20)


if __name__ == "__main__":
    main()
