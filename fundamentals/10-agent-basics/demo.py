"""基础巩固 10：Agent 最小示例。"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


def get_required_env(name: str) -> str:
    """读取必需环境变量。

    保持和前几组基础案例一致，
    让你把注意力集中在 Agent 的行为差异上，而不是配置读取细节。
    """
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
        "深圳": "建议留意降雨，随身带伞。",
    }
    return tip_map.get(city, f"暂时没有 {city} 的出行建议。")


def main() -> None:
    """演示 Agent 如何自动完成多步工具决策。

    这一组和上一组最大的区别在于：
    上一组是你手动接住 tool_calls 再回给模型；
    这一组是框架自动帮你跑这个循环。
    """
    # 先加载 .env。
    load_dotenv()

    # 创建聊天模型对象。
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 创建 Agent。
    # 这里把两个工具一起交给 Agent，
    # Agent 会自己判断：要不要调用、调用哪个、是否还要继续调用下一个。
    agent = create_agent(
        model=model,
        tools=[get_weather, get_travel_tip],
        system_prompt="你是一名简洁清晰的旅行助手，必要时调用工具后再回答。",
    )

    # 给 Agent 一个需要整合多个信息的问题。
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我明天去上海，请告诉我天气情况，并给一个简短的出行建议。",
                }
            ]
        }
    )

    # Agent 返回的是完整消息轨迹。
    # 这里打印最终一条消息，方便你先抓住“最终产出”。
    print("最终回答：")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
