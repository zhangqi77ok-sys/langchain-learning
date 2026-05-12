"""带流式文本输出和结构化结果的旅行助手示例。"""

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


def get_required_env(name: str) -> str:
    """读取必需环境变量，缺失时直接报错。"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


class TravelPlan(BaseModel):
    """定义旅行助手的结构化输出。"""

    model_config = ConfigDict(populate_by_name=True)

    city: str = Field(
        validation_alias=AliasChoices("城市", "地点", "city"),
        description="用户提到的城市",
    )
    weather: str = Field(
        validation_alias=AliasChoices("天气", "weather"),
        description="该城市的天气信息",
    )
    travel_tip: str | list[str] = Field(
        validation_alias=AliasChoices("出行建议", "travel_tip"),
        description="该城市的出行建议",
    )
    summary: str | None = Field(
        default=None,
        validation_alias=AliasChoices("总结", "summary"),
        description="面向用户的一句简短总结",
    )


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
        "上海": "建议带一把折叠伞，优先选择地铁出行。",
        "北京": "建议注意早晚温差，备一件薄外套。",
        "深圳": "建议留意降雨，随身带伞。",
    }
    return tip_map.get(city, f"暂时没有 {city} 的出行建议。")


def main() -> None:
    """先流式展示旅行建议，再输出结构化结果。"""
    load_dotenv()

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入你的旅行问题：").strip()
    if not question:
        raise ValueError("问题不能为空")

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    agent = create_agent(
        model=model,
        tools=[get_weather, get_travel_tip],
        system_prompt="你是一名简洁清晰的旅行助手，必要时调用工具后再回答。",
    )

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
    final_answer = result["messages"][-1].content

    print("streaming answer:")
    stream_model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get_required_env("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 这里用第二次模型调用做流式展示，目的是同时保留 Agent 工具链和终端流式体验。
    for chunk in stream_model.stream(
        f"请把下面这段旅行建议原样转述给用户，保持简洁清晰：\n\n{final_answer}"
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
    print("-" * 20)

    structured_model = model.with_structured_output(TravelPlan)
    structured = structured_model.invoke(
        (
            "请根据下面这段旅行建议，整理出结构化结果。"
            "如果缺少某个字段，请基于文本内容提取，不要编造文本里没有的信息。\n\n"
            f"{final_answer}"
        )
    )
    travel_tip = structured.travel_tip
    if isinstance(travel_tip, list):
        travel_tip = "；".join(travel_tip)
    summary = structured.summary or f"{structured.city}{structured.weather}，{travel_tip}"

    print("structured result:")
    print(f"city: {structured.city}")
    print(f"weather: {structured.weather}")
    print(f"travel_tip: {travel_tip}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
