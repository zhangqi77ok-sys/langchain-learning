"""Chapter 07: LangGraph 最小两节点示例。"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class LearningState(TypedDict):
    """定义图里流转的共享状态。"""

    user_input: str
    topic: str
    summary: str


def prepare_topic(state: LearningState) -> dict:
    """第一个节点：先根据输入整理出主题。"""
    user_input = state["user_input"].strip()
    topic = f"当前主题：{user_input}"

    print("进入节点 1：prepare_topic")
    print(f"读到的 user_input: {user_input}")
    print(f"准备写回的 topic: {topic}")
    print("-" * 40)

    return {"topic": topic}


def build_summary(state: LearningState) -> dict:
    """第二个节点：读取前一节点写入的主题，再生成总结。"""
    topic = state["topic"]
    summary = f"{topic}。这说明第二个节点已经拿到了第一个节点写入的状态。"

    print("进入节点 2：build_summary")
    print(f"读到的 topic: {topic}")
    print(f"准备写回的 summary: {summary}")
    print("-" * 40)

    return {"summary": summary}


def build_graph():
    """构建最小两节点图。"""
    graph_builder = StateGraph(LearningState)

    graph_builder.add_node("prepare_topic", prepare_topic)
    graph_builder.add_node("build_summary", build_summary)

    graph_builder.add_edge(START, "prepare_topic")
    graph_builder.add_edge("prepare_topic", "build_summary")
    graph_builder.add_edge("build_summary", END)

    return graph_builder.compile()


def main() -> None:
    """运行最小 LangGraph，并观察状态如何流转。"""
    graph = build_graph()

    initial_state: LearningState = {
        "user_input": "LangGraph 的最小工作流",
        "topic": "",
        "summary": "",
    }

    print("第 1 步：准备初始状态")
    print(initial_state)
    print("=" * 40)

    print("第 2 步：运行图")
    final_state = graph.invoke(initial_state)
    print("=" * 40)

    print("第 3 步：查看最终状态")
    print(final_state)
    print("=" * 40)

    print("结论：")
    print("LangGraph 的基础能力，就是让状态在多个节点之间按顺序流转。")


if __name__ == "__main__":
    main()
