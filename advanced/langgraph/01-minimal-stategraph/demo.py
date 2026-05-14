"""LangGraph 进阶 01：最小 StateGraph 示例。"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class LearningState(TypedDict):
    """定义图里流转的共享状态。"""

    user_input: str
    topic: str
    summary: str


def prepare_topic(state: LearningState) -> dict:
    """从用户输入里整理出当前要学习的主题。"""
    # 这里故意保持最小化。
    # 先不接大模型，只演示“节点读取旧状态，再写回新状态”这件事。
    user_input = state["user_input"].strip()
    topic = f"当前主题：{user_input}"

    print("进入节点 prepare_topic")
    print(f"读取到的 user_input: {user_input}")
    print(f"准备写回的 topic: {topic}")
    print("-" * 20)

    return {"topic": topic}


def build_summary(state: LearningState) -> dict:
    """基于上一个节点产出的 topic 生成一句总结。"""
    # 这个节点不再依赖原始输入，
    # 而是依赖前一个节点已经写进状态里的 topic。
    topic = state["topic"]
    summary = f"{topic}。这说明图已经把上一个节点的结果传到了下一个节点。"

    print("进入节点 build_summary")
    print(f"读取到的 topic: {topic}")
    print(f"准备写回的 summary: {summary}")
    print("-" * 20)

    return {"summary": summary}


def build_graph():
    """构建最小两节点图。"""
    graph_builder = StateGraph(LearningState)

    # 注册节点：告诉图里有哪些处理步骤。
    graph_builder.add_node("prepare_topic", prepare_topic)
    graph_builder.add_node("build_summary", build_summary)

    # 注册边：明确执行顺序。
    graph_builder.add_edge(START, "prepare_topic")
    graph_builder.add_edge("prepare_topic", "build_summary")
    graph_builder.add_edge("build_summary", END)

    # compile 之后，图才变成一个可以真正执行的对象。
    return graph_builder.compile()


def main() -> None:
    """运行最小 StateGraph，并打印最终状态。"""
    graph = build_graph()

    # 初始状态只提供起点数据。
    # 后续字段由节点逐步补齐。
    initial_state: LearningState = {
        "user_input": "LangGraph 的最小工作流",
        "topic": "",
        "summary": "",
    }

    print("初始状态：")
    print(initial_state)
    print("=" * 20)

    final_state = graph.invoke(initial_state)

    print("最终状态：")
    print(final_state)


if __name__ == "__main__":
    main()
