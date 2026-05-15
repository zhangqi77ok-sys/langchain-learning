"""LangGraph 进阶 03：多节点工作流示例。"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    """定义工作流中流转的共享状态。"""

    user_question: str
    normalized_question: str
    draft_answer: str
    polished_answer: str
    final_answer: str


def normalize_question(state: WorkflowState) -> dict:
    """先把原始问题整理成更适合后续处理的形式。"""
    question = state["user_question"].strip()
    normalized_question = question.replace("？", "?")

    print("进入节点 normalize_question")
    print(f"读取到的 user_question: {question}")
    print(f"写回 normalized_question: {normalized_question}")
    print("-" * 20)

    return {"normalized_question": normalized_question}


def draft_answer(state: WorkflowState) -> dict:
    """基于规范化后的问题先生成草稿答案。"""
    normalized_question = state["normalized_question"]
    draft_answer = f"草稿答案：围绕「{normalized_question}」先给出一个直接回复。"

    print("进入节点 draft_answer")
    print(f"读取到的 normalized_question: {normalized_question}")
    print(f"写回 draft_answer: {draft_answer}")
    print("-" * 20)

    return {"draft_answer": draft_answer}


def polish_answer(state: WorkflowState) -> dict:
    """对草稿答案做一次整理，让表达更完整。"""
    draft_answer = state["draft_answer"]
    polished_answer = f"{draft_answer} 这里再补一句：这样拆分节点更容易维护。"

    print("进入节点 polish_answer")
    print(f"读取到的 draft_answer: {draft_answer}")
    print(f"写回 polished_answer: {polished_answer}")
    print("-" * 20)

    return {"polished_answer": polished_answer}


def finalize_answer(state: WorkflowState) -> dict:
    """把最终结果收口成对外输出的答案。"""
    polished_answer = state["polished_answer"]
    final_answer = f"最终答案：{polished_answer}"

    print("进入节点 finalize_answer")
    print(f"读取到的 polished_answer: {polished_answer}")
    print(f"写回 final_answer: {final_answer}")
    print("-" * 20)

    return {"final_answer": final_answer}


def build_graph():
    """构建一个线性的多节点工作流。"""
    graph_builder = StateGraph(WorkflowState)

    graph_builder.add_node("normalize_question", normalize_question)
    graph_builder.add_node("draft_answer", draft_answer)
    graph_builder.add_node("polish_answer", polish_answer)
    graph_builder.add_node("finalize_answer", finalize_answer)

    graph_builder.add_edge(START, "normalize_question")
    graph_builder.add_edge("normalize_question", "draft_answer")
    graph_builder.add_edge("draft_answer", "polish_answer")
    graph_builder.add_edge("polish_answer", "finalize_answer")
    graph_builder.add_edge("finalize_answer", END)

    return graph_builder.compile()


def main() -> None:
    """运行多节点工作流，并观察状态如何逐步变完整。"""
    graph = build_graph()

    initial_state: WorkflowState = {
        "user_question": "LangGraph 为什么适合做工作流？",
        "normalized_question": "",
        "draft_answer": "",
        "polished_answer": "",
        "final_answer": "",
    }

    print("初始状态：")
    print(initial_state)
    print("=" * 20)

    final_state = graph.invoke(initial_state)

    print("最终状态：")
    print(final_state)


if __name__ == "__main__":
    main()
