"""LangGraph 进阶 02：条件分支示例。"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


QuestionType = Literal["simple", "complex"]
RouteName = Literal["answer_simple_question", "suggest_rag_path"]


class BranchState(TypedDict):
    """定义分支图里流转的共享状态。"""

    user_question: str
    question_type: QuestionType
    answer: str


def classify_question(state: BranchState) -> dict:
    """先根据问题内容做一次简单分类。"""
    question = state["user_question"].strip()

    # 这里故意用非常直白的规则，避免第一次学习就被模型调用干扰。
    # 当前规则只是为了演示“先分类，再分流”的工作流结构。
    complex_keywords = ["比较", "区别", "方案", "设计", "架构", "项目"]
    question_type: QuestionType = "simple"
    if any(keyword in question for keyword in complex_keywords):
        question_type = "complex"

    print("进入节点 classify_question")
    print(f"读取到的问题: {question}")
    print(f"分类结果 question_type: {question_type}")
    print("-" * 20)

    return {"question_type": question_type}


def route_by_question_type(state: BranchState) -> RouteName:
    """根据分类结果决定下一步走哪条边。"""
    question_type = state["question_type"]

    print("进入路由函数 route_by_question_type")
    print(f"当前 question_type: {question_type}")

    if question_type == "simple":
        print("路由到: answer_simple_question")
        print("-" * 20)
        return "answer_simple_question"

    print("路由到: suggest_rag_path")
    print("-" * 20)
    return "suggest_rag_path"


def answer_simple_question(state: BranchState) -> dict:
    """处理简单问题分支。"""
    question = state["user_question"]
    answer = f"这是一个简单问题：{question}。当前流程选择直接回答，不进入检索链路。"

    print("进入节点 answer_simple_question")
    print(f"准备写回的 answer: {answer}")
    print("-" * 20)

    return {"answer": answer}


def suggest_rag_path(state: BranchState) -> dict:
    """处理复杂问题分支。"""
    question = state["user_question"]
    answer = (
        f"这是一个复杂问题：{question}。"
        "当前流程判断更适合先做知识检索，再基于检索结果生成回答。"
    )

    print("进入节点 suggest_rag_path")
    print(f"准备写回的 answer: {answer}")
    print("-" * 20)

    return {"answer": answer}


def build_graph():
    """构建带条件分支的最小图。"""
    graph_builder = StateGraph(BranchState)

    graph_builder.add_node("classify_question", classify_question)
    graph_builder.add_node("answer_simple_question", answer_simple_question)
    graph_builder.add_node("suggest_rag_path", suggest_rag_path)

    graph_builder.add_edge(START, "classify_question")

    # 这里不是固定 add_edge，而是条件边。
    # LangGraph 会先执行 classify_question，
    # 再调用路由函数决定后面去哪一个节点。
    graph_builder.add_conditional_edges(
        "classify_question",
        route_by_question_type,
    )

    graph_builder.add_edge("answer_simple_question", END)
    graph_builder.add_edge("suggest_rag_path", END)

    return graph_builder.compile()


def run_case(graph, question: str) -> None:
    """运行单个问题案例，方便对比不同分支。"""
    initial_state: BranchState = {
        "user_question": question,
        "question_type": "simple",
        "answer": "",
    }

    print("初始状态：")
    print(initial_state)
    print("=" * 20)

    final_state = graph.invoke(initial_state)

    print("最终状态：")
    print(final_state)
    print("=" * 40)


def main() -> None:
    """分别演示简单问题和复杂问题如何走不同分支。"""
    graph = build_graph()

    print("案例 1：简单问题")
    run_case(graph, "什么是 LangGraph？")

    print("案例 2：复杂问题")
    run_case(graph, "LangChain 和 LangGraph 在项目设计上的区别是什么？")


if __name__ == "__main__":
    main()
