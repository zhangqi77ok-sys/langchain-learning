"""Chapter 08: 项目视角说明脚本。"""

from pathlib import Path


def main() -> None:
    """打印当前学习内容和项目之间的对应关系。"""
    project_path = Path("projects/multi-kb-rag-web").resolve()

    print("第 1 步：确认当前项目入口")
    print(project_path)
    print("-" * 40)

    print("第 2 步：理解项目可以拆成哪些层")
    print("1. Web 层：页面输入问题，页面展示答案")
    print("2. 应用层：接收问题，组织检索和模型调用")
    print("3. 检索层：加载资料、切分、向量化、召回")
    print("4. 模型层：根据上下文生成最终答案")
    print("-" * 40)

    print("第 3 步：把前面章节映射到项目")
    print("chapter-01-basics -> 模型最小调用")
    print("chapter-02-prompting -> 提示词组织")
    print("chapter-03-rag -> 检索问答主链路")
    print("chapter-04-memory -> 多轮上下文")
    print("chapter-05-agents -> 工具决策")
    print("chapter-06-hybrid-retrieval -> 检索增强")
    print("chapter-07-langgraph -> 流程编排")
    print("-" * 40)

    print("结论：")
    print("从这一章开始，学习重点是把知识点放回项目，而不是继续孤立看 demo。")


if __name__ == "__main__":
    main()
