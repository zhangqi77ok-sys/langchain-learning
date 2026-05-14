"""基础巩固 06：Document 与文本切分最小示例。"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    """演示如何把一个长文档切成多个检索友好的文本块。

    这一组的重点不是模型调用，
    而是“知识资料在进入 RAG 之前，应该先被整理成什么样子”。
    """
    # 这里先手工构造一个 Document。
    # page_content 是正文内容，
    # metadata 是文档的附加信息。
    document = Document(
        page_content=(
            "LangChain 是一个帮助开发者构建大语言模型应用的开发框架。"
            "它把提示词、模型调用、输出解析、检索、工具调用等能力组织起来。"
            "如果只是简单问答，直接调用模型 API 往往已经够用。"
            "但当系统需要知识库、工作流或结构化输出时，LangChain 会更有价值。"
        ),
        metadata={
            "source": "manual_note",
            "topic": "langchain_intro",
        },
    )

    # 创建一个递归字符切分器。
    # chunk_size 决定每块大致有多长，
    # chunk_overlap 决定相邻块之间保留多少重叠内容。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=10,
    )

    # 这里不是直接切字符串，而是切 Document。
    # 好处是：切分后的每个块还能保留原始 metadata。
    chunks = splitter.split_documents([document])

    print("切分结果：")
    for index, chunk in enumerate(chunks, start=1):
        # 这里逐块打印，目的是让你看清：
        # 一个长文档经过切分后，会变成多个更适合检索的小块。
        print(f"chunk {index}")
        print(f"content: {chunk.page_content}")
        print(f"metadata: {chunk.metadata}")
        print("-" * 20)


if __name__ == "__main__":
    main()
