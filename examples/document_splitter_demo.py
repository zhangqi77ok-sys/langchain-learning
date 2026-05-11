"""LangChain Document 与文本切分最小示例。"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    """构造一个文档并将长文本切成多个片段。"""
    document = Document(
        page_content=(
            "LangChain 是一个帮助开发者构建大模型应用的框架。"
            "它把提示词、模型调用、输出解析、检索、工具调用等能力组织起来。"
            "如果你直接调用模型 API，也能完成简单问答。"
            "但当应用开始变复杂，比如需要知识库、工作流或结构化输出时，"
            "LangChain 可以帮助你更系统地组织代码。"
        ),
        metadata={"source": "manual_note", "topic": "langchain_intro"},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=10,
    )
    chunks = splitter.split_documents([document])

    # 这里打印切分结果，目的是让你看清 Document 如何变成多个文本块。
    for index, chunk in enumerate(chunks, start=1):
        print(f"chunk {index}")
        print(f"content: {chunk.page_content}")
        print(f"metadata: {chunk.metadata}")
        print("-" * 20)


if __name__ == "__main__":
    main()
