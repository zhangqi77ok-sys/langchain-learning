# langchain-learning

一个用于从 0 到 1 学习 LangChain 的实验仓库。

## 学习目标

- 理解 LangChain 在大模型应用中的作用
- 跑通最小聊天、Prompt、结构化输出、RAG、Tool Calling、Agent、Memory、Streaming 示例
- 用最少代码建立完整认知路径

## 推荐学习顺序

1. 基础调用
2. Prompt 与消息结构
3. 输出解析与结构化输出
4. RAG 基础
5. Tool Calling
6. Agent
7. Memory
8. Streaming

## 示例目录

### 1. 基础调用

- [`examples/hello_langchain.py`](</E:/langchain-learning/examples/hello_langchain.py>)：最小聊天模型调用

### 2. Prompt 与消息结构

- [`examples/prompt_template_demo.py`](</E:/langchain-learning/examples/prompt_template_demo.py>)：单条提示词模板
- [`examples/chat_messages_demo.py`](</E:/langchain-learning/examples/chat_messages_demo.py>)：`system` + `human` 消息结构
- [`examples/chat_prompt_template_demo.py`](</E:/langchain-learning/examples/chat_prompt_template_demo.py>)：聊天消息模板化

### 3. 输出解析与结构化输出

- [`examples/output_parser_demo.py`](</E:/langchain-learning/examples/output_parser_demo.py>)：把模型结果解析成纯字符串
- [`examples/structured_output_demo.py`](</E:/langchain-learning/examples/structured_output_demo.py>)：让模型按固定字段返回结果

### 4. RAG 基础

- [`examples/document_splitter_demo.py`](</E:/langchain-learning/examples/document_splitter_demo.py>)：`Document` 与文本切分
- [`examples/vector_retriever_demo.py`](</E:/langchain-learning/examples/vector_retriever_demo.py>)：本地 embedding + 向量检索
- [`examples/rag_qa_demo.py`](</E:/langchain-learning/examples/rag_qa_demo.py>)：最小 RAG 问答闭环

### 5. Tool Calling

- [`examples/tool_calling_demo.py`](</E:/langchain-learning/examples/tool_calling_demo.py>)：模型触发工具并读取工具结果

### 6. Agent

- [`examples/agent_demo.py`](</E:/langchain-learning/examples/agent_demo.py>)：Agent 自动决定调用哪些工具

### 7. Memory

- [`examples/memory_demo.py`](</E:/langchain-learning/examples/memory_demo.py>)：多轮对话历史示例

说明：当前示例用于理解记忆机制，运行可用，但底层 `RunnableWithMessageHistory` 已出现弃用告警，后续建议单独学习 LangGraph persistence。

### 8. Streaming

- [`examples/streaming_demo.py`](</E:/langchain-learning/examples/streaming_demo.py>)：流式输出最小示例

## 环境准备

1. 创建虚拟环境
2. 安装依赖
3. 配置 `.env`

### 安装依赖

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### `.env` 示例

```env
OPENAI_API_KEY=你的第三方key
OPENAI_BASE_URL=https://mikasa.icu/v1
OPENAI_MODEL=gpt-5.4
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 运行示例

例如运行最小 RAG 示例：

```powershell
.\.venv\Scripts\python examples\rag_qa_demo.py
```

## 当前说明

- 聊天模型使用第三方 OpenAI 兼容接口
- 向量检索使用本地 `sentence-transformers`
- 示例以“学习机制”为优先，不追求生产级封装
