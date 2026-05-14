# fundamentals

这一目录用于做 LangChain 基础巩固。

目标不是继续横向堆功能，而是把核心基础点拆开，一组一组讲清楚：
- 它解决什么问题
- 它和前后知识点是什么关系
- 在真实项目里会怎么用

## 推荐学习顺序

### 01. model-call

- 目录：[`01-model-call`](</E:/langchain-learning/fundamentals/01-model-call>)
- 解决问题：最小模型调用怎么跑通
- 你要建立的认知：LangChain 最底层先是“把一次模型调用组织清楚”

### 02. prompt-template

- 目录：[`02-prompt-template`](</E:/langchain-learning/fundamentals/02-prompt-template>)
- 解决问题：为什么提示词要模板化，而不是一直写死字符串
- 你要建立的认知：提示词 = 固定骨架 + 动态变量

### 03. chat-messages

- 目录：[`03-chat-messages`](</E:/langchain-learning/fundamentals/03-chat-messages>)
- 解决问题：聊天模型为什么接收消息列表，而不是普通字符串
- 你要建立的认知：`system` 和 `human` 是不同角色，不是同一层信息

### 04. output-parser

- 目录：[`04-output-parser`](</E:/langchain-learning/fundamentals/04-output-parser>)
- 解决问题：为什么模型输出有时不能直接给程序用
- 你要建立的认知：parser 的价值是把输出整理成程序更好处理的形式

### 05. structured-output

- 目录：[`05-structured-output`](</E:/langchain-learning/fundamentals/05-structured-output>)
- 解决问题：为什么有时不能只要文本，而必须拿到固定字段
- 你要建立的认知：结构化输出更适合 API、流程判断和程序继续处理

### 06. document-and-splitter

- 目录：[`06-document-and-splitter`](</E:/langchain-learning/fundamentals/06-document-and-splitter>)
- 解决问题：知识资料进入 RAG 前为什么要先封装和切分
- 你要建立的认知：RAG 的第一步不是问模型，而是先把资料整理成适合检索的形态

### 07. retriever-basics

- 目录：[`07-retriever-basics`](</E:/langchain-learning/fundamentals/07-retriever-basics>)
- 解决问题：为什么问题和文档都要向量化，再做相似度检索
- 你要建立的认知：检索阶段的输出是候选资料，不是最终答案

### 08. rag-basics

- 目录：[`08-rag-basics`](</E:/langchain-learning/fundamentals/08-rag-basics>)
- 解决问题：检索和生成是怎么拼成最小 RAG 闭环的
- 你要建立的认知：RAG = 检索 + 上下文组织 + 基于上下文生成

### 09. tool-calling

- 目录：[`09-tool-calling`](</E:/langchain-learning/fundamentals/09-tool-calling>)
- 解决问题：模型为什么会先请求调用工具，而不是直接回答
- 你要建立的认知：Tool Calling 是“模型先决定，系统再执行工具”

### 10. agent-basics

- 目录：[`10-agent-basics`](</E:/langchain-learning/fundamentals/10-agent-basics>)
- 解决问题：Agent 和普通 Tool Calling 的本质区别是什么
- 你要建立的认知：Agent 的关键价值是自动多步决策

### 11. memory-basics

- 目录：[`11-memory-basics`](</E:/langchain-learning/fundamentals/11-memory-basics>)
- 解决问题：多轮对话为什么能“记住”前文
- 你要建立的认知：Memory 很多时候本质是历史消息再次进入模型输入

### 12. streaming-basics

- 目录：[`12-streaming-basics`](</E:/langchain-learning/fundamentals/12-streaming-basics>)
- 解决问题：为什么 `invoke()` 和 `stream()` 的差异更多是返回方式
- 你要建立的认知：streaming 主要改善用户体验，不是让模型突然更强

## 这些基础点之间的关系

可以把它们看成 4 层：

1. 模型输入层
- 01 `model-call`
- 02 `prompt-template`
- 03 `chat-messages`

2. 模型输出层
- 04 `output-parser`
- 05 `structured-output`

3. RAG 数据层
- 06 `document-and-splitter`
- 07 `retriever-basics`
- 08 `rag-basics`

4. 任务编排层
- 09 `tool-calling`
- 10 `agent-basics`
- 11 `memory-basics`
- 12 `streaming-basics`

## 学完这一轮后你应该能回答的问题

- 最小模型调用怎么写
- 为什么 prompt 要模板化
- 为什么聊天模型要用消息列表
- parser 和 structured output 的区别
- `Document`、切分、检索分别在 RAG 里做什么
- Tool Calling 和 Agent 的区别
- Memory 的本质是什么
- Streaming 到底改变了什么

## 建议的复习方式

1. 每组先看 `README.md`
2. 再运行 `demo.py`
3. 然后自己改一个参数或输入
4. 最后试着不用看代码，自己解释“这一组到底解决了什么问题”

## 当前阶段的目标

不是“多写几个功能”，而是把基础认知打扎实。

如果这 12 组你都能自己讲清楚，后面再做 LangGraph、项目级 RAG、Web 化、多工具系统，就会稳很多。
