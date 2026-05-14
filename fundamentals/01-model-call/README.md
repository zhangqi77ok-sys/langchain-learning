# 01-model-call

这一组案例只讲一件事：**怎么用 LangChain 完成一次最小模型调用**。

## 你要学会什么

- 什么是 `ChatOpenAI`
- 什么是最小输入
- 什么是最小输出
- 为什么 `.env` 里的模型配置会影响调用

## 这个案例的核心流程

1. 读取环境变量
2. 创建聊天模型对象
3. 传入一个问题
4. 拿到模型返回结果
5. 打印结果

## 你要特别注意

- 这里还没有 PromptTemplate
- 这里还没有消息列表
- 这里还没有 RAG、Tool、Agent
- 这里只是最基础的“模型能不能调通”

## 运行方式

在仓库根目录 `E:\langchain-learning` 下执行：

```powershell
.\.venv\Scripts\python fundamentals\01-model-call\demo.py
```

## 建议你自己改的地方

1. 改提问内容
2. 改 `OPENAI_MODEL`
3. 故意删掉 `OPENAI_API_KEY`，观察报错

## 这一组案例的目标

不是“做功能”，而是让你彻底理解：

**LangChain 最基础的一层，就是把一次模型调用组织成更清晰的代码。**
