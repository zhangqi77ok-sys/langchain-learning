# 10-agent-basics

这一组案例只讲一件事：**Agent 和普通 Tool Calling 的本质区别是什么。**

## 你要学会什么

- 什么是 Agent
- Agent 和 Tool Calling 的区别
- 为什么 Agent 可以自动循环调用工具
- Agent 为什么更适合多步骤任务

## 这个案例的核心流程

1. 定义多个本地工具
2. 创建 Agent
3. 给 Agent 一个问题
4. Agent 自己决定要不要调用工具、调用哪个工具
5. Agent 根据工具结果继续推进，直到给出最终答案

## 你要特别注意

- Tool Calling 更像“模型触发一次工具，再回到模型”
- Agent 更像“模型在一个循环里不断判断下一步要做什么”
- Agent 的核心不是“更高级”，而是“更适合多步决策”

## 为什么要学这个

因为后面你做：
- 多工具协作
- 多步骤信息整合
- 任务拆解
- 工作流型助手

都会从普通 Tool Calling 走向 Agent。

## 运行方式

在仓库根目录 `E:\langchain-learning` 下执行：

```powershell
.\.venv\Scripts\python fundamentals\10-agent-basics\demo.py
```

## 建议你自己改的地方

1. 改提问城市
2. 改工具内容
3. 改成只需要一个工具的问题
4. 观察 Agent 是否仍然会调用多个工具

## 这一组案例的目标

让你真正理解：

**Agent 的价值不在于“会调工具”本身，而在于“会根据中间结果继续决策下一步”。**
