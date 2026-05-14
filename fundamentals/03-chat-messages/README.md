# 03-chat-messages

这一组案例只讲一件事：**聊天模型为什么不只是接收一个普通字符串，而是接收“消息列表”。**

## 你要学会什么

- 什么是 `SystemMessage`
- 什么是 `HumanMessage`
- 为什么聊天模型更适合用消息结构表达上下文
- `system` 和 `human` 分别承担什么作用

## 这个案例的核心流程

1. 读取环境变量
2. 创建聊天模型对象
3. 构造消息列表
4. 把消息列表发给模型
5. 打印模型返回结果

## 你要特别注意

- `system` 不是用户提问，它更像“规则”或“角色说明”
- `human` 才是用户真正的问题
- 同样一个问题，如果 `system` 变化，模型回答风格也可能明显变化

## 为什么要学这个

因为后面你学：
- 多轮对话
- Tool Calling
- Agent
- Memory

本质上都会回到“消息列表”这个输入结构。

## 运行方式

在仓库根目录 `E:\langchain-learning` 下执行：

```powershell
.\.venv\Scripts\python fundamentals\03-chat-messages\demo.py
```

## 建议你自己改的地方

1. 改 `SystemMessage` 的角色设定
2. 改 `HumanMessage` 的问题
3. 删掉 `SystemMessage`，观察差异

## 这一组案例的目标

让你真正理解：

**聊天模型的输入不是一段普通文本，而是一组带角色语义的消息。**
