# 12-streaming-basics

这一组案例只讲一件事：**为什么 `invoke()` 和 `stream()` 的差别不在“模型更强了”，而在“结果返回方式变了”。**

## 你要学会什么

- 什么是流式输出
- `invoke()` 和 `stream()` 的区别
- 为什么聊天页面常常需要 streaming
- 为什么 streaming 更像“体验优化”，不是“模型能力升级”

## 这个案例的核心流程

1. 读取环境变量
2. 创建聊天模型对象
3. 调用 `model.stream(...)`
4. 按 chunk 逐步打印内容

## 你要特别注意

- 模型本身没有变
- 问题本身也没有变
- 变化的是：你不再等完整结果一次性返回，而是边生成边拿结果

## 为什么要学这个

因为后面你做：
- Web 聊天页面
- 终端助手
- 较长回答展示
- 更自然的用户体验

都会需要 streaming。

## 运行方式

在仓库根目录 `E:\langchain-learning` 下执行：

```powershell
.\.venv\Scripts\python fundamentals\12-streaming-basics\demo.py
```

## 建议你自己改的地方

1. 改问题内容
2. 改成更长的问题
3. 和 `invoke()` 的体验做对比

## 这一组案例的目标

让你真正理解：

**streaming 的价值主要在用户体验层，而不是让模型突然变聪明。**
