# 06-document-and-splitter

这一组案例只讲一件事：**为什么 RAG 不是直接把整篇文档塞给模型，而是先变成 Document，再做切分。**

## 你要学会什么

- 什么是 `Document`
- 为什么文档不仅有正文，还有 `metadata`
- 为什么长文本不能直接整段拿去检索
- `chunk_size` 和 `chunk_overlap` 分别影响什么

## 这个案例的核心流程

1. 创建一个 `Document`
2. 给它正文和元数据
3. 创建文本切分器
4. 把长文本切成多个文本块
5. 打印每个文本块和对应元数据

## 你要特别注意

- `Document` 是 RAG 里最基础的数据单位
- `metadata` 很重要，因为后面来源追踪、引用展示、权限控制都可能用到它
- 切分不是可有可无，它直接影响后面检索质量

## 为什么要学这个

因为后面你做：
- 本地知识库
- 多文件 RAG
- 来源引用
- 检索优化

都离不开“文档如何组织”和“文本如何切分”。

## 运行方式

在仓库根目录 `E:\langchain-learning` 下执行：

```powershell
.\.venv\Scripts\python fundamentals\06-document-and-splitter\demo.py
```

## 建议你自己改的地方

1. 改 `chunk_size`
2. 改 `chunk_overlap`
3. 改长文本内容
4. 改 `metadata`

## 这一组案例的目标

让你真正理解：

**RAG 的第一步不是“问模型”，而是先把知识资料整理成适合检索的形态。**
