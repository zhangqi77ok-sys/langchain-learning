# multi-kb-rag-web

一个基于 FastAPI 的多知识库问答 Web 项目。

## 目标

- 提供一个网页问答界面
- 从本地多文件知识库中检索相关内容
- 基于检索结果调用大模型生成答案
- 展示答案和引用来源

## 当前范围

- 项目骨架已可运行
- 已具备最小 Web 问答能力
- 已接入最小混合检索：BM25 + 向量检索融合
- 后续会继续强化检索、对话和流程编排能力

## 当前检索链路

现在 `/ask` 的检索流程是：

1. 读取 `data/` 下的本地文本
2. 做文本切分
3. 同时执行两路检索
   - BM25：看关键词
   - 向量检索：看语义
4. 把两路结果做最小融合
5. 取前几个片段拼成上下文
6. 交给模型生成最终回答

## 运行方式

安装依赖：

```powershell
.\.venv\Scripts\python -m pip install -r projects\multi-kb-rag-web\requirements.txt
```

启动服务：

```powershell
.\.venv\Scripts\python -m uvicorn projects.multi-kb-rag-web.app.main:app --reload
```

如果上面的模块路径在当前环境下不可用，也可以直接进入项目目录后启动：

```powershell
cd E:\langchain-learning\projects\multi-kb-rag-web
..\..\.venv\Scripts\python -m uvicorn app.main:app --reload
```

打开浏览器后访问：

```text
http://127.0.0.1:8000
```
