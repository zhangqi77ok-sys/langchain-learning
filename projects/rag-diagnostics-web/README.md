# rag-diagnostics-web

一个专门用于学习 RAG 内部过程的教学型 Web 项目。

## 这个项目解决什么问题

很多人在学 RAG 时，能看到“最后答案”，但看不清中间到底发生了什么：

- 为什么命中了这几个片段
- 为什么这个片段排第一
- 为什么这次会拒答
- 为什么推荐去看某个菜单

这个项目的目标，不是先做一个花哨产品，而是把 **RAG 内部过程打开给你看**。

## 你会在页面里看到什么

提交一个问题后，页面会展示：

1. 最终答案
2. 是否触发拒答
3. BM25 检索结果
4. 向量检索结果
5. 融合后的排序结果
6. 实际送进模型的上下文
7. 推荐菜单
8. 推荐原因

## 当前项目结构

- `app/main.py`
  后端主入口，包含最小混合检索、拒答判断和诊断信息返回

- `app/templates/index.html`
  前端页面，用来展示问题、答案和诊断细节

- `data/*.txt`
  教学用知识文本

- `requirements.txt`
  本项目依赖

## 运行方式

先安装依赖：

```powershell
.\.venv\Scripts\python -m pip install -r projects\rag-diagnostics-web\requirements.txt
```

进入项目目录后启动：

```powershell
cd E:\langchain-learning\projects\rag-diagnostics-web
..\..\.venv\Scripts\python -m uvicorn app.main:app --reload
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 这个项目和 `multi-kb-rag-web` 的区别

`multi-kb-rag-web` 更像一个正在逐步产品化的项目。

这个 `rag-diagnostics-web` 更像一个教学项目：

- 不追求页面花哨
- 优先把内部过程讲清楚
- 适合边看页面边理解检索链路

## 建议你怎么学这个项目

1. 先问一个明显相关的问题，比如“什么是 RAG”
2. 再看 BM25、向量检索、融合排序分别命中了什么
3. 再问一个明显无关的问题，比如披萨、天气之类
4. 观察为什么触发拒答
5. 最后修改 `data/` 里的知识内容，再重新测试
