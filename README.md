# langchain-learning

这是一个按章节递进学习 LangChain、RAG、Agent 和项目实践的仓库。

## 当前结构

- `learning/`
  新的章节式学习区，按章节一步步学习。
- `projects/`
  项目级练习区，用来放完整的小项目和 Web 化实践。
- `roadmaps/`
  学习方向和后续规划记录。

## 学习入口

如果你现在是从头开始学，先看：

- [`learning/README.md`](</E:/langchain-learning/learning/README.md>)

如果你要看项目级练习，去看：

- [`projects/multi-kb-rag-web`](</E:/langchain-learning/projects/multi-kb-rag-web>)

## 当前说明

- 旧的 `examples/`、`fundamentals/`、`advanced/` 学习目录已废弃
- 后续内容统一重建到 `learning/`
- 现有项目目录继续保留，不受这次学习结构调整影响

## 环境准备

安装依赖：

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

配置根目录 `.env`：

```env
OPENAI_API_KEY=你的第三方key
OPENAI_BASE_URL=https://mikasa.icu/v1
OPENAI_MODEL=gpt-5.4
```
