# 学习目录重建设计

## 目标

把当前仓库的学习内容重建为按章节递进的结构，参考 `hello-agents` 的组织方式，不再使用零散的 `examples/`、`fundamentals/`、`advanced/` 目录。

## 已确认范围

- 已清空：
  - `examples/`
  - `fundamentals/`
  - `advanced/`
- 保留：
  - `projects/`
  - `roadmaps/`
  - 根目录配置

## 新的目录方案

新学习目录统一放在 `learning/` 下。

第一层章节目录采用“章节号 + 主题名”：

- `learning/chapter-01-basics`
- `learning/chapter-02-prompting`
- `learning/chapter-03-rag`
- `learning/chapter-04-memory`
- `learning/chapter-05-agents`
- `learning/chapter-06-hybrid-retrieval`
- `learning/chapter-07-langgraph`
- `learning/chapter-08-projects`

## 每章最小结构

每章默认只保留最小必要内容：

- `README.md`
- `demo.py`
- `data/`（仅在该章确实需要资料文件时才创建）

不额外引入多余抽象层。

## 教学组织原则

- 一章只讲一个明确主题
- 每章先有 `README.md` 解释“为什么学、学什么、怎么运行”
- 再有一个最小可运行示例
- 后续章节在前一章基础上递进

## 第一阶段实现顺序

先只做最小起步，不一次铺满所有章节。

第一步：

- 创建 `learning/`
- 创建 `learning/chapter-01-basics/`
- 写 `README.md`
- 写 `demo.py`

## 当前决策

- 目录命名采用“章节号 + 主题名”
- 教学形式采用“章节式递进”
- 继续保留 `projects/` 作为项目级练习区

## 风险与边界

- 当前只重建学习结构，不修改 `projects/`
- 当前不恢复旧案例内容，只按新结构逐章重建
- 当前不一次创建全部章节文件，只先创建框架和第一章
