# Agents 指南

本项目是一个 Sphinx 文档教程，主题为"大模型 Agent 核心技术"，使用 reStructuredText（.rst）编写。以下是 AI 助手在编辑本项目时应遵守的规则。

## 项目概览

- **构建工具**：Sphinx + sphinx-rtd-theme + sphinxcontrib-mermaid
- **构建命令**：`make html`（本地预览 `make serve`）
- **文档语言**：中文（zh-cn）
- **源文件格式**：reStructuredText（.rst）
- **源目录**：`source/`

## 写作风格

1. **用大白话讲透技术概念**：避免学术论文式的干瘪叙述。先给出直觉类比或实际场景，再做技术拆解。多用"为什么"、"但这里有一个容易被忽略的点"这类过渡来引导读者思考。

2. **代码示例要完整可运行**：每段代码都应是实际可用的 Python 片段，包含必要的 import 和注释。注释用中文写，解释"为什么这么做"而非"这段代码在做什么"。

3. **讲解 > 知识点罗列**：不要只列 bullet points 和结论。每个知识点要展开讲解其原理、工程权衡和常见误区。多用 `.. admonition::` 来放"坑"和"经验"。

4. **站在 Agent 开发者视角**：所有技术点的讲解最终要落到"这对 Agent 开发者意味着什么"。比如讲 Attention 时，重点不是数学公式，而是 Lost in the Middle 如何影响 Agent 的提示词设计。

5. **用对比表格整理核心差异**：对多方案对比（如不同规划策略、不同检索方法），用 `.. list-table::` 列出维度对比，帮助读者决策。

## 文件结构规范

- 每章一个目录 `source/chapter_XX_xxx/`
- 每小节一个文件 `source/chapter_XX_xxx/0N_topic.rst`
- 章节目录在 `index.rst` 中用 `.. toctree::` 组织
- 附录在 `source/appendix/`
- 产品介绍在 `source/examples/`

## 编辑注意事项

1. **不要创建新文件**除非绝对必要。优先编辑已有的 .rst 文件。
2. **不要创建 Markdown（.md）文件**，除非 README.md 或 agents.md 这类项目级配置。
3. **不要新增文档化文件**（README、使用说明等），除非用户明确要求。
4. **避免过度工程**：不要添加与当前任务无关的功能、重构或注释。
5. **不要添加 docstring 或注释**到未修改过的代码中。
6. **Git 提交必须通过 pre-commit hook**，禁止使用 `--no-verify`。
7. 代码块用 `.. code-block:: python` 等 directive，不要用 Markdown 代码块语法。
8. Mermaid 图表用 `.. mermaid::` directive。

## 常见 directive 速查

```
.. admonition:: 标题
   :class: tip | caution | warning | note

.. code-block:: python

.. list-table::
   :header-rows: 1

.. mermaid::

.. code-block:: bash
