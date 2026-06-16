.. _examples-hermes:

===============================
Hermes Agent：自进化 AI Agent
===============================

Hermes Agent 由 Nous Research 于 2026 年 2 月发布，是一个**开源、自进化**
的 AI Agent 框架。与 OpenClaw 不同，Hermes 的核心哲学是"越用越强"——它拥有
跨会话的持久记忆，能在完成任务后自动沉淀为可复用的技能。

核心理念
============

.. list-table::
   :header-rows: 1

   * - 维度
     - Hermes Agent
     - 传统 AI 工具
   * - 记忆
     - 跨会话持久记忆
     - 每次对话从零开始
   * - 技能
     - 自动创建 + 自我改进
     - 需手动编写
   * - 学习
     - GEPA 自我进化引擎
     - 无持续学习机制
   * - 部署
     - 本地 / VPS / Serverless
     - 云端绑定
   * - 模型
     - 200+ 模型自由切换
     - 单一供应商锁定

六项核心技术
================

1. **GEPA 自我进化引擎**
   以类反向传播方式优化 prompt。传统 RL 需上万次评估，GEPA 仅 100-500 次即可完成
   策略迭代：行为记录 → 效果评估 → 策略优化 → 技能沉淀。

2. **持久记忆架构**
   通过 MEMORY.md（环境事实）和 USER.md（用户偏好）两个文件实现跨会话记忆，
   底层使用 SQLite FTS5 全文搜索 + LLM 摘要。

3. **技能自动学习**
   完成复杂任务后自动将方案提炼为可复用的 Skill 文件（SKILL.md 格式），
   遵循 agentskills.io 开放标准，可在社区共享。

4. **200+ 模型零锁定**
   支持 Anthropic、OpenAI、DeepSeek、Hugging Face 等供应商。
   一条命令 ``hermes model`` 即可切换，零代码改动。

5. **15+ 消息平台**
   单网关同时接入 Telegram、Discord、Slack、WhatsApp、飞书、钉钉、企业微信等。

6. **企业级安全**
   Docker 容器沙箱隔离、路径遍历防护、SSRF 缓解、凭证管理，至今零 CVE。

快速上手
============

.. code-block:: bash

   # 一键安装
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

   # 配置模型（支持 OpenRouter / 本地 vLLM / 自定义端点）
   hermes model

   # 启动交互
   hermes

技能自动生成演示
====================

.. code-block:: text

   # 用户：帮我分析一下这个项目的代码质量
   # Hermes 完成任务后自动生成技能文件

   → 分析完成，已自动创建技能：code-quality-review
   → 技能位置：~/.hermes/skills/code-quality-review.skill.md
   → 下次可直接使用：hermes --skill code-quality-review

技能文件内容示例（SKILL.md）：

.. code-block:: markdown

   # Code Quality Review Skill

   ## Trigger
   当用户要求分析代码质量时自动匹配

   ## Workflow
   1. 克隆/读取目标仓库
   2. 扫描目录结构和关键文件
   3. 检查：错误处理、重复代码、类型注解、文档覆盖
   4. 生成包含严重等级的报告

   ## Level 0（快速预览）
   - 检查项数量：12
   - 预计耗时：30 秒

MCP 集成
============

Hermes Agent 原生支持 MCP 协议，可连接任意 MCP 服务器扩展能力：

.. code-block:: bash

   # 连接 MCP 服务器
   hermes mcp add my-server --transport stdio --command "python mcp_server.py"

.. admonition:: Hermes vs OpenClaw
   :class: tip

   - **OpenClaw** 强在 IM 集成和技能市场的丰富度（5400+ 技能），上手即用
   - **Hermes Agent** 强在自我进化和技能自动生成，越用越聪明
   - 两者都开源（MIT 协议），可根据需求选择或配合使用

参考资源
============

- GitHub: https://github.com/NousResearch/hermes-agent
- Skills Hub: https://agentskills.io
- 官方文档: https://hermes-agent.nousresearch.com
