.. _examples-hermes:

===============================
Hermes Agent：自进化 AI Agent
===============================

Hermes Agent 由 Nous Research 于 2026 年 2 月发布，是一个**开源、自进化**
的 AI Agent 框架。与 OpenClaw 不同，Hermes 的核心哲学是"越用越强"——它拥有
跨会话的持久记忆，能在完成任务后自动沉淀为可复用的技能。

如果说 OpenClaw 是"一个能干活的好帮手"，那 Hermes Agent 就是"一个会
自己学习的员工"。它不只是执行指令，而是从每次交互中积累经验、优化行为、
创造可复用的能力。

六项核心技术
================

1. GEPA 自我进化引擎
------------------------------

Hermes 最独特的技术。GEPA 以类反向传播方式优化 Agent 的 prompt。
传统强化学习需要上万次评估才能收敛，GEPA 仅需 100-500 次即可完成
策略迭代。

.. code-block:: text

   GEPA 的进化循环：
   1. 行为记录 → Agent 执行任务并记录完整轨迹
   2. 效果评估 → 分析任务完成度和用户反馈
   3. 策略优化 → 调整 prompt 和行为策略
   4. 技能沉淀 → 将成功模式提炼为可复用的技能

2. 持久记忆架构
------------------------------

通过 MEMORY.md（环境事实）和 USER.md（用户偏好）两个文件实现跨会话记忆，
底层使用 SQLite FTS5 全文搜索 + LLM 摘要。

.. code-block:: text

   # MEMORY.md（自动维护）
   ## 项目信息
   - 当前正在开发：Hermes Agent 文档生成器
   - 常用模型：Claude Sonnet 4.5

   ## 用户偏好
   - 代码风格：使用 4 空格缩进
   - 日志级别：INFO

3. 技能自动学习
------------------------------

完成复杂任务后自动将方案提炼为可复用的 Skill 文件，遵循 agentskills.io
开放标准，可在社区共享。

.. code-block:: text

   # 用户：帮我分析一下这个项目的代码质量
   # Hermes 完成任务后自动生成技能文件

   → 分析完成，已自动创建技能：code-quality-review
   → 技能位置：~/.hermes/skills/code-quality-review.skill.md
   → 下次可直接使用：hermes --skill code-quality-review

技能文件的完整格式：

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

   ## Level 1（完整分析）
   - 包含：代码风格、复杂度、测试覆盖
   - 预计耗时：2 分钟

4. 200+ 模型零锁定
------------------------------

支持 Anthropic、OpenAI、DeepSeek、Hugging Face 等供应商。
一条命令 ``hermes model`` 即可切换，无需修改配置文件。

.. code-block:: bash

   # 查看可用模型
   hermes model list

   # 切换到 DeepSeek
   hermes model set deepseek-chat

   # 使用本地模型
   hermes model set http://localhost:11434/v1/qwen2.5:14b

5. 多平台网关
------------------------------

单网关同时接入 Telegram、Discord、Slack、WhatsApp、飞书、钉钉、企业微信。
用户可以在不同平台上和同一个 Hermes Agent 对话，记忆和技能保持同步。

.. code-block:: bash

   # 添加消息平台
   hermes platform add telegram --token YOUR_BOT_TOKEN
   hermes platform add discord --bot-token YOUR_BOT_TOKEN

   # 查看已接入平台
   hermes platform list
   # 输出：Telegram ✓ | Discord ✓ | Slack ✓（共 3 个平台在线）

6. 企业级安全
------------------------------

Docker 容器沙箱隔离、路径遍历防护、SSRF 缓解、凭证管理，至今零 CVE 记录。

.. code-block:: text

   安全机制：
   - 所有代码执行在 Docker 沙箱中
   - 文件系统访问限制在 ~/.hermes 目录
   - 网络请求经过白名单过滤
   - 敏感操作需要身份二次确认

OpenClaw vs Hermes 实战对比
===============================

.. list-table::
   :header-rows: 1

   * - 对比维度
     - OpenClaw
     - Hermes Agent
   * - 上手难度
     - 低，配置即用
     - 中等，第一次需配置学习偏好
   * - 技能系统
     - 5400+ 社区技能，安装即用
     - 技能需从使用中沉淀，但自动生成
   * - 记忆能力
     - Markdown 文件 + 向量搜索
     - SQLite FTS5 + LLM 摘要，更智能
   * - 自进化
     - 无（技能由社区维护）
     - GEPA 引擎自动优化行为
   * - 适用人群
     - 追求快速上手和丰富技能
     - 追求长期价值和个性化
   * - 消息平台
     - 20+ 平台支持
     - 15+ 平台支持
   * - 许可证
     - MIT
     - MIT

实际使用案例
==============

.. code-block:: text

   场景：每天早晨查看行业新闻

   OpenClaw 方案：
   1. 搜索 "newsletter" 技能并安装
   2. 在 agents.md 中配置每天早上 8:00 执行
   3. 完成

   Hermes Agent 方案：
   1. 告诉 Hermes "每天早上 8 点推送 AI 行业新闻"
   2. Hermes 创建定时任务并记住你的偏好
   3. 一周后，Hermes 自动学习了你的信息偏好
   4. 逐渐开始主动推荐你可能感兴趣的新闻

快速上手
============

.. code-block:: bash

   # 一键安装
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

   # 配置模型
   hermes model set claude-sonnet-4-20260514

   # 启动交互
   hermes

   # 初次使用时，Hermes 会通过对话了解你的偏好
   hermes > 你好，我是 Hermes。在开始之前，我想了解一下你的偏好。
            你主要使用哪种编程语言？
            你喜欢简洁还是详细的回答？
            你希望我主动推送信息还是等你问再回答？

MCP 集成
============

Hermes Agent 原生支持 MCP 协议，可连接任意 MCP 服务器扩展能力：

.. code-block:: bash

   # 连接 MCP 服务器
   hermes mcp add my-server --transport stdio --command "python mcp_server.py"

   # 查看已连接的 MCP 服务器
   hermes mcp list

.. admonition:: 如何选择？
   :class: tip

   - 想要"装好就能用"、有丰富技能库 → **OpenClaw**
   - 想要"越用越聪明"、长期个性化价值 → **Hermes Agent**
   - 很多高级用户两种都装：OpenClaw 处理日常任务，Hermes 处理需要深度理解的任务

参考资源
============

- GitHub: https://github.com/NousResearch/hermes-agent
- Skills Hub: https://agentskills.io
- 官方文档: https://hermes-agent.nousresearch.com
