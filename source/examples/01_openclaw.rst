.. _examples-openclaw:

===============================
OpenClaw：开源通用 AI Agent
===============================

OpenClaw（原名 Clawdbot → Moltbot → OpenClaw）是 2025 年末由奥地利开发者
Peter Steinberger 创建的开源 AI Agent 框架，**60 天内斩获 35 万+ GitHub Stars**，
成为 2026 年最受关注的开源项目。

核心理念
============

OpenClaw 是一个让 LLM 拥有"身体"的框架——它不再是聊天框里的文字生成器，
而是一个能直接操作你电脑的智能体。

.. mermaid::

   flowchart LR
       User[用户] --> IM[微信/飞书/Telegram/WhatsApp]
       IM --> Gateway[OpenClaw Gateway]
       Gateway --> Brain[Agent 核心引擎]
       Brain --> Tools[工具层]
       Tools --> Files[文件系统]
       Tools --> Browser[浏览器]
       Tools --> Shell[Shell 命令]
       Tools --> Code[代码执行]
       Brain --> Memory[持久记忆<br>Markdown + 向量库]
       Brain --> Skills[Skill 技能生态<br>5400+ 社区插件]

核心特性
============

.. list-table::
   :header-rows: 1

   * - 特性
     - 说明
   * - 7×24 后台运行
     - Heartbeat 心跳机制 + Cron 调度，即使无人交互也能主动执行任务
   * - 多消息平台
     - Telegram、WhatsApp、微信、飞书、钉钉、Discord、Slack 等 20+ 平台
   * - 本地文件访问
     - 完整的文件读写能力，管理本地文档和项目
   * - 浏览器自动化
     - 基于 Chrome DevTools Protocol，支持已登录态操作
   * - 技能生态
     - ClawHub 社区提供 5400+ 可安装的 Skill 插件
   * - 持久记忆
     - 跨会话的 Markdown 文件 + 向量数据库记忆系统
   * - 多模型支持
     - 可接入 Claude、GPT-4、DeepSeek、Qwen 等主流模型

配置文件驱动
================

OpenClaw 通过多个 Markdown 配置文件定义 Agent 的行为和个性：

- **soul.md** — Agent 的性格和身份定义
- **agents.md** — 行为规则和工作模式
- **identity.md** — 身份信息
- **user.md** — 用户偏好记录
- **memory.md** — 长期记忆存储

.. code-block:: text

   # soul.md（示例）
   ## 身份
   你是一个高效的个人助手，名叫"小爪"。

   ## 行为规则
   1. 回答简洁，不要废话
   2. 使用工具前先确认用户意图
   3. 敏感操作需要用户二次确认

部署方式
============

.. code-block:: bash

   # 一键安装
   curl -fsSL https://get.openclaw.sh | bash

   # 配置消息平台（以 Telegram 为例）
   openclaw config set platform telegram --token YOUR_BOT_TOKEN

   # 启动
   openclaw start

安全注意事项
================

.. warning::

   OpenClaw 拥有完整的本地文件系统和浏览器访问权限。生产环境中务必注意：
   - 限制 Agent 的文件访问范围
   - 敏感操作（文件删除、代码执行）设置二次确认
   - 不要在多用户环境中共享 API Key
   - 定期审查 memory.md 中存储的敏感信息

参考资源
============

- GitHub: https://github.com/openclaw/openclaw
- ClawHub（技能市场）: https://clawhub.io
- 官方文档: https://docs.openclaw.dev
