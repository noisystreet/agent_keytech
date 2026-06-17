.. _examples-openclaw:

===============================
OpenClaw：开源通用 AI Agent
===============================

OpenClaw（原名 Clawdbot → Moltbot → OpenClaw）是 2025 年末由奥地利开发者
Peter Steinberger 创建的开源 AI Agent 框架，**60 天内斩获 35 万+ GitHub Stars** ，
成为 2026 年最受关注的开源项目。它的成功不是偶然——OpenClaw 抓住了一个
核心需求："让 AI 帮你做事，而不仅仅陪你聊天。"

和其他 AI 工具不同，OpenClaw 定位为**后台运行的智能助理**——你通过消息
应用跟它说话，它在后台 7×24 运行，能读写文件、操控浏览器、执行代码、
定时调度。

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

Heartbeat 机制详解
====================

OpenClaw 最突出的技术特性是它的 Heartbeat（心跳）机制。常规的 AI 助手
只在用户输入时响应，而 OpenClaw 能在后台持续运行、主动执行任务。

.. code-block:: python

   # Heartbeat 的工作机制（简化版）
   class OpenClawHeartbeat:
       """
       OpenClaw 的 Heartbeat 每 30 分钟触发一次。
       Agent 检查是否有待办任务、是否需要执行定时操作。
       """
       def __init__(self, agent):
           self.agent = agent
           self.scheduled_tasks = []

       def heartbeat_cycle(self):
           """一次心跳周期"""
           # 检查定时任务
           for task in self.scheduled_tasks:
               if task.should_run():
                   result = self.agent.run(task.instruction)
                   task.mark_done(result)

           # 检查待处理消息
           pending = self.agent.check_messages()
           for msg in pending:
               self.agent.process(msg)

           # 自我状态检查
           self.agent.check_health()

配置文件驱动
================

OpenClaw 通过多个 Markdown 配置文件定义 Agent 的行为和个性。这种
"配置即行为"的设计让非开发者也能量身定制自己的 Agent。

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

   # agents.md（示例）
   ## 工具权限
   - search: 允许
   - shell: 需要确认
   - file_delete: 禁止

   ## 调度任务
   - 每天早上 8:00 推送新闻摘要
   - 每周五 18:00 生成周报

Skill 生态与 ClawHub
=======================

OpenClaw 的技能系统通过 SKILL.md 文件定义，社区贡献的技能托管在 ClawHub 上。
截至 2026 年中，ClawHub 已有 5400+ 个社区技能。

.. code-block:: text

   # 一个典型的 SKILL.md 文件（Web 抓取技能）
   ## Name
   web_scraper

   ## Trigger
   当用户要求"抓取网页"、"提取页面内容"时触发

   ## Dependencies
   - requests
   - beautifulsoup4

   ## Workflow
   1. 用户提供 URL
   2. 发送 HTTP 请求获取页面
   3. 解析 HTML 提取正文内容
   4. 格式化输出结果

   ## Safety
   - 只允许 HTTP/HTTPS 协议
   - 超时设置：10 秒
   - 最大响应体：5MB

安装技能的方式也很简单——在聊天中直接告诉 OpenClaw 即可：

.. code-block:: text

   用户: "安装 newsletter 技能"
   OpenClaw: "正在从 ClawHub 安装 newsletter...
              已安装成功。技能说明：帮你生成和管理邮件通讯。"

部署方式
============

.. code-block:: bash

   # 一键安装
   curl -fsSL https://get.openclaw.sh | bash

   # 配置消息平台（以 Telegram 为例）
   openclaw config set platform telegram --token YOUR_BOT_TOKEN

   # 启动
   openclaw start

   # 查看运行状态
   openclaw status
   # 输出：运行中 | 消息平台: Telegram, Discord | 上次心跳: 2分钟前

安全注意事项
================

.. admonition:: OpenClaw 的安全风险
   :class: warning

   OpenClaw 拥有完整的本地文件系统和浏览器访问权限。这是它强大的原因，
   也是它危险的原因。生产环境中务必注意：
   - 限制 Agent 的文件访问范围（通过 agents.md 配置）
   - 敏感操作（文件删除、代码执行）设置二次确认
   - 不要在多用户环境中共享 API Key
   - 定期审查 memory.md 中存储的敏感信息
   - 将 OpenClaw 运行在 Docker 容器中隔离

OpenClaw 适用的场景
=====================

.. list-table::
   :header-rows: 1

   * - 场景
     - 是否推荐
     - 说明
   * - 个人信息助手
     - 强烈推荐
     - 收集信息、定时推送、文件管理
   * - 客服机器人
     - 推荐
     - 多平台接入 + 知识库检索
   * - 代码开发辅助
     - 不推荐
     - Cursor / Claude Code 更适合
   * - 自动化测试
     - 推荐
     - 定时运行测试 + 结果推送
   * - 企业内部工具
     - 推荐
     - 配合安全策略部署

参考资源
============

- GitHub: https://github.com/openclaw/openclaw
- ClawHub（技能市场）: https://clawhub.io
- 官方文档: https://docs.openclaw.dev
