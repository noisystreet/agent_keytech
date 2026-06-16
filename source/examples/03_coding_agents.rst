.. _examples-coding-agents:

===============================
AI 编程 Agent：Claude Code vs Codex vs Cursor
===============================

2026 年 AI 编程工具进入"Agent 自治"时代。Claude Code、OpenAI Codex、Cursor
三分天下，各自代表了**终端原生 / 模型原生 / IDE 集成** 三条不同的技术路线。

全景对比
============

.. list-table::
   :header-rows: 1

   * - 维度
     - Claude Code
     - OpenAI Codex
     - Cursor
   * - 出品方
     - Anthropic
     - OpenAI
     - Anysphere
   * - 定位
     - 终端原生 Agent
     - 云端异步 Agent
     - AI 原生 IDE
   * - 运行方式
     - CLI 终端 / VS Code 扩展
     - 云端沙箱 / VS Code 扩展
     - 独立 IDE（基于 VS Code）
   * - 核心模型
     - Claude 系列（Sonnet/Opus/Haiku）
     - GPT-5.5 / GPT-5.3-codex
     - 多模型（支持 Claude/GPT/Gemini 等）
   * - 上下文
     - 200K tokens
     - 云端沙箱
     - 依赖 IDE 窗口
   * - 工作模式
     - 自主代理，端到端执行
     - 异步委托，完成后通知
     - Composer / Agent / Ask / Tab
   * - 开源
     - 否
     - 否
     - 否
   * - 价格
     - \$20-200/月
     - \$20/月（ChatGPT Plus）
     - \$20/月（Pro）

Claude Code：终端原生 Agent
===============================

**核心优势：对代码库的深度理解**

Claude Code 选择终端作为落脚点，直接访问文件系统、Git 工作流和整个代码库的拓扑结构。

.. code-block:: text

   # Claude Code 工作流示例
   $ claude "帮我重构 auth 模块，提取公共逻辑，更新所有引用"

   1. 读取 auth/ 目录结构 → 理解模块依赖
   2. 识别公共逻辑 → 提取到 shared/auth_utils.py
   3. 更新所有 import 引用 → 自动修复
   4. 运行测试 → 验证重构正确性
   5. 提交 PR → 生成变更说明

**适合谁：** 喜欢终端工作流的后端/全栈开发者，需要深度理解代码库的复杂任务。

.. code-block:: python

   # 使用 Claude Code API 进行代码审查
   import anthropic

   client = anthropic.Anthropic()

   response = client.messages.create(
       model="claude-sonnet-4-20260514",
       max_tokens=8192,
       tools=[{
           "name": "review_code",
           "description": "审查 pull request 的代码变更",
           "input_schema": {
               "type": "object",
               "properties": {
                   "diff": {"type": "string"},
                   "language": {"type": "string"}
               },
               "required": ["diff"]
           }
       }],
       messages=[{"role": "user", "content": "审查这个 PR 的代码质量"}]
   )

OpenAI Codex：云端异步 Agent
===============================

**核心优势：异步并行，无需本地环境**

Codex 运行在云端沙箱中，可同时分配给多个任务，完成后返回 diff 或 PR。

.. code-block:: text

   # Codex 工作流示例
   1. 用户："实现用户注册功能，包含邮箱验证"
   2. Codex 在云端沙箱中拉起代码仓库
   3. 自动完成：数据库模型 → API 路由 → 业务逻辑 → 测试用例
   4. 提交 PR，通知用户审查

**适合谁：** 团队协作场景，需要批量处理明确定义的任务，不依赖本地 IDE。

.. code-block:: text

   # Codex 的多任务并行
   $ codex task "为 user 模块添加单元测试"
   $ codex task "重构 payment 模块的错误处理"
   $ codex task "更新 API 文档"
   # 三个任务在云端并行执行，完成后统一通知

Cursor：AI 原生 IDE
===============================

**核心优势：可视化体验，零学习成本**

Cursor 基于 VS Code fork，核心卖点是 Composer 模式——一句话描述需求，
自动跨文件生成代码。

.. code-block:: text

   # Cursor Composer 示例
   用户输入："添加一个深色模式切换按钮"

   1. Cursor 理解需求 → 定位相关组件
   2. 创建 ThemeContext → 封装主题状态
   3. 修改布局文件 → 添加切换按钮
   4. 更新全局样式 → 支持 CSS 变量切换
   5. 实时预览 → 用户立即看到效果

**适合谁：** 前端/全栈开发者，习惯 GUI 编辑器的用户。

.. mermaid::

   flowchart LR
       subgraph Claude Code [终端原生]
           A1[终端命令] --> A2[文件读写]
           A2 --> A3[Git 操作]
           A3 --> A4[测试运行]
       end
       subgraph Codex [云端原生]
           B1[自然语言描述] --> B2[云端沙箱]
           B2 --> B3[代码生成]
           B3 --> B4[PR 提交]
       end
       subgraph Cursor [IDE 原生]
           C1[编辑器操作] --> C2[Composer]
           C2 --> C3[多文件编辑]
           C3 --> C4[实时预览]
       end

如何选择？
============

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐工具
     - 原因
   * - 深度重构 / 大规模迁移
     - Claude Code
     - 200K 上下文，全仓库理解
   * - 批量任务 / 异步处理
     - OpenAI Codex
     - 云端并行，不阻塞本地工作
   * - 前端开发 / 快速原型
     - Cursor
     - Composer 模式，即时预览
   * - 团队协作 / 代码审查
     - Codex + Claude Code
     - 异步委托 + 深度审查
   * - IDE 重度用户
     - Cursor
     - 零迁移成本

趋势：技术路线融合
====================

三大工具正在快速趋同。Claude Code 推出了桌面应用和 VS Code 扩展，Codex 发布了
桌面客户端和 IDE 插件，Cursor 推出了 Agent 模式和 CLI。选择哪款更多取决于
你的工作流偏好，而非功能差异。

.. admonition:: 未来展望
   :class: note

   2026 年 AI 编程 Agent 的关键趋势：
   - **多 Agent 协作**：Claude Code 已支持 pipeline() 和 parallel() 编排
   - **安全治理**：工具调用需要更精细的权限控制和审计
   - **MCP 统一**：标准化工具接入协议，降低集成成本
   - **本地模型**：开源模型在编程场景中快速追赶闭源模型
