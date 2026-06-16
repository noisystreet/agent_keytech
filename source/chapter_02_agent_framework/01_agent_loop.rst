.. _chapter-02-agent-loop:

===============================
Agent 主循环
===============================

Agent 的核心是一个**感知-推理-行动**的循环。你可以把它想象成
一个"思考-行动"的节拍器：每次节拍响起，Agent 回顾当前状态、决定下一步、
执行动作、观察结果，然后进入下一拍。

这个循环听起来简单，但实际生产环境中的 Agent 循环远比一个 `for` 循环复杂。
它需要处理半途中止的工具调用、崩溃后恢复、长时间运行的超时控制、
以及多轮对话中的上下文衰减——这些都是在写第一版 Agent 时最容易忽略的问题。

三种循环架构
================

不同场景需要不同粒度的循环控制。以下是三种最常见的架构模式。

1. 简单 ReAct 循环
------------------------------

最简单的实现：思考→行动→观察→重复。

.. code-block:: python

   class SimpleReActLoop:
       """
       最基础的 Agent 循环。适合简单任务、快速原型。
       局限：没有错误恢复、没有状态持久化、没有超时控制。
       """
       def __init__(self, llm, tools, max_steps=10):
           self.llm = llm
           self.tools = tools  # {"tool_name": callable}
           self.max_steps = max_steps
           self.messages = []

       def run(self, task: str) -> str:
           self.messages.append({"role": "user", "content": task})

           for step in range(self.max_steps):
               response = self.llm.generate(self.messages, tools=self.tools)

               action = self._parse_action(response)

               if action["type"] == "answer":
                   return action["content"]

               # 调用工具
               tool_result = self.tools[action["name"]](**action["args"])
               self.messages.append({
                   "role": "tool",
                   "content": f"{action['name']} 返回: {tool_result}"
               })

           return "已达最大步数"

这个循环的问题很明显。工具调用失败怎么办？没有重试。工具调用超时怎么办？
没有超时控制。模型输出了不合法的 JSON 怎么办？没有格式校验。
它只适合"一切顺利"的场景——而生产环境中，"一切顺利"是例外而非常态。

2. 状态机驱动循环
------------------------------

状态机将 Agent 循环建模为有限状态之间的转换。每个状态有明确的进入条件、
执行逻辑和退出条件。

.. mermaid::

   flowchart TD
       IDLE[空闲] -->|收到任务| THINKING[思考中]
       THINKING -->|需要工具| WAITING[等待工具]
       THINKING -->|可直接回答| ANSWERING[回答中]
       WAITING -->|工具返回| THINKING
       WAITING -->|超时| ERROR[错误处理]
       WAITING -->|工具异常| ERROR
       ERROR -->|重试| WAITING
       ERROR -->|放弃| ANSWERING
       ANSWERING -->|完成| IDLE
       THINKING -->|超出步数| IDLE

.. code-block:: python

   class StateMachineLoop:
       """
       状态机驱动的 Agent 循环。核心优势：
       - 每个状态边界明确，易于调试
       - 错误处理独立成状态，不会污染正常流程
       - 便于扩展（增加状态不影响已有逻辑）
       """
       def __init__(self, llm, tools, max_steps=10, timeout=30):
           self.llm = llm
           self.tools = tools
           self.max_steps = max_steps
           self.timeout = timeout  # 工具调用超时
           self.state = "idle"
           self.step_count = 0
           self.messages = []
           self.retry_count = 0
           self.max_retries = 2

       def run(self, task: str) -> str:
           self.messages.append({"role": "user", "content": task})
           self.state = "thinking"
           self.step_count = 0

           while self.state != "done":
               if self.step_count > self.max_steps:
                   return "超出最大步数限制"

               handler = {
                   "thinking": self._handle_thinking,
                   "waiting": self._handle_waiting,
                   "error": self._handle_error,
                   "answering": self._handle_answering,
               }[self.state]

               result = handler()
               if result:  # 有最终结果
                   return result

           return "未知状态退出"

       def _handle_thinking(self):
           """思考状态：LLM 决定下一步"""
           self.step_count += 1
           response = self.llm.generate(
               self.messages,
               tools=list(self.tools.keys())
           )
           action = self._parse_action(response)

           if action is None:
               # 模型输出格式不对，尝试重新生成
               self.messages.append({
                   "role": "user",
                   "content": "请使用正确的 JSON 格式输出。"
               })
               return None

           if action["type"] == "answer":
               self.pending_answer = action["content"]
               self.state = "answering"
               return None

           self.pending_action = action
           self.state = "waiting"
           self.action_start_time = time.time()
           return None

       def _handle_waiting(self):
           """等待状态：执行工具调用"""
           try:
               result = self._execute_with_timeout(
                   self.pending_action,
                   timeout=self.timeout
               )
               self.messages.append({
                   "role": "tool",
                   "content": result
               })
               self.retry_count = 0
               self.state = "thinking"
           except TimeoutError:
               self.state = "error"
               self.error = f"工具 {self.pending_action['name']} 超时"
           except Exception as e:
               self.state = "error"
               self.error = str(e)
           return None

       def _handle_error(self):
           """错误处理状态：决定是否重试或放弃"""
           if self.retry_count < self.max_retries:
               self.retry_count += 1
               self.state = "waiting"
               self.messages.append({
                   "role": "system",
                   "content": f"上一步出错：{self.error}，正在重试第 {self.retry_count} 次"
               })
           else:
               self.messages.append({
                   "role": "system",
                   "content": f"工具 {self.pending_action['name']} 多次失败，请尝试其他方法"
               })
               self.state = "thinking"
           return None

       def _handle_answering(self):
           """回答状态：输出最终结果"""
           self.state = "done"
           return self.pending_answer

3. 事件驱动循环
------------------------------

对于需要并行处理多个任务的复杂场景（比如多 Agent 系统中的编排器），
事件驱动模型更合适。每个 Agent 实例在事件总线上发布和订阅消息。

.. code-block:: python

   class EventDrivenLoop:
       """
       事件驱动的 Agent 循环。适合：
       - 多 Agent 协作场景
       - 需要异步处理工具调用的场景
       - 需要响应外部事件（如新消息到达）的场景
       """
       def __init__(self, llm, tools, event_bus):
           self.llm = llm
           self.tools = tools
           self.bus = event_bus  # 事件总线
           self.callbacks = {}   # 事件类型 → 处理器

       def register_handler(self, event_type, handler):
           self.callbacks[event_type] = handler

       async def run(self, task: str):
           # 发布任务事件
           await self.bus.publish("task.received", {"task": task})

           while True:
               # 监听事件
               event = await self.bus.wait_for_event()
               handler = self.callbacks.get(event.type)
               if handler:
                   result = await handler(event)
                   if event.type == "task.completed":
                       return result

循环的生命周期管理
======================

无论哪种循环架构，都需要处理四个生命周期阶段。

1. 进入（Entry）
------------------------------

Agent 循环的启动不只是一句 `run(task)`。在进入循环前需要：

.. code-block:: python

   class AgentLifecycle:
       def run_with_lifecycle(self, task: str) -> str:
           # Phase 1: 前置检查
           if not self._validate_task(task):
               return "任务格式不合法"
           if self._rate_limited():
               return "请求过于频繁，请稍后重试"

           # Phase 2: 初始化上下文
           context = self._build_initial_context(task)

           # Phase 3: 启动追踪
           trace_id = self._start_trace(task)

           try:
               # Phase 4: 执行主循环
               result = self._main_loop(context)
               self._record_success(trace_id, result)
               return result
           except Exception as e:
               self._record_failure(trace_id, e)
               return f"执行出错：{str(e)}"

这段代码展示了一个关键思路：Agent 循环不应该只处理"一切顺利"的情况。
你需要明确定义什么情况下拒绝进入循环（Phase 1）、进入时需要哪些
基础设施（Phase 3）、以及出错时如何收尾（finally 块）。

2. 退出（Exit）
------------------------------

循环的退出条件是 Agent 设计中最容易被低估的细节。不是什么情况都等
Agent 自己说"我完成了"。

.. list-table::
   :header-rows: 1

   * - 退出条件
     - 触发场景
     - 处理方式
   * - 正常完成
     - Agent 生成最终答案
     - 返回答案，记录追踪
   * - 超出步数
     - 循环超出 max_steps
     - 返回已收集的部分结果 + 警告
   * - 超时
     - 工具调用或推理超过时限
     - 立即中止，返回已有内容
   * - 用户中断
     - 用户主动停止
     - 返回当前已生成的内容
   * - 安全触达
     - 检测到危险操作
     - 拒绝执行，记录安全事件
   * - 异常崩溃
     - 非预期异常
     - 返回错误信息，标记失败

.. code-block:: python

   class ExitConditionManager:
       """管理循环的退出条件"""
       def __init__(self, max_steps=10, timeout=60):
           self.max_steps = max_steps
           self.timeout = timeout
           self.start_time = None
           self.steps = 0
           self.aborted = False

       def should_exit(self) -> tuple:
           """返回 (是否退出, 退出原因)"""
           if self.aborted:
               return True, "用户中断"

           elapsed = time.time() - self.start_time
           if elapsed > self.timeout:
               return True, f"超时（{elapsed:.0f}s > {self.timeout}s）"

           if self.steps >= self.max_steps:
               return True, f"超出最大步数（{self.steps} > {self.max_steps}）"

           return False, None

3. 暂停与恢复
------------------------------

在异步场景中（比如 Agent 等待一个慢速 API 返回），你不应该让循环
空转等待。更好的做法是：

.. code-block:: python

   class PausableLoop:
       """
       支持暂停/恢复的 Agent 循环。
       场景：Agent 调用外部 API 需要 10 秒，这期间可以释放线程资源。
       """
       def __init__(self):
           self.state_store = {}  # 保存循环状态以便恢复
           self.is_paused = False

       def pause(self):
           """暂停当前循环，保存所有状态"""
           self.is_paused = True
           self.state_store = {
               "messages": self.messages,
               "step": self.step_count,
               "pending_action": getattr(self, "pending_action", None),
               "state": self.state,
           }

       def resume(self):
           """从保存的状态恢复执行"""
           self.messages = self.state_store["messages"]
           self.step_count = self.state_store["step"]
           self.state = "waiting"  # 从等待工具结果的状态恢复
           self.is_paused = False
           return self._handle_waiting()  # 继续之前的工具等待

4. 安全熔断
------------------------------

如果 Agent 连续出错（工具调用失败、模型输出格式错误等），
你应该触发熔断机制，而不是让它无限重试。

.. code-block:: python

   class CircuitBreaker:
       """
       安全熔断器：当错误率超过阈值时，直接拒绝执行。
       """
       def __init__(self, failure_threshold=5, cooldown=60):
           self.failure_count = 0
           self.threshold = failure_threshold
           self.cooldown = cooldown
           self.last_failure_time = 0
           self.state = "closed"  # closed / open / half-open

       def call(self, fn):
           if self.state == "open":
               if time.time() - self.last_failure_time > self.cooldown:
                   self.state = "half-open"
               else:
                   raise CircuitBreakerOpen("熔断器已打开，拒绝执行")

           try:
               result = fn()
               self.failure_count = 0
               if self.state == "half-open":
                   self.state = "closed"
               return result
           except Exception as e:
               self.failure_count += 1
               self.last_failure_time = time.time()
               if self.failure_count >= self.threshold:
                   self.state = "open"
               raise

三种架构的选型对比
====================

.. list-table::
   :header-rows: 1

   * - 维度
     - 简单 ReAct
     - 状态机
     - 事件驱动
   * - 实现复杂度
     - 低（~30 行）
     - 中（~150 行）
     - 高（~300 行）
   * - 错误处理
     - 基本无
     - 内置重试/降级
     - 事件驱动恢复
   * - 可调试性
     - 差（全部混在一起）
     - 好（状态边界清晰）
     - 中（异步追踪复杂）
   * - 适用规模
     - 单一 Agent
     - 单 Agent 生产级
     - 多 Agent 系统
   * - 并行能力
     - 无
     - 无（串行状态）
     - 原生支持
   * - 典型框架
     - 早期 AutoGPT
     - LangChain Agent
     - CrewAI / 自建

循环中的上下文管理
====================

Agent 循环的核心约束是**上下文窗口有限**。你不能把每步的消息
都无限制地堆进对话历史。需要在循环中主动管理上下文。

.. code-block:: python

   class ContextManager:
       """
       管理 Agent 循环中的上下文预算。
       每步结束后检查 token 消耗，必要时触发压缩。
       """
       def __init__(self, llm, max_tokens=32000, reserve_ratio=0.3):
           self.llm = llm
           self.max_tokens = max_tokens
           self.reserve = int(max_tokens * reserve_ratio)

       def add_message(self, message: dict):
           self.messages.append(message)
           if self._total_tokens() > self.max_tokens - self.reserve:
               self._compress()

       def _compress(self):
           # 找到最早的"思考-行动-观察"三元组
           for i in range(1, len(self.messages) - 2):
               if self._is_react_triplet(i):
                   # 用摘要替换这三个消息
                   summary = self.llm.generate(
                       f"将以下 Agent 推理步骤压缩为一句话："
                       f"{self.messages[i:i+3]}"
                   )
                   self.messages[i] = {
                       "role": "system",
                       "content": f"[已压缩] {summary}"
                   }
                   del self.messages[i+1:i+3]
                   break

       def _is_react_triplet(self, i) -> bool:
           """判断从 i 开始是否是 thinking-action-observation 三元组"""
           # 基于消息类型的判断逻辑
           return False

循环追踪与调试
================

最后，一个实用的建议：**在生产环境中，永远不要让 Agent 循环静默运行**。
至少记录每步的输入输出、耗时和 token 消耗。

.. code-block:: python

   # Agent 循环的逐步骤追踪格式
   [
     {
       "step": 1,
       "state": "thinking",
       "tokens": 340,
       "latency_ms": 1200,
       "action": {"name": "search", "args": {"query": "北京天气"}},
       "action_result_summary": "晴，25°C",
       "action_latency_ms": 800,
     },
     {
       "step": 2,
       "state": "answering",
       "tokens": 80,
       "latency_ms": 600,
       "result": "北京今天天气晴朗，气温 25°C。",
     },
   ]

有了这种追踪数据，你就能回答"我的 Agent 时间花在哪里"这个问题。
大多数时候，答案不是"模型推理太慢"，而是"某个工具调用太慢"。
有了数据，优化方向才清晰。
