.. _chapter-06-guardrails:

===============================
护栏机制
===============================

护栏（Guardrails）是 Agent 安全的第一道防线，在 LLM 的输入和输出端
进行拦截验证。Agent 拥有工具执行能力后，安全不再只是"言论问题"——
一个错误的工具调用可能造成真实的经济损失或数据泄露。

我觉得可以把护栏理解为**门禁系统**。门禁不是用来阻止所有进入的——
它只阻止不应该进入的人。好的护栏也是这样：它只拦截真正有害的内容，
不会把正常的请求也挡在外面。护栏设计得过于宽松，恶意请求会穿透；
设计得过于严格，Agent 就没法正常工作了。这个度在哪里，正是护栏
设计中最难把握的地方。

为什么要三层？
================

单层护栏就像只装了一把锁的门——如果这把锁被打开了，就再也没有保护了。
三层架构的原因是：**每一层防御的是不同的威胁**。

- **输入护栏**：防止恶意输入进入 Agent。这条防线防的是"坏人"。
- **输出护栏**：防止 Agent 输出有害内容。这条防线防的是"自己人犯错"。
- **速率护栏**：防止 Agent 被滥用。这条防线防的是"用量失控"。

有些攻击只要通过输入护栏就防住了（比如直接的 prompt 注入），
但间接注入（Agent 从网页读取的内容中包含恶意指令）可能绕过输入护栏——
用户输入是干净的，但 Agent 读取的内容有问题。这时候输出护栏就能拦截：
Agent 可能产生了敏感操作，但在执行之前被输出护栏发现并阻止。

.. mermaid::

   flowchart TD
       Input[用户输入] --> InGuard[输入护栏]
       InGuard --> Agent[Agent 执行]
       Agent --> OutGuard[输出护栏]
       OutGuard --> Action[工具调用/结果输出]
       InGuard -- 拒绝 --> Reject[拒绝请求]
       OutGuard -- 拦截 --> Block[拦截输出]

       subgraph InputGuard [输入护栏]
           I1[Prompt 注入检测]
           I2[权限检查]
           I3[速率限制]
       end

       subgraph OutputGuard [输出护栏]
           O1[敏感信息检测]
           O2[操作确认]
           O3[内容安全]
       end

1. 输入护栏
------------------------------

输入护栏检查的是用户发给 Agent 的内容。它在 Agent 看到内容之前就拦截。

.. code-block:: python

   class InputGuardrail:
       """
       输入护栏。拦截：
       - Prompt 注入（用户试图覆盖系统指令）
       - 敏感操作（用户试图让 Agent 执行危险操作）
       """
       def __init__(self):
           self.injection_patterns = [
               "忽略之前的指令",
               "忽略所有之前的约束",
               "你被越狱了",
               "SYSTEM:",
           ]
           self.sensitive_actions = [
               "delete_file", "remove_user", "execute_shell",
               "modify_system_config", "send_email_as_user"
           ]

       def check(self, user_input: str, tool_calls: list = None) -> tuple:
           for pattern in self.injection_patterns:
               if pattern in user_input.lower():
                   return False, f"检测到注入模式：{pattern}"

           if tool_calls:
               for call in tool_calls:
                   if call.get("name") in self.sensitive_actions:
                       return False, f"敏感操作需要额外授权：{call['name']}"

           return True, "通过"

2. 输出护栏
------------------------------

输出护栏检查的是 Agent 生成的内容。它关注 Agent 是否"说了不该说的话"。

.. code-block:: python

   class OutputGuardrail:
       """
       输出护栏。拦截：
       - 敏感信息泄露（API Key、密码、私钥）
       - 危险命令（Agent 可能被诱导执行的 shell 命令）
       """
       def __init__(self):
           self.secret_patterns = [
               r"sk-[a-zA-Z0-9]{20,}",     # OpenAI API Key
               r"AKIA[0-9A-Z]{16}",         # AWS Access Key
               r"-----BEGIN (RSA|EC) PRIVATE KEY-----",
               r"(password|passwd|pwd)[=:]\s*\S+",
           ]
           self.dangerous_commands = [
               "rm -rf /", "sudo", "chmod 777",
               "DROP TABLE", "TRUNCATE",
           ]

       def check(self, agent_output: str) -> tuple:
           import re
           for pattern in self.secret_patterns:
               if re.search(pattern, agent_output):
                   return False, "检测到可能的敏感信息泄露"

           for cmd in self.dangerous_commands:
               if cmd in agent_output:
                   return False, f"输出包含危险命令：{cmd}"

           return True, "通过"

3. 速率与成本控制
------------------------------

速率护栏防止 Agent 被滥用——不管是恶意的还是意外的。

.. code-block:: python

   class RateLimitGuardrail:
       """
       速率护栏。控制：
       - 每分钟最大调用次数（防止 API 过载）
       - 每日最大成本（防止预算超支）
       """
       def __init__(self, max_calls_per_minute=30, max_daily_cost=10.0):
           self.max_cpm = max_calls_per_minute
           self.max_daily_cost = max_daily_cost
           self.call_log = []

       def check(self, estimated_cost: float = 0) -> tuple:
           now = time.time()
           self.call_log = [(t, c) for t, c in self.call_log
                          if now - t < 60]

           if len(self.call_log) >= self.max_cpm:
               return False, f"速率限制：每分钟最多 {self.max_cpm} 次调用"

           daily_cost = sum(c for t, c in self.call_log
                          if time.localtime(t).tm_yday == time.localtime(now).tm_yday)
           if daily_cost + estimated_cost > self.max_daily_cost:
               return False, f"日预算限制：当日已用 ${daily_cost:.2f}"

           return True, "通过"

Guardrails 框架集成
====================

把三个护栏组合成一个系统，自动对 Agent 的输入和输出进行安全检查。

.. code-block:: python

   class GuardrailSystem:
       """
       完整的护栏系统。自动包装 Agent 执行：
       Agent.run(task) → check_input → Agent → check_output → result
       """
       def __init__(self):
           self.input_guard = InputGuardrail()
           self.output_guard = OutputGuardrail()
           self.rate_guard = RateLimitGuardrail()

       def wrap_agent(self, agent):
           class GuardedAgent:
               def run(self, user_input):
                   if err := self.guard.check_input(user_input):
                       return err
                   if err := self.guard.check_rate():
                       return err
                   output = agent.run(user_input)
                   if err := self.guard.check_output(output):
                       return err
                   return output
           return GuardedAgent()

护栏的误报与漏报权衡
======================

护栏设计中最难的问题是**误报** （把正常请求拦了）和**漏报** （没拦住恶意请求）
之间的权衡。

.. list-table::
   :header-rows: 1

   * - 策略
     - 误报率
     - 漏报率
     - 适合场景
   * - 宽松（仅关键词检测）
     - 低
     - 高
     - 内部工具、低风险场景
   * - 均衡（关键词 + 规则）
     - 中
     - 中
     - 大多数场景
   * - 严格（关键词 + 规则 + LLM 检测）
     - 高
     - 低
     - 面向外部用户、高风险场景

没有完美的护栏，只有适合你场景的护栏。我的经验是：**先用宽松策略上线，
然后根据实际拦截日志逐步收紧**。不要一开始就上最严格的配置——
你会在"误报"上浪费大量时间。宁可先漏几个恶意请求（反正有日志可以追溯），
也不要把正常用户全挡在门外。

.. admonition:: 护栏 vs 对齐
   :class: tip

   - **护栏** 是运行时防护——在输入和输出端拦截不安全的内容，可独立于模型
   - **对齐** 是训练时的防护——通过训练让模型本身偏向安全和有益
   - 两者互补：护栏提供确定性的保障（一定能拦截一个关键词），
     对齐提供底层倾向性（模型自己就不想说有害的话）
   - 生产环境必须同时使用两者

参考文献
============

- Lakera AI, "Guardrails for LLM-based Applications", 2024
- Nvidia, "NeMo Guardrails: A Framework for LLM Safety", 2023
