.. _chapter-06-guardrails:

===============================
护栏机制
===============================

护栏（Guardrails）是 Agent 安全的第一道防线，在 LLM 的输入和输出端
进行拦截验证。Agent 拥有工具执行能力后，安全不再只是"言论问题"——
一个错误的工具调用可能造成真实的经济损失或数据泄露。

三层护栏架构
================

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

.. code-block:: python

   class InputGuardrail:
       def __init__(self):
           self.injection_patterns = [
               "忽略之前的指令",
               "忽略所有之前的约束",
               "你被越狱了",
               "SYSTEM:",
               # 更多注入模式...
           ]
           self.sensitive_actions = [
               "delete_file", "remove_user", "execute_shell",
               "modify_system_config", "send_email_as_user"
           ]

       def check(self, user_input: str, tool_calls: list = None) -> tuple:
           """返回 (是否通过, 原因)"""
           # 检查注入
           for pattern in self.injection_patterns:
               if pattern in user_input.lower():
                   return False, f"检测到注入模式：{pattern}"

           # 检查工具调用合法性
           if tool_calls:
               for call in tool_calls:
                   if call.get("name") in self.sensitive_actions:
                       return False, f"敏感操作需要额外授权：{call['name']}"

           return True, "通过"

2. 输出护栏
------------------------------

.. code-block:: python

   class OutputGuardrail:
       def __init__(self):
           self.secret_patterns = [
               r"sk-[a-zA-Z0-9]{20,}",     # OpenAI API Key
               r"AKIA[0-9A-Z]{16}",         # AWS Access Key
               r"-----BEGIN (RSA|EC) PRIVATE KEY-----",  # 私钥
               r"(password|passwd|pwd)[=:]\s*\S+",       # 密码
           ]
           self.dangerous_commands = [
               "rm -rf /", "sudo", "chmod 777",
               "DROP TABLE", "TRUNCATE",
           ]

       def check(self, agent_output: str) -> tuple:
           """检查输出是否安全"""
           import re
           # 检查敏感信息泄露
           for pattern in self.secret_patterns:
               if re.search(pattern, agent_output):
                   return False, "检测到可能的敏感信息泄露"

           # 检查危险命令
           for cmd in self.dangerous_commands:
               if cmd in agent_output:
                   return False, f"输出包含危险命令：{cmd}"

           return True, "通过"

3. 速率与成本控制
------------------------------

.. code-block:: python

   class RateLimitGuardrail:
       def __init__(self, max_calls_per_minute=30, max_daily_cost=10.0):
           self.max_cpm = max_calls_per_minute
           self.max_daily_cost = max_daily_cost
           self.call_log = []  # [(timestamp, cost)]

       def check(self, estimated_cost: float = 0) -> tuple:
           now = time.time()

           # 清理一分钟前的记录
           self.call_log = [(t, c) for t, c in self.call_log
                          if now - t < 60]

           # 检查速率
           if len(self.call_log) >= self.max_cpm:
               return False, f"速率限制：每分钟最多 {self.max_cpm} 次调用"

           # 检查当日总花费
           daily_cost = sum(c for t, c in self.call_log
                          if time.localtime(t).tm_yday == time.localtime(now).tm_yday)
           if daily_cost + estimated_cost > self.max_daily_cost:
               return False, f"日预算限制：当日已用 ${daily_cost:.2f}"

           return True, "通过"

Guardrails 框架集成
====================

.. code-block:: python

   class GuardrailSystem:
       def __init__(self):
           self.input_guard = InputGuardrail()
           self.output_guard = OutputGuardrail()
           self.rate_guard = RateLimitGuardrail()

       def check_input(self, user_input: str) -> str:
           """输入检查，返回错误信息或 None"""
           passed, reason = self.input_guard.check(user_input)
           if not passed:
               return f"安全策略拒绝：{reason}"
           return None

       def check_output(self, agent_output: str) -> str:
           """输出检查，返回错误信息或 None"""
           passed, reason = self.output_guard.check(agent_output)
           if not passed:
               return f"输出拦截：{reason}"
           return None

       def check_rate(self) -> str:
           """速率检查"""
           passed, reason = self.rate_guard.check()
           if not passed:
               return f"限制：{reason}"
           return None

       def wrap_agent(self, agent):
           """包装 Agent，自动执行安全检查"""
           class GuardedAgent:
               def run(self, user_input):
                   # 检查输入
                   if err := self.guard.check_input(user_input):
                       return err
                   # 检查速率
                   if err := self.guard.check_rate():
                       return err
                   # 执行 Agent
                   output = agent.run(user_input)
                   # 检查输出
                   if err := self.guard.check_output(output):
                       return err
                   return output
           return GuardedAgent()

.. admonition:: 护栏 vs 对齐
   :class: tip

   - **护栏** 是运行时防护——在输入和输出端拦截不安全的内容，可独立于模型
   - **对齐** 是训练时的防护——通过训练让模型本身偏向安全和有益
   - 两者互补：护栏提供确定性保障，对齐提供底层倾向性
   - 生产环境**必须同时使用两者**

参考文献
============

- Lakera AI, "Guardrails for LLM-based Applications", 2024
- Nvidia, "NeMo Guardrails: A Framework for LLM Safety", 2023
