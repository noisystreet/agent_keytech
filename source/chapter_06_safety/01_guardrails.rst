.. _chapter-06-guardrails:

===============================
护栏机制
===============================

护栏（Guardrails）是 Agent 安全的第一道防线，在 LLM 的输入和输出端进行拦截验证。

.. code-block:: python

   class Guardrail:
       """Agent 安全的护栏系统"""
       def check_input(self, user_input: str) -> bool:
           # 阻止注入攻击
           if contains_injection(user_input):
               return False
           # 阻止超出范围的工具调用
           if attempts_restricted_actions(user_input):
               return False
           return True

       def check_output(self, agent_output: str) -> bool:
           # 防止 Agent 泄露敏感信息
           if contains_secrets(agent_output):
               return False
           # 防止 Agent 执行危险操作
           if is_dangerous_action(agent_output):
               return False
           return True

       def run(self, user_input: str, agent) -> str:
           if not self.check_input(user_input):
               return "输入被安全策略拒绝"
           output = agent.run(user_input)
           if not self.check_output(output):
               return "输出被安全策略拦截，请联系管理员"
           return output
