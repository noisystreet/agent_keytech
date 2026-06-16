.. _chapter-01-prompt-engineering:

===============================
Prompt Engineering
===============================

提示词工程（Prompt Engineering）是与 LLM 交互的基础技能。在 Agent 场景中，
提示词不仅是"指令"，更是驱动 Agent 行为的核心逻辑载体。

Agent 提示词的特殊性
========================

与普通对话不同，Agent 的提示词需要同时完成三件事：

1. **定义角色和身份**：告诉模型它是谁、能做什么
2. **描述工具接口**：告诉模型有哪些工具可用及如何调用
3. **指定输出格式**：确保模型输出可以被解析器正确解析

.. code-block:: python

   AGENT_SYSTEM_PROMPT = """
   你是 AI 助手，拥有以下工具。每次回复必须严格按照 JSON 格式：

   {
       "thought": "你的推理过程",
       "tool": "要调用的工具名称（或 null）",
       "params": {"参数名": "参数值"},
       "answer": "最终答案（或 null）"
   }

   可用工具：
   - search(query: str): 搜索互联网获取信息
   - calculator(expr: str): 执行数学计算
   """
