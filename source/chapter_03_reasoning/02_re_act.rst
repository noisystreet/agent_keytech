.. _chapter-03-react:

===============================
ReAct 模式
===============================

ReAct（Reasoning + Acting）是 Agent 推理的核心模式，由 Yao et al. (2022) 提出。
它将推理步骤和行动步骤交替进行，让模型能够边思考边行动。

.. mermaid::

   sequenceDiagram
       participant LLM
       participant Agent
       participant Tool

       Agent->>LLM: 初始问题
       LLM->>Agent: Thought: 我需要搜索...
       Agent->>Tool: search(...)
       Tool->>Agent: 结果...
       Agent->>LLM: 观察到搜索结果
       LLM->>Agent: Thought: 根据结果，我得出结论...
       Agent->>User: 最终答案

.. code-block:: python

   REACT_PROMPT = """
   你是 AI 助手。交替执行推理和行动：

   Thought: 分析当前状态
   Action: 工具名(参数)
   Observation: 工具的返回结果
   （重复以上步骤）
   Thought: 我可以给出最终答案了
   Answer: 最终答案
   """
