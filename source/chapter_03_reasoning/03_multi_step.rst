.. _chapter-03-multi-step:

===============================
多步推理
===============================

多步推理是 Agent 在面对需要多次工具调用才能解决的复杂任务时的核心能力。
与单步 ReAct 不同——每一步的行动结果可能改变下一步的推理路径，
Agent 需要动态调整计划。

多步推理和 ReAct 的核心区别在于**规划粒度**。ReAct 是"边走边看"——
每一步由 LLM 实时决定做什么。多步推理是"先想再看"——先规划一个
执行框架，再逐歩执行，中间可以根据结果调整。

什么时候需要多步推理？
=========================

.. list-table::
   :header-rows: 1

   * - 任务复杂度
     - 建议方案
     - 原因
   * - 1 步（如"现在几点？"）
     - 单次 LLM 调用
     - 不需要工具，直接回答
   * - 2-3 步（如"查北京的天气"）
     - ReAct 循环
     - 一步搜索，一步回答
   * - 4-8 步（如"对比 A 和 B 的产品"）
     - 多步推理
     - 需要先规划后执行
   * - 8 步以上（如"帮我写一份市场分析报告"）
     - 层级规划 + 多步推理
     - 需要分解为子任务

从单步到多步
================

.. mermaid::

   flowchart TD
       Task[复杂任务] --> Decompose[任务分解]
       Decompose --> Step1[子任务 1]
       Step1 --> Result1[结果 1]
       Result1 --> Decide{下一步？}
       Decide --> Step2[子任务 2<br>基于结果 1 调整]
       Step2 --> Result2[结果 2]
       Result2 --> Step3[子任务 3]
       Step3 --> Result3[结果 3]
       Result3 --> Synthesize[综合所有结果]
       Synthesize --> Answer[最终答案]

任务分解策略
================

1. 线性分解
------------------------------

将任务拆分为固定顺序的子步骤。适合流水线式任务。

.. code-block:: python

   class LinearDecomposition:
       """线性分解：固定的子任务序列，每步依赖上一步的结果"""
       def solve(self, task: str) -> str:
           steps = self._decompose(task)
           intermediate_results = []

           for step in steps:
               result = self._execute_step(step)
               intermediate_results.append(result)

           return self._synthesize(intermediate_results)

       def _decompose(self, task: str) -> list:
           prompt = f"将以下任务拆解为 3-5 个可执行的子步骤：{task}"
           return llm.generate(prompt).split("\n")

2. 动态规划
------------------------------

每一步的下一步决策取决于上一步的结果。路径不是预定的。

.. code-block:: python

   class DynamicPlanner:
       """
       动态规划：每一步根据当前进度决定下一步。
       适合搜索类任务（你不知道搜索到结果后会发生什么）。
       """
       def solve(self, task: str, max_steps=10) -> str:
           context = {"task": task, "completed": [], "pending": []}
           history = []

           for step in range(max_steps):
               decision = self._decide_next(context, history)
               if decision["type"] == "answer":
                   return decision["content"]

               result = self._execute(decision)
               history.append({"decision": decision, "result": result})

               context["completed"].append(decision)
               context["pending"].extend(result.get("new_tasks", []))

               # 如果发现当前路径走不通，尝试回溯
               if result.get("status") == "failed":
                   alternative = self._find_alternative(history)
                   if alternative:
                       return self.solve(alternative)

       def _decide_next(self, context, history) -> dict:
           prompt = f"""
           任务：{context['task']}
           已完成：{context['completed']}
           待完成：{context['pending']}

           下一步应该做什么？输出 JSON：
           - {{"type": "action", "tool": "...", "args": {{...}}}}
           - {{"type": "answer", "content": "..."}}
           """
           return llm.generate(prompt, temperature=0.0)

3. 回溯探索
------------------------------

当某条路径失败时，回到之前的分支点尝试其他方案。

.. code-block:: python

   class BacktrackingPlanner:
       """
       回溯探索：当一条路径走不通时，回到分支点尝试其他方案。
       这需要 Agent 在执行过程中记录"决策点"。
       """
       def __init__(self, llm):
           self.llm = llm
           self.max_backtracks = 3

       def solve(self, task: str) -> str:
           stack = [{"path": [], "state": task, "attempts": 0}]

           while stack:
               node = stack.pop()
               if self._is_complete(node["state"]):
                   return node["state"]

               if node["attempts"] >= self.max_backtracks:
                   continue  # 该分支已耗尽尝试次数

               candidates = self._generate_candidates(node["state"])

               # 反向入栈，优先尝试第一个候选
               for cand in reversed(candidates):
                   stack.append({
                       "path": node["path"] + [cand],
                       "state": cand["next_state"],
                       "attempts": 0,
                   })

               # 当前方案也放回栈中，增加尝试次数
               node["attempts"] += 1
               if node["attempts"] < self.max_backtracks:
                   stack.append(node)

           return "无法完成任务"

多步推理的最佳实践
====================

.. admonition:: 每步的上下文管理
   :class: tip

   多步推理最容易出的问题是上下文膨胀。每步的输入输出都在增长，

   .. code-block:: python

       # 每步只传必要信息
       def build_step_context(step_n, current_result, original_task, previous_summary):
           return f"""
           原始任务：{original_task}
           已完成步骤摘要：{previous_summary}
           当前步骤结果：{current_result}

           下一步计划：
           """

这种方法比把完整历史传给 LLM 更节省 token，而且效果往往更好——
因为 LLM 不会被历史中的噪声干扰。

.. admonition:: 何时该回溯？
   :class: caution

   不是所有失败都需要回溯。判断标准：
   - **工具调用失败** （如网络超时）：重试 2-3 次，不需要回溯
   - **信息不足** （如搜索返回空结果）：重写查询，不需要回溯
   - **逻辑矛盾** （如"今天是 2026 年"但找到的是 2024 年的数据）：需要回溯
   - **目标偏离** （Agent 在执行中发现需要不同方向）：需要回溯并重新规划

多步推理 vs 单步 ReAct
========================

.. list-table::
   :header-rows: 1

   * - 维度
     - ReAct
     - 多步推理
   * - 规划时机
     - 实时决策，走一步看一步
     - 先规划框架再执行
   * - 路径可变性
     - 每一步都受上一步结果影响
     - 有预设框架，但可调整
   * - 最适合
     - 2-3 步的简单交互
     - 4-8 步的复杂任务
   * - 上下文开销
     - 低（只保留最近几步）
     - 需要管理中间结果

实践中建议组合使用：先多步分解任务框架，再在每一步中用 ReAct 具体执行。

参考文献
============

- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", 2023
- Khot et al., "Decomposed Prompting: A Modular Approach for Solving Complex Tasks", 2022
