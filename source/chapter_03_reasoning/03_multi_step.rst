.. _chapter-03-multi-step:

===============================
多步推理
===============================

多步推理是 Agent 在面对需要多次工具调用才能解决的复杂任务时的核心能力。
与单步 ReAct 不同——每一步的行动结果可能改变下一步的推理路径，
Agent 需要动态调整计划。

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

.. code-block:: python

   # 多步推理示例：查询某人的最新论文
   task = "查一下李飞飞教授 2024 年发表的论文"

   # Step 1: 搜索学者基本信息
   thought1 = "我需要先找到李飞飞教授的 Google Scholar 页面"
   action1 = search("李飞飞 Stanford professor Google Scholar")

   # Step 2: 根据搜索结果定位最新论文
   thought2 = "找到了她的页面，现在筛选 2024 年的论文"
   action2 = get_page("https://scholar.google.com/...")

   # Step 3: 整理结果
   thought3 = "找到了 5 篇 2024 年的论文，按引用排序"
   answer = "李飞飞教授 2024 年发表的论文包括：..."

任务分解策略
================

1. 线性分解
------------------------------

将任务拆分为固定顺序的子步骤。适合流水线式任务。

.. code-block:: python

   class LinearDecomposition:
       def solve(self, task: str) -> str:
           # 先将任务拆解为子步骤列表
           steps = self._decompose(task)

           # 按顺序执行每个步骤
           intermediate_results = []
           for step in steps:
               result = self._execute_step(step)
               intermediate_results.append(result)

           # 综合所有结果
           return self._synthesize(intermediate_results)

       def _decompose(self, task: str) -> list:
           prompt = f"将以下任务拆解为 3-5 个可执行的子步骤：{task}"
           return llm.generate(prompt).split("\n")

2. 动态规划
------------------------------

每一步的下一步决策取决于上一步的结果，路径不是预定的。

.. code-block:: python

   class DynamicPlanner:
       def solve(self, task: str, max_steps=10) -> str:
           context = {"task": task, "completed": [], "pending": []}
           history = []

           for step in range(max_steps):
               # 根据当前进度决定下一步
               decision = self._decide_next(context, history)
               if decision["type"] == "answer":
                   return decision["content"]

               # 执行决策
               result = self._execute(decision)
               history.append({"decision": decision, "result": result})

               # 更新上下文
               context["completed"].append(decision)
               context["pending"].extend(result.get("new_tasks", []))

       def _decide_next(self, context, history) -> dict:
           prompt = f"""
           任务：{context['task']}
           已完成：{context['completed']}
           待完成：{context['pending']}

           下一步应该做什么？请以 JSON 格式输出：
           - {{"type": "action", "tool": "...", "args": {{...}}}}
           - {{"type": "answer", "content": "..."}}
           """
           return llm.generate(prompt, temperature=0.0)

3. 回溯探索
------------------------------

当某条路径失败时，回到之前的分支点尝试其他方案。

.. code-block:: python

   class BacktrackingPlanner:
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

               # 生成多个候选方案
               candidates = self._generate_candidates(node["state"], node["path"])

               for cand in reversed(candidates):  # 反向入栈，优先尝试第一个
                   stack.append({
                       "path": node["path"] + [cand],
                       "state": cand["next_state"],
                       "attempts": 0,
                   })
               # 当前方案失败，增加尝试计数
               node["attempts"] += 1
               if node["attempts"] < self.max_backtracks:
                   stack.append(node)

           return "无法完成任务"

       def _generate_candidates(self, state, path) -> list:
           prompt = f"当前状态：{state}\n已完成步骤：{path}\n请给出接下来的 2-3 种可行方案。"
           return self.llm.generate(prompt, temperature=0.7)

.. admonition:: 多步推理 vs 单步 ReAct
   :class: tip

   两者的核心区别在于**规划粒度**：
   - **ReAct** 是"边想边做"——思考和行动交替，每一步由 LLM 实时决定
   - **多步推理** 是"先想再做"——先规划再执行，子任务之间的关系更明确

   实践中建议组合使用：先多步分解任务框架，再在每一步中用 ReAct 具体执行。

错误恢复策略
================

.. code-block:: python

   class RobustMultiStep:
       def solve(self, task: str) -> str:
           max_retries = 2
           for attempt in range(max_retries + 1):
               try:
                   plan = self._plan(task)
                   return self._execute_plan(plan)
               except StepFailedError as e:
                   if attempt >= max_retries:
                       raise
                   # 重规划：基于失败信息调整方案
                   task = f"{task}\n\n注意：之前的方案在以下步骤失败：{e}，请调整方案。"
           return "任务失败"

参考文献
============

- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", 2023
- Khot et al., "Decomposed Prompting: A Modular Approach for Solving Complex Tasks", 2022
