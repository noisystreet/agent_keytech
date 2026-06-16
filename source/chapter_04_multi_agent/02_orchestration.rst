.. _chapter-04-orchestration:

===============================
编排模式
===============================

编排（Orchestration）是多 Agent 系统的"导演"，决定了 Agent 之间的
工作流和交互顺序。良好的编排模式能让多个 Agent 高效协作，而不良的编排
会导致资源浪费和任务失败。

三种经典编排模式
====================

1. 顺序链（Sequential Chain）
------------------------------

Agent 依次处理，每个 Agent 的输出是下一个的输入。

.. mermaid::

   flowchart LR
       A1[Agent A<br>分析师] --> A2[Agent B<br>搜索者]
       A2 --> A3[Agent C<br>写作者]
       A3 --> Result[最终输出]

.. code-block:: python

   class SequentialOrchestrator:
       def __init__(self, agents: list):
           self.agents = agents

       def run(self, task: str) -> str:
           result = task
           for agent in self.agents:
               result = agent.run(result)
               self._validate_output(result)  # 检查是否合法
           return result

       def _validate_output(self, output):
           if "error" in output.lower():
               raise StepFailedError(f"Agent 返回错误：{output}")

2. 路由模式（Router）
------------------------------

编排器根据任务类型分发给不同的专用 Agent。

.. code-block:: python

   class RouterOrchestrator:
       def __init__(self, agents: dict):
           # agents = {"coding": code_agent, "writing": write_agent, ...}
           self.agents = agents
           self.default_agent = agents.get("general")

       def route(self, task: str) -> str:
           # 1. 分析任务类型
           category = self._classify(task)

           # 2. 选择对应 Agent
           agent = self.agents.get(category, self.default_agent)

           # 3. 执行
           return agent.run(task)

       def _classify(self, task: str) -> str:
           prompt = f"将以下任务分类（coding/writing/analysis/other）：{task}"
           return llm.generate(prompt, temperature=0.0).strip().lower()

3. 分层模式（Hierarchical）
------------------------------

高级 Agent（管理者）分解任务，低级 Agent（工作者）执行子任务。

.. code-block:: python

   class HierarchicalOrchestrator:
       def __init__(self, manager, workers: dict):
           self.manager = manager
           self.workers = workers  # {"worker_type": worker_agent}

       def run(self, task: str) -> str:
           # 管理者分解任务
           plan = self.manager.run(f"将以下任务分解为子任务：{task}")
           subtasks = self._parse_plan(plan)

           # 分配并监控执行
           results = {}
           for subtask in subtasks:
               worker_type = subtask["type"]
               worker = self.workers[worker_type]
               results[subtask["id"]] = worker.run(subtask["description"])

           # 管理者综合结果
           summary = self.manager.run(
               f"综合以下结果，生成最终答案：{results}"
           )
           return summary

动态编排模式
================

1. 监督者模式（Supervisor）
------------------------------

监督者不直接执行任务，而是监控其他 Agent 的执行质量，在发现问题时介入。

.. code-block:: python

   class SupervisorPattern:
       def __init__(self, workers, supervisor, quality_threshold=0.8):
           self.workers = workers
           self.supervisor = supervisor
           self.threshold = quality_threshold

       def run(self, task: str) -> str:
           result = None
           for worker in self.workers:
               candidate = worker.run(task)
               score = self._evaluate_quality(task, candidate)
               if score > self.threshold:
                   result = candidate
                   break
           return result

       def _evaluate_quality(self, task, result) -> float:
           eval_prompt = f"任务：{task}\n结果：{result}\n质量评分（0-1）："
           score = float(self.supervisor.run(eval_prompt))
           return score

2. 竞争模式（Competition）
------------------------------

多个 Agent 独立解决相同问题，选择最优结果。

.. code-block:: python

   class CompetitionPattern:
       def __init__(self, agents, judge_agent=None):
           self.agents = agents
           self.judge = judge_agent

       def run(self, task: str) -> str:
           # 并行执行所有 Agent
           from concurrent.futures import ThreadPoolExecutor
           with ThreadPoolExecutor() as executor:
               results = list(executor.map(
                   lambda a: a.run(task), self.agents
               ))

           if self.judge:
               # 由评审 Agent 选择最佳结果
               prompt = f"从以下候选中选择最佳答案：\n" + \
                        "\n".join(f"候选{i}: {r}" for i, r in enumerate(results))
               return self.judge.run(prompt)

           return max(results, key=self._score)

.. admonition:: 编排模式选择指南
   :class: tip

   - **简单流水线** → 顺序链（最简单）
   - **任务类型多样** → 路由模式（灵活性好）
   - **复杂任务** → 分层模式（分而治之）
   - **质量敏感** → 监督者模式（把关质量）
   - **追求最优** → 竞争模式（消耗最大）

错误恢复闭环
================

.. code-block:: python

   class ResilientOrchestrator:
       def __init__(self, orchestrator, max_retries=2):
           self.orchestrator = orchestrator
           self.max_retries = max_retries

       def run(self, task: str) -> str:
           for attempt in range(self.max_retries + 1):
               try:
                   return self.orchestrator.run(task)
               except AgentError as e:
                   if attempt >= self.max_retries:
                       raise
                   # 重新编排：更换 Agent 或调整策略
                   task = self._reroute(task, e)

       def _reroute(self, task, error):
           return f"{task}\n[注意：之前的尝试失败：{error}，请使用不同策略]"
