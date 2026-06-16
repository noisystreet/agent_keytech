.. _chapter-02-planning:

===============================
规划能力
===============================

规划（Planning）是 Agent 面对复杂任务时，将目标分解为可执行子任务的能力。
简单任务可以通过 ReAct 一步完成，但复杂任务需要显式的规划。

任务分解策略
================

.. code-block:: python

   # Plan-then-Execute：先规划再执行
   class PlanThenExecute:
       def plan(self, task: str) -> List[str]:
           prompt = f"将以下任务分解为最多 5 个步骤：\n{task}"
           response = llm.generate(prompt)
           steps = parse_steps(response)
           return steps

       def execute(self, steps: List[str]) -> str:
           results = []
           for step in steps:
               result = agent.run(step)
               results.append(result)
           return "\n".join(results)

   # 动态规划：执行中可调整
   class DynamicPlanning:
       def run(self, task: str):
           plan = [task]  # 从单一目标开始
           while plan:
               current = plan.pop(0)
               result = agent.run(current)
               feedback = evaluate(result)
               if needs_refinement(feedback):
                   # 根据反馈调整后续计划
                   plan = refine_plan(plan, feedback)
            return result
