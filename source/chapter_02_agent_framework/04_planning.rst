.. _chapter-02-planning:

===============================
规划能力
===============================

规划（Planning）是 Agent 面对复杂任务时，将目标分解为可执行子任务的能力。
简单任务可以通过 ReAct 一步完成（比如"查天气"），但对复杂任务（比如
"帮我对比三款云服务的定价方案并写一份报告"），Agent 需要一个更清晰的
路线图——这就是规划。

规划解决的是 ReAct 模式的根本缺陷：**ReAct 走一步看一步，没有全局视野**。
当任务需要 5 步以上时，ReAct 很容易跑偏——Agent 被中间某步的工具返回
结果"带歪"，忘记了最初的目标。规划就是在执行之前先画一张地图，
然后在执行过程中不断对照地图确认方向。

为什么需要规划？
=================

不规划和规划的差异，在日常工作中很容易体会到：

.. list-table::
   :header-rows: 1

   * - 场景
     - 无规划（纯 ReAct）
     - 有规划
   * - 5 步以内
     - 效果不错，简单直接
     - 规划开销 > 收益
   * - 5-10 步
     - 开始跑偏，中间结果干扰决策
     - 方向明确，步骤清晰
   * - 10 步以上
     - 大概率忘记初始目标
     - 能保持一致性
   * - 多分支任务
     - 只探索一条路径
     - 可以权衡多条路径
   * - 错误恢复
     - 不知道"该回到哪一步"
     - 知道当前进度，能回退到上一步

三种规划策略
================

1. 先规划再执行（Plan-then-Execute）
--------------------------------------

最直观的策略：先让 LLM 制定完整计划，然后按计划执行。

.. code-block:: python

   class PlanThenExecute:
       """
       先规划再执行。适合：
       - 步骤明确的流水线任务
       - 任务环境相对稳定（工具返回结果可预测）
       - 不允许在中途大幅调整方向
       """
       def run(self, task: str) -> str:
           # Phase 1: 制定计划
           plan = self._create_plan(task)
           # 返回格式：[{"step": 1, "action": "search", "target": "..."}, ...]
           if not plan:
               return "无法为任务制定计划"

           # Phase 2: 按计划执行
           results = []
           for step in plan:
               result = self._execute_step(step)
               results.append(result)

           # Phase 3: 综合结果
           return self._synthesize(task, results)

       def _create_plan(self, task: str) -> list:
           prompt = f"""
           你需要为以下任务制定一个详细的执行计划：

           任务：{task}

           可用工具：
           - search(query): 搜索互联网
           - read_url(url): 读取网页内容
           - calculator(expr): 数学计算

           请按以下格式输出计划，最多 5 步：
           [{{"step": 1, "tool": "search", "args": {{"query": "..."}}, "expected": "..."}},
            ...]

           注意：
           - 每步应该只依赖上一步的结果
           - 不假设中间结果的内容
           - 给每步设定明确目标
           """
           return parse_plan(llm.generate(prompt, temperature=0.0))

2. 动态规划（Dynamic Planning）
--------------------------------------

每一步的下一步取决于上一步的结果。路径不是预定的。

.. code-block:: python

   class DynamicPlanner:
       """
       动态规划：走一步看一步。
       适合：
       - 每一步的结果不可预测（如开放域搜索）
       - 需要在执行中调整方向
       - 任务涉及探索性分析
       """
       def run(self, task: str, max_steps=10):
           context = {
               "task": task,
               "completed": [],
               "current_findings": [],
               "remaining_questions": [],
           }

           for step in range(max_steps):
               decision = self._decide_next(context)
               if decision["type"] == "answer":
                   return decision["content"]

               result = self._execute(decision)
               context["completed"].append({
                   "action": decision,
                   "result": result["summary"]
               })
               context["current_findings"].extend(result.get("findings", []))
               context["remaining_questions"] = result.get("new_questions", [])

           return f"已完成 {len(context['completed'])} 步，但未能得出最终结论。当前进展：{context['current_findings']}"

       def _decide_next(self, context) -> dict:
           prompt = f"""
           任务：{context['task']}

           已完成的步骤：
           {format_steps(context['completed'])}

           当前发现：
           {context['current_findings']}

           待解决的问题：
           {context['remaining_questions']}

           下一步应该做什么？请用 JSON 格式输出：
           - {{"type": "action", "tool": "...", "args": {{...}}, "reason": "..."}}
           - {{"type": "answer", "content": "..."}}
           """
           return parse_decision(llm.generate(prompt, temperature=0.0))

3. Plan-and-Solve
--------------------------------------

Plan-and-Solve（Wang et al., 2023）是 Plan-then-Execute 的改进版：
在制定计划时，不仅仅罗列步骤，还要**推演每一步的预期产出**。

.. code-block:: python

   class PlanAndSolve:
       """
       Plan-and-Solve：计划 + 推演。比 Plan-then-Execute 多了一步：
       在制定计划时推演每步的可能结果和后备方案。
       """
       def run(self, task: str) -> str:
           # 计划阶段：三步走
           plan = self._create_detailed_plan(task)
           return self._execute_with_contingency(task, plan)

       def _create_detailed_plan(self, task):
           prompt = f"""
           任务：{task}

           请按以下格式制定计划：

           ## 步骤 1
           工具：search
           参数：{{"query": "A 公司的产品定价"}}
           预期产出：A 公司的定价页面 URL
           后备方案：如果搜索不到，换个 search 词搜索 "A 公司 价格"

           ## 步骤 2
           工具：read_url
           参数：{{"url": "[上一步的结果]"}}
           预期产出：A 公司的定价详情
           后备方案：如果页面打不开，搜索 A 公司定价的新闻报道

           ## 综合
           预期最终产出：A 公司与 B 公司的定价对比报告
           """
           return parse_detailed_plan(llm.generate(prompt, temperature=0.0))

三种策略的对比
================

.. list-table::
   :header-rows: 1

   * - 维度
     - Plan-then-Execute
     - Dynamic Planning
     - Plan-and-Solve
   * - 灵活性
     - 低（计划固定）
     - 高（实时调整）
     - 中（有后备方案）
   * - 稳定性
     - 高（按部就班）
     - 低（易跑偏）
     - 高（推演过风险）
   * - 错误恢复
     - 差（计划崩了就崩了）
     - 好（可以重新规划）
     - 好（提前准备了后备方案）
   * - Token 消耗
     - 低（只规划一次）
     - 高（每步都决策）
     - 中（规划更详细）
   * - 适合任务
     - 流水线、已知步骤
     - 探索、未知领域
     - 重要任务、不允许失败

退出与重规划
================

规划再好，实际执行中也难免偏离。关键是：**什么时候应该重规划？**

.. code-block:: python

   class ReplanningManager:
       """
       管理何时需要重规划。
       核心逻辑：当实际进展与计划偏差超过阈值时触发重规划。
       """
       def __init__(self, replan_threshold=0.3):
           self.planned = []
           self.actual = []
           self.threshold = replan_threshold

       def check_and_replan(self, task: str, current_step: dict) -> bool:
           """
           检查是否需要重规划。
           返回 True 表示需要。
           """
           # 计算偏差
           deviation = self._calculate_deviation(current_step)
           if deviation > self.threshold:
               # 触发重规划
               prompt = f"""
               原任务：{task}

               已完成步骤：{self.actual}

               当前步骤结果与计划偏差较大（{deviation:.0%}）。
               请根据实际进展重新规划后续步骤。
               """
               new_plan = llm.generate(prompt, temperature=0.0)
               self.planned = parse_plan(new_plan)
               return True
           return False

       def _calculate_deviation(self, step_result) -> float:
           """计算实际结果与预期的偏差程度"""
           # 简化实现：0（完全符合）到 1（完全偏离）
           return 0.0

层级规划
============

对于大型复杂任务（比如"帮我搭建一个电商网站"），一个平铺的计划
根本不够用。你需要**层级规划**：高级计划决定"做什么"，
低级计划决定"怎么做"。

.. mermaid::

   flowchart TD
       Goal[目标: 搭建电商网站] --> Level1_1[阶段1: 后端]
       Goal --> Level1_2[阶段2: 前端]
       Goal --> Level1_3[阶段3: 部署]
       Level1_1 --> Level2_1[用户模块]
       Level1_1 --> Level2_2[商品模块]
       Level1_1 --> Level2_3[订单模块]
       Level2_1 --> Level3_1[注册登录]
       Level2_1 --> Level3_2[权限管理]

.. code-block:: python

   class HierarchicalPlanner:
       """
       层级规划：高级 Agent 做战略规划，低级 Agent 做战术执行。
       """
       def run(self, task: str) -> str:
           # 战略层：分解为阶段
           phases = self._strategic_plan(task)
           phase_results = []

           for phase in phases:
               # 战术层：为每个阶段制定详细步骤
               steps = self._tactical_plan(phase)
               phase_result = self._execute_steps(steps)
               phase_results.append(phase_result)

           return self._synthesize(task, phases, phase_results)

       def _strategic_plan(self, task) -> list:
           prompt = f"""
           将以下任务分解为 3-4 个独立的阶段。
           每个阶段应该有一个清晰的目标和验收标准。

           任务：{task}
           """
           return parse_phases(llm.generate(prompt))

任务分解的常见陷阱
======================

写到这里，我想分享一些在实际部署规划型 Agent 时反复踩到的坑。

.. admonition:: 陷阱 1：过度分解
   :class: caution

   把任务拆得太细。"帮我查天气"拆成"打开浏览器 → 输入网址 →
   输入城市 → 点击搜索 → 读取结果"——每个 Agent 调用都有成本，
   过度分解让 Token 开销暴涨。经验：子任务粒度和工具调用粒度一致即可。

.. admonition:: 陷阱 2：假设结果可预测
   :class: caution

   计划假设"搜索 A 公司的定价"会返回定价页面，但可能返回的是新闻稿、
   招聘页面、或者不相关的内容。好的规划不假设"一切顺利"，
   而是准备后备方案。Plan-and-Solve 比 Plan-then-Execute 更实用的原因就在于此。

.. admonition:: 陷阱 3：规划与执行脱节
   :class: caution

   Agent 按计划执行到第三步时发现"第二步的结果和计划假设的不一样"，
   但规划已经定型了。解决方案：在每步执行后检查"是否需要重规划"。
   没有重规划机制的规划型 Agent，比纯 ReAct 更脆弱——因为它多了一层
   错误假设。

一个实用的规划模板
======================

.. code-block:: python

   PLANNING_TEMPLATE = """
   任务分析：
   {task}

   可用工具：
   {tools}

   计划：
   步骤 1: [工具名称]
     参数: [参数]
     预期: [这一步完成后应该得到什么]
     备选: [如果预期没达到怎么办]

   步骤 2: [工具名称]
     参数: 依赖步骤 1 的结果
     预期: ...
     备选: ...

   综合方案：
   [如何整合各步的结果得出最终答案]

   注意：
   - 每步依赖上一步的结果，不要假设你不知道的信息
   - 如果某步失败，使用备选方案
   - 如果备选方案也失败，向用户说明情况
   """
