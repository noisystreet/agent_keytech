.. _chapter-03-tot:

===============================
思维树（Tree-of-Thoughts）
===============================

树状思维（Tree-of-Thoughts, ToT）由 Yao et al. (2023) 提出，是思维链（CoT）
的泛化扩展。CoT 只沿着一条推理路径前进，而 ToT 在每个推理步骤进行**多方向探索**
并评估各分支的可行性，从而显著提升复杂推理和规划任务的成功率。

从链到树
============

.. list-table::
   :header-rows: 1

   * - 方法
     - 探索方式
     - 评估机制
     - 适用场景
   * - CoT
     - 单路径
     - 无中间评估
     - 简单推理
   * - Self-Consistency
     - 多路径独立采样
     - 最终答案投票
     - 有标准答案的任务
   * - ToT
     - 树状搜索
     - 每步评估剪枝
     - 复杂规划、数学、搜索

核心流程
============

ToT 包含四个关键步骤，循环执行直到找到最终答案：

.. mermaid::

   flowchart TD
       Start[初始状态] --> Expand[分支扩展<br>生成多个候选推理步]
       Expand --> Eval[状态评估<br>打分每个候选]
       Eval --> Prune[剪枝<br>保留高分分支]
       Prune --> Check{达到终止条件?}
       Check -- 否 --> Expand
       Check -- 是 --> Result[输出最终答案]

.. code-block:: python

   import copy
   from typing import List, Dict, Any

   class TreeOfThoughts:
       def __init__(self, llm, max_branches=3, max_depth=5):
           self.llm = llm
           self.max_branches = max_branches  # 每步分支数
           self.max_depth = max_depth         # 最大搜索深度

       def solve(self, problem: str) -> str:
           # 每个节点: {"thought": str, "value": float, "parent": Node}
           root = {"thought": problem, "value": 1.0, "parent": None}
           frontier = [root]

           for depth in range(self.max_depth):
               new_frontier = []

               for node in frontier:
                   # Step 1: 从当前状态扩展多个候选
                   candidates = self._expand(node, depth)
                   if not candidates:
                       continue

                   # Step 2: 评估每个候选
                   for cand in candidates:
                       cand["value"] = self._evaluate(cand)

                   # Step 3: 按分数降序排列，取 top-K 继续探索
                   candidates.sort(key=lambda x: x["value"], reverse=True)
                   new_frontier.extend(candidates[:self.max_branches])

               frontier = new_frontier

               # 检查是否已有足够好的最终答案
               for node in frontier:
                   if self._is_solution(node):
                       return self._extract_answer(node)

           # 返回最优路径的答案
           best = max(frontier, key=lambda x: x["value"])
           return self._extract_answer(best)

       def _expand(self, node: Dict, depth: int) -> List[Dict]:
           """从当前节点生成多个后续推理步骤"""
           prompt = (
               f"问题：{node['thought']}\n"
               f"当前进度：第 {depth + 1} 步\n"
               f"请列出 {self.max_branches} 种不同的下一步推理方向，"
               f"每行一个。"
           )
           response = self.llm.generate(prompt, temperature=0.8)
           branches = response.strip().split("\n")

           return [
               {"thought": f"{node['thought']} → {b}", "value": 0, "parent": node}
               for b in branches[:self.max_branches]
           ]

       def _evaluate(self, node: Dict) -> float:
           """评估当前推理路径的可行性"""
           prompt = (
               f"当前推理路径：{node['thought']}\n"
               f"这条推理路径是否合理？请给出 0-1 之间的分数。"
           )
           response = self.llm.generate(prompt, temperature=0.0)
           return float(response.strip())

       def _is_solution(self, node: Dict) -> bool:
           """判断是否已达到最终答案"""
           return "答案是" in node["thought"] or "最终答案" in node["thought"]

       def _extract_answer(self, node: Dict) -> str:
           """沿父指针回溯完整推理路径"""
           path = []
           current = node
           while current:
               path.append(current["thought"])
               current = current["parent"]
           return "\n".join(reversed(path))

.. admonition:: BFS vs DFS
   :class: tip

   ToT 支持两种搜索策略：
   - **广度优先（BFS）**：每层保留 top-K 节点，适合深度较浅、分支较多的任务
   - **深度优先（DFS）**：优先深入探索最有希望的分支，可回溯，适合深度较大的任务

   Agent 场景中一般推荐 BFS——Agent 任务的分支通常较多（工具选择多），
   但推理深度有限（通常 3-5 步即可完成）。

Agent 中的应用
====================

ToT 在 Agent 场景中有三个典型应用：

1. **工具选择规划**：面对多个可用工具，Agent 用 ToT 探索不同工具组合的路径
2. **多步任务分解**：复杂任务如"帮我规划一次旅行"，每个分支对应不同的方案
3. **错误恢复**：当某个路径的执行结果不理想时，回溯到上游分支尝试其他方案

.. code-block:: python

   # Agent 场景：工具选择中的 ToT
   task = "查一下本周 AI 领域的重大新闻，并总结成简报"

   # 可能的工具组合路径
   branches = [
       "搜索新闻 → 用浏览器打开前 5 条 → 提取内容 → 总结",     # 路径 A
       "搜索新闻 → 只打开引用最多的 3 条 → 提取 → 总结",         # 路径 B
       "直接搜索 '本周 AI 新闻摘要' → 提取 → 总结",              # 路径 C
   ]

.. admonition:: 成本考量
   :class: warning

   ToT 的 LLM 调用量远高于 CoT——最坏情况下调用次数为 O(branches×depth)。
   生产环境中建议限制搜索深度（depth≤3）或引入缓存机制。

参考文献
============

- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", 2023
- Long, "Large Language Model Guided Tree-of-Thought", 2023
