.. _chapter-03-tot:

===============================
思维树（Tree-of-Thoughts）
===============================

树状思维（Tree-of-Thoughts, ToT）由 Yao et al. (2023) 提出，是思维链（CoT）
的泛化扩展。CoT 只沿着一条推理路径前进，而 ToT 在每个推理步骤进行**多方向探索**
并评估各分支的可行性，从而显著提升复杂推理和规划任务的成功率。

为什么要用树而不是链？
========================

我见过的读者第一次接触 ToT 时最常见的反应是："CoT 已经效果很好了，
为什么要搞得这么复杂？"

答案是：**有些问题不是"逐步推理"能解决的。**

考虑这个问题："从 1 到 9 中选择三个不重复的数字，使它们的和为 15。"
CoT 可能会试错一次——"5+4+6=15，找到了！"但如果答案是错的，CoT 没有
回头路可走。它只能沿着当前路径走下去，越走越偏。

ToT 的做法是：同时尝试多个不同的组合，快速淘汰明显错的，只深入探索
有希望的方案。这其实就是**人类解决问题的方式**——你不会只试一条路，
而是想好几种可能，然后排除最不靠谱的，再深入验证最有可能的。

.. list-table::
   :header-rows: 1

   * - 方法
     - 探索方式
     - 评估机制
     - 适用场景
   * - CoT
     - 单路径，无回头路
     - 无中间评估，一步错步步错
     - 简单推理、有标准流程的任务
   * - Self-Consistency
     - 多路径独立采样，互不干扰
     - 最终答案投票，不关心中间过程
     - 有标准答案的任务、数学题
   * - ToT
     - 树状搜索，可回溯可剪枝
     - 每步评估，提前淘汰坏分支
     - 复杂规划、开放域搜索、博弈

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
           root = {"thought": problem, "value": 1.0, "parent": None}
           frontier = [root]

           for depth in range(self.max_depth):
               new_frontier = []

               for node in frontier:
                   candidates = self._expand(node, depth)
                   if not candidates:
                       continue

                   for cand in candidates:
                       cand["value"] = self._evaluate(cand)

                   candidates.sort(key=lambda x: x["value"], reverse=True)
                   new_frontier.extend(candidates[:self.max_branches])

               frontier = new_frontier

               for node in frontier:
                   if self._is_solution(node):
                       return self._extract_answer(node)

           best = max(frontier, key=lambda x: x["value"])
           return self._extract_answer(best)

       def _expand(self, node: Dict, depth: int) -> List[Dict]:
           """从当前节点生成多个后续推理步骤"""
           prompt = (
               f"问题：{node['thought']}\n"
               f"当前进度：第 {depth + 1} 步\n"
               f"请列出 {self.max_branches} 种不同的下一步推理方向，每行一个。"
           )
           response = self.llm.generate(prompt, temperature=0.8)
           branches = response.strip().split("\n")
           return [
               {"thought": f"{node['thought']} → {b}", "value": 0, "parent": node}
               for b in branches[:self.max_branches]
           ]

BFS vs DFS：两种搜索策略
============================

ToT 支持两种搜索策略，选择哪种取决于任务特点。

**广度优先（BFS）**：每层保留 top-K 节点，逐层推进。

.. code-block:: python

   class BFSToT(TreeOfThoughts):
       """广度优先：每层保留最好的 K 个节点"""
       def solve(self, problem):
           nodes = [{"thought": problem, "value": 1.0, "depth": 0}]

           for depth in range(self.max_depth):
               # 扩展当前层的所有节点
               all_candidates = []
               for node in nodes:
                   candidates = self._expand(node, depth)
                   all_candidates.extend(candidates)

               # 对所有候选评分并排序
               for cand in all_candidates:
                   cand["value"] = self._evaluate(cand)

               all_candidates.sort(key=lambda x: x["value"], reverse=True)

               # 只保留 top-K
               nodes = all_candidates[:self.max_branches]

               # 检查是否有解
               for node in nodes:
                   if self._is_solution(node):
                       return self._extract_answer(node)

           return self._extract_answer(nodes[0])

**深度优先（DFS）**：优先深入一条最有希望的分支，失败时回溯。

.. code-block:: python

   class DFSToT(TreeOfThoughts):
       """深度优先：先深入探索最有希望的分支，不行再回溯"""
       def solve(self, node, depth=0):
           if self._is_solution(node):
               return self._extract_answer(node)

           if depth >= self.max_depth:
               return None

           candidates = self._expand(node, depth)
           for cand in candidates:
               cand["value"] = self._evaluate(cand)

           candidates.sort(key=lambda x: x["value"], reverse=True)

           for cand in candidates:
               result = self.solve(cand, depth + 1)
               if result:
                   return result

           return None  # 回溯

.. list-table::
   :header-rows: 1

   * - 维度
     - BFS
     - DFS
   * - 内存消耗
     - 高（需要保存整层的节点）
     - 低（只保存当前路径）
   * - 搜索完整性
     - 更完整（不会漏掉浅层的解）
     - 可能漏掉浅层分支
   * - 适合场景
     - 深度浅、分支多的任务
     - 深度大、有明确探索方向
   * - Agent 推荐
     - 默认选择（3-5 步的任务居多）
     - 需要长链推理时使用

ToT 与 MCTS 的结合
====================

ToT 的一个重要的改进方向是引入**蒙特卡洛树搜索（MCTS）**。
MCTS 通过模拟（rollout）来评估节点的价值，而不是直接让 LLM 打分。

.. code-block:: python

   class MCTSNode:
       """MCTS 树节点"""
       def __init__(self, state, parent=None):
           self.state = state
           self.parent = parent
           self.children = []
           self.visits = 0
           self.value = 0.0

       def ucb_score(self, exploration_weight=1.4):
           """UCB（Upper Confidence Bound）公式"""
           if self.visits == 0:
               return float("inf")
           exploitation = self.value / self.visits
           exploration = exploration_weight * sqrt(log(self.parent.visits) / self.visits)
           return exploitation + exploration

MCTS 比纯 LLM 评估更客观——它不是问 LLM"这个方案好不好"（LLM 可能给出
有偏见的评估），而是通过实际"尝试"一个分支来验证它的价值。
代价是每次 rollout 都需要额外的 LLM 调用。

ToT 在 Agent 中的应用
=========================

1. 工具选择规划
------------------------------

面对多个可用工具，Agent 用 ToT 探索不同工具组合的路径。

.. code-block:: python

   task = "查一下本周 AI 领域的重大新闻，并总结成简报"

   # ToT 生成的多条工具调用路径
   branches = [
       "搜索新闻 → 用浏览器打开前 5 条 → 提取内容 → 总结",
       "搜索新闻 → 只打开引用最多的 3 条 → 提取 → 总结",
       "直接搜索 '本周 AI 新闻摘要' → 提取 → 总结",
   ]

2. 错误恢复
------------------------------

当 Agent 的某个路径执行结果不理想时，回溯到上游分支尝试其他方案。

.. code-block:: python

   class AgentWithToTRecovery:
       """用 ToT 做错误恢复"""
       def run(self, task: str) -> str:
           root = {"path": [], "result": None, "parent": None}

           while True:
               # 尝试当前最优路径
               current = self._best_path()
               result = self._execute_path(current)

               if result["success"]:
                   return result["output"]

               # 当前路径失败，回溯到上一个分支点
               backtrace = self._find_alternative(current)
               if not backtrace:
                   return f"所有路径都失败: {result['error']}"

               current = backtrace

参考文献
============

- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", 2023
- Long, "Large Language Model Guided Tree-of-Thought", 2023
