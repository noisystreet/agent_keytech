.. _chapter-03-self-consistency:

===============================
自一致性（Self-Consistency）
===============================

自一致性由 Wang et al. (2022) 提出，通过多次采样推理路径并取多数结果
来提升准确率。这对 Agent 系统非常实用——单次推理可能产生幻觉，
但多次推理的一致性可以提供置信度信号。

直觉：为什么多数投票有效？
============================

想象你在做一个复杂的选择题。你不知道正确答案，但你问了 10 个朋友。
如果 8 个人选了 A，2 个人选了 B，你大概率会选 A——不是因为每个人都是
对的，而是**集体犯同样错误的概率远远小于单人犯错**。

这就是自一致性的核心直觉。单次 LLM 推理有随机性（尤其是 temperature > 0 时），
但正确的答案往往在多次采样中更稳定。

.. code-block:: python

   class SelfConsistency:
       def __init__(self, llm, n_samples=5, temperature=0.7):
           self.llm = llm
           self.n_samples = n_samples
           self.temperature = temperature

       def answer(self, question: str) -> tuple:
           """
           多次采样 → 投票 → 返回最一致的答案和置信度

           返回：(答案, 置信度)
           置信度 = 最高得票数 / 总采样数
           """
           responses = []
           for _ in range(self.n_samples):
               response = self.llm.generate(
                   question, temperature=self.temperature
               )
               answer = self._extract_answer(response)
               responses.append(answer)

           from collections import Counter
           votes = Counter(responses)
           most_common = votes.most_common(1)[0]

           confidence = most_common[1] / self.n_samples
           return most_common[0], confidence

这里有一个实操细节：**temperature 的选择直接影响多样性**。temperature=0
时每次输出都一样，自一致性没有意义。temperature=0.7 时输出开始有变化，
temperature=1.0 时变化更大但可能引入噪声。经验值是 0.5-0.7。

在 Agent 中的应用
====================

自一致性在 Agent 中有三个典型用途：

.. code-block:: python

   class AgentWithSelfConsistency:
       """用自一致性增强 Agent 的决策质量"""

       def decide_tool_call(self, task: str, tools: list, n_samples=3):
           """
           用途 1：工具调用纠错
           多次请求 LLM 决定调用哪个工具，取大多数结果。
           这可以防止"幻觉式"的工具调用。
           """
           votes = []
           for _ in range(n_samples):
               decision = self.llm.generate(
                   f"任务：{task}\n可用工具：{tools}\n请选择要调用的工具：",
                   temperature=0.5
               )
               votes.append(decision.strip())

           from collections import Counter
           selected_tool = Counter(votes).most_common(1)[0][0]
           agreement = Counter(votes).most_common(1)[0][1] / n_samples

           if agreement < 0.5:
               # 分歧大，说明 LLM 不确定，需要更多信息
               return self._ask_clarification(task)
           return selected_tool

       def verify_answer(self, question: str, candidate: str) -> bool:
           """
           用途 2：答案校验
           LLM 生成答案后，用自一致性验证答案的可靠性。
           """
           votes = []
           for _ in range(3):
               verification = self.llm.generate(
                   f"问题：{question}\n候选答案：{candidate}\n"
                   f"这个答案正确吗？请回答 '正确' 或 '错误'。",
                   temperature=0.3
               )
               votes.append(verification.strip())

           agreement = Counter(votes).most_common(1)[0][1] / 3
           return agreement >= 0.67  # 2/3 以上认为正确才通过

       def validate_step_result(self, step_result, expected_outcome):
           """
           用途 3：多步推理校验
           Agent 每完成一步，验证结果是否合理。
           """
           return self.verify_answer(
               f"期望：{expected_outcome}\n实际：{step_result}",
               "结果是否合理？"
           )

自一致性的代价
================

自一致性的主要代价是 LLM 调用次数乘以 n 倍。5 次采样 = 5 倍 LLM 调用。

.. code-block:: python

   # 成本对比
   cost_comparison = {
       "single": {"calls": 1, "cost": "$0.01", "accuracy": "80%"},
       "self-consistency_3": {"calls": 3, "cost": "$0.03", "accuracy": "87%"},
       "self-consistency_5": {"calls": 5, "cost": "$0.05", "accuracy": "90%"},
       "self-consistency_10": {"calls": 10, "cost": "$0.10", "accuracy": "92%"},
   }

   # 收益递减曲线
   # 从 1→3 次：准确率 +7%，收益显著
   # 从 3→5 次：准确率 +3%，边际递减
   # 从 5→10 次：准确率 +2%，得不偿失

经验值是 3-5 次采样，再多边际收益太低。

适用场景
============

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐
     - 原因
   * - 工具调用
     - 强烈推荐
     - 防止幻觉式工具调用，安全关键
   * - 答案校验
     - 推荐
     - 提高最终答案可靠性
   * - 简单事实查询
     - 不推荐
     - 一次就够了，多花成本无意义
   * - 创意生成
     - 不推荐
     - 创意没有"标准答案"，投票无意义
   * - 数学推理
     - 推荐
     - 答案唯一，投票效果好

.. admonition:: 自一致性 vs CoT
   :class: tip

   两者互补：
   - 自一致性是 "用多次推理的开销换准确率"
   - CoT 是 "用更多 token 思考的开销换准确率"
   可以组合使用：CoT 推理 + 自一致性校验，效果更好但开销也是叠加的。
   只在关键决策上启用这个组合，日常任务只用其中之一。
