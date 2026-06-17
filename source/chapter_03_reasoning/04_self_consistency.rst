.. _chapter-03-self-consistency:

===============================
自一致性（Self-Consistency）
===============================

自一致性由 Wang et al. (2022) 提出，通过多次采样推理路径并取多数结果
来提升准确率。这对 Agent 系统非常实用——单次推理可能产生幻觉，
但多次推理的一致性可以提供置信度信号。

.. admonition:: 自一致性与"群体的智慧"
   :class: story

   自一致性的数学基础和 1907 年 Francis Galton 发现的一个现象一脉相承：
   在一个乡村集市上，800 个人各自猜测一头牛的重量，没有人猜中准确值
   （1198 磅），但**所有人的中位数**是 1207 磅——误差不到 1%。
   这就是"群体的智慧"（Wisdom of the Crowds）：大量独立判断的平均值
   往往比任何单个专家更准确。

   自一致性正是这个思想在 LLM 中的翻版：把每个带随机性的 LLM 调用
   看作一个"独立判断者"，汇集多个判断消除个体偏差。

直觉：为什么多数投票有效？
============================

想象你在做一个复杂的选择题。你不知道正确答案，但你问了 10 个朋友。
如果 8 个人选了 A，2 个人选了 B，你大概率会选 A——不是因为每个人都是
对的，而是**集体犯同样错误的概率远远小于单人犯错**。

这就是自一致性的核心直觉。单次 LLM 推理有随机性（尤其是 temperature > 0 时），
但正确的答案往往在多次采样中更稳定。这和我们平时说的"三个臭皮匠，顶个诸葛亮"
是同一个道理——每个 LLM 调用都有独立的随机噪声，多次采样可以"平均掉"这些噪声。

但有一个前提：**采样之间必须是独立的**。如果你用 temperature=0 采样 10 次，
得到的 10 个结果完全一样，自一致性没有任何意义。只有 temperature > 0 时，
每次采样才会产生不同的输出路径，投票才有意义。

.. code-block:: python

   class SelfConsistency:
       """
       自一致性：多次采样 → 投票 → 返回最一致的答案

       temperature 的选择直接影响效果：
       - 0.0：每次输出完全一致，自一致性无效
       - 0.3：轻微变化，适合格式化输出
       - 0.5：适度多样性，推荐用于多数场景
       - 0.7：高多样性，适合创意探索
       - 1.0：可能引入过多噪声
       """
       def __init__(self, llm, n_samples=5, temperature=0.5):
           self.llm = llm
           self.n_samples = n_samples
           self.temperature = temperature

       def answer(self, question: str) -> tuple:
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

为什么不是所有场景都适合自一致性？
=====================================

我见过一些团队盲目地在所有请求上都跑自一致性，结果发现成本涨了 3 倍，
效果却没有明显提升。原因很简单：**不是所有问题都有"标准答案"。**

.. list-table::
   :header-rows: 1

   * - 场景
     - 是否推荐
     - 原因
   * - 事实问答（"首都是哪里？"）
     - 不推荐
     - 一次就够了，自一致性意义不大
   * - 数学计算
     - 推荐
     - 答案唯一，投票效果好
   * - 工具调用决策
     - 强烈推荐
     - 防止幻觉式工具调用，安全关键
   * - 创意生成
     - 不推荐
     - 没有"标准答案"，投票无意义
   * - 代码生成
     - 推荐
     - 正确代码往往在多次采样中更稳定
   * - 多步推理验证
     - 推荐
     - 每步验证，防止推理链断裂

自一致性在 Agent 中的应用
=============================

Agent 场景中，自一致性有三个最实用的用途。

1. 工具调用纠错
---------------------------------

如果 LLM 不确定该调用哪个工具，多次采样取多数可以避免"抽风式"的工具选择。

.. code-block:: python

   def decide_tool_with_consensus(task: str, tools: list, n_samples=3):
       """
       多次请求 LLM 决定调用哪个工具，取大多数结果。
       这可以防止"幻觉式"的工具调用——比如模型突然想调用一个不存在的工具。
       """
       votes = []
       for _ in range(n_samples):
           decision = llm.generate(
               f"任务：{task}\n可用工具：{tools}\n请选择要调用的工具：",
               temperature=0.5
           )
           votes.append(decision.strip())

       from collections import Counter
       selected_tool, count = Counter(votes).most_common(1)[0]
       agreement = count / n_samples

       if agreement < 0.5:
           # 分歧大，说明 LLM 不确定，需要更多信息
           return None  # 触发询问用户的逻辑
       return selected_tool

2. 答案校验
---------------------------------

Agent 生成答案后，用自一致性验证答案的可靠性。这在 Agent 得出结论前
提供了一道"自我检查"关卡。

.. code-block:: python

   def verify_with_consistency(question: str, candidate: str) -> bool:
       """
       LLM 生成答案后，验证答案的可靠性。

       如果 LLM 自己反复确认"这个答案没问题"，那可信度就高；
       如果 LLM 每次验证都在犹豫，那就不应该直接输出。
       """
       votes = []
       for _ in range(3):
           verification = llm.generate(
               f"问题：{question}\n候选答案：{candidate}\n"
               f"这个答案正确吗？请回答'正确'或'错误'。",
               temperature=0.3
           )
           votes.append(verification.strip())

       agreement = Counter(votes).most_common(1)[0][1] / 3
       return agreement >= 0.67  # 2/3 以上认为正确才通过

3. 多步推理校验
---------------------------------

Agent 每完成一步，验证该步的结果是否合理，再决定是否继续。

.. code-block:: python

   def validate_step(step_result, expected_outcome):
       """
       每步完成后自验证。如果当前步骤的结果不合理，
       就不要用它作为下一步的输入——宁可重试也不要"带错走下去"。
       """
       return verify_with_consistency(
           f"期望：{expected_outcome}\n实际：{step_result}",
           "结果是否合理？"
       )

自一致性的成本考量
====================

这是自一致性最现实的约束：**3 次采样 = 3 倍 LLM 调用开销**。

.. code-block:: python

   # 成本 vs 准确率的权衡
   tradeoff = {
       1:  {"calls": 1, "cost": "基准",     "accuracy": "基准"},
       3:  {"calls": 3, "cost": "3 倍",     "accuracy": "+5~8%"},
       5:  {"calls": 5, "cost": "5 倍",     "accuracy": "+8~12%"},
       10: {"calls": 10, "cost": "10 倍",   "accuracy": "+10~14%"},
   }

   # 数据很清楚：从 1→3 收益最大，后面边际递减
   # 对于大多数场景，3-5 次采样是最优性价比

在实践中，我会这么用：
- **关键决策** （工具调用、敏感操作确认）：5 次采样
- **常规决策** （检索策略选择、答案验证）：3 次采样
- **简单任务** （问候、常识问答）：1 次，不做自一致性

自一致性 vs CoT
=================

两者都是提升推理准确率的方法，但思路不同，适用场景也不同。

.. list-table::
   :header-rows: 1

   * - 维度
     - 自一致性
     - CoT
   * - 开销来源
     - 多次 LLM 调用（每次完整推理）
     - 更长输出（每次推理生成更多 token）
   * - 多样性来源
     - temperature > 0 采样
     - 无（CoT 本身是确定性推理）
   * - 适用场景
     - 有标准答案的任务
     - 需要逐步推理的任务
   * - 组合使用
     - 可以和 CoT 一起用（CoT + 自一致性）
     - 可以和自一致性一起用
   * - 推荐
     - 关键决策场景
     - 复杂推理场景

它们不冲突，反而可以互补：先用 CoT 做逐步推理，再用自一致性验证最终答案。
但注意开销叠加——CoT 已经增加了 token 消耗，再加上自一致性的多次调用，
成本可能翻 5-10 倍。**只在最关键的任务上同时使用两者。**



参考文献
============

- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", 2022
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", 2023
