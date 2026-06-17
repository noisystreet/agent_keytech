.. _chapter-04-debate:

===============================
多 Agent 辩论
===============================

辩论（Debate）是多 Agent 系统中提升推理质量的重要机制。两个或多个 Agent
围绕同一问题展开辩论，通过互相质疑和反驳来逼近正确答案。这在复杂决策、
事实核查和多角度分析场景中尤其有效。

辩论解决的是一类很特殊的问题：**当你可以问多个"人"但不确定该信谁时，
最好的办法是让他们吵一架。**

这和现实生活中的决策很相似——如果你只有一个顾问，他说的你可能全信也可能
全不信。但如果你有两个顾问，让他们各自提出观点、然后互相反驳，
你就能看到每个观点的弱点在哪里，做出更明智的判断。

为什么辩论有效？
====================

单 Agent 推理有一种"自动驾驶"的倾向：一旦模型开始朝着某个方向推理，
它倾向于沿着这个方向继续走下去，即使方向不对也很难自我纠正。

辩论引入了一个"对手"——你有一个推理方向不合理的 Agent，另一个 Agent
会直接指出来。这个外部纠正信号比 Agent 的自我纠正要强得多。

.. list-table::
   :header-rows: 1

   * - 问题
     - 单 Agent 回答
     - 多 Agent 辩论
   * - 偏见放大
     - 模型偏见在推理链中自我强化
     - 对方 Agent 直接指出偏见
   * - 确认偏误
     - 倾向于支持已有的初始判断
     - 对方提出相反证据，迫使重新审视
   * - 幻觉隐藏
     - 错误事实被包装在流畅的文字中
     - 对方 Agent 发现事实矛盾并质疑
   * - 视角局限
     - 只有一种分析视角
     - 多方辩论覆盖多种视角

.. mermaid::

   flowchart TD
       Q[问题] --> A1[Agent A<br>正方]
       Q --> A2[Agent B<br>反方]
       A1 --> Position1[初始观点]
       A2 --> Position2[初始观点]
       Position1 --> Round1[辩论第 1 轮]
       Position2 --> Round1
       Round1 --> Rebuttal1[A 反驳 B]
       Round1 --> Rebuttal2[B 反驳 A]
       Rebuttal1 --> Round2[辩论第 2 轮]
       Rebuttal2 --> Round2
       Round2 --> Judge[评审 Agent<br>综合判断]
       Judge --> Answer[最终答案]

辩论系统的关键设计
====================

一个辩论系统有几个容易搞反的设计点，值得单独拿出来讲。

**1. 什么情况下辩论反而有害？**

如果两个 Agent 的知识背景和提示词方向完全一样，它们的"辩论"就只是
两个相同的人互相重复对方的观点——没有新信息，只有 token 浪费。
所以辩论的前提是**角色差异化**：给 A 和 B 不同的角色、不同的提示词方向、
甚至不同的参考材料。

**2. 辩论轮次多少合适？**

3 轮左右效果最好。1 轮往往不够（双方刚表述完立场），5 轮以上容易
观点趋同（"我说了 5 遍了，你还不信，算了我也同意你吧"）或
偏离主题（越吵越远）。

**3. 评审 Agent 该不该参与辩论？**

不该。评审 Agent 一旦参与辩论，它就失去了中立性。它应该只看到
完整的辩论记录，然后做出判断。

经典辩论系统实现
====================

.. code-block:: python

   class DebateSystem:
       """
       多轮辩论系统。

       关键设计：
       - 每个 Agent 独立维护立场
       - 每轮互相反驳
       - 评审 Agent 在最后做综合判断
       """
       def __init__(self, agents, judge_agent=None, rounds=3):
           self.agents = agents
           self.judge = judge_agent
           self.rounds = rounds
           self.history = []

       def debate(self, question: str) -> str:
           # 初始化各方立场
           positions = {}
           for i, agent in enumerate(self.agents):
               positions[i] = agent.run(
                   f"{question}\n请给出你的初步观点，包含理由和证据。"
               )
           self.history.append(positions)

           # 多轮辩论
           for r in range(self.rounds):
               new_positions = {}
               for i, agent in enumerate(self.agents):
                   others = {j: p for j, p in positions.items() if j != i}
                   rebuttal = agent.run(
                       f"原问题：{question}\n\n"
                       f"其他专家的观点：\n{self._format_others(others)}\n\n"
                       f"请逐一反驳其他观点中的错误，并提出修正后的立场。"
                   )
                   new_positions[i] = rebuttal
               positions = new_positions
               self.history.append(positions)

           if self.judge:
               return self._final_judgment(question, positions)
           return self._consensus_fallback(positions)

       def _final_judgment(self, question, positions):
           """评审 Agent 综合判断"""
           debate_log = "\n\n".join(
               f"--- 第 {i+1} 轮 ---\n{p}"
               for i, p in enumerate(self.history)
           )
           return self.judge.run(
               f"问题：{question}\n\n辩论记录：\n{debate_log}\n\n"
               f"请基于辩论中的论据质量，给出最可靠的最终答案。"
           )

辩论变体
============

1. 苏格拉底式辩论
------------------------------

苏格拉底式辩论不以"谁对谁错"为目标。它的目的是通过持续追问，
**暴露推理链条中的薄弱环节**。

.. code-block:: python

   class SocraticDebate:
       """
       苏格拉底式追问。
       核心思路：不判断对错，只追问"你凭什么这么认为"。

       适用场景：
       - 验证推理的严谨性
       - 发现知识盲区
       - 自我审查（单 Agent 场景也可用）
       """
       def __init__(self, agent, questioner=None, max_questions=5):
           self.agent = agent
           self.questioner = questioner
           self.max_questions = max_questions

       def run(self, topic: str) -> dict:
           answer = self.agent.run(topic)
           qa_pairs = [{"q": "请阐述你的观点", "a": answer}]

           for _ in range(self.max_questions):
               question = self.questioner.run(
                   f"针对以下回答，提出一个深入追问：{answer}"
               )
               answer = self.agent.run(question)
               qa_pairs.append({"q": question, "a": answer})

           return qa_pairs

2. 多角色模拟辩论
------------------------------

为每个 Agent 分配特定角色，从不同角度分析问题。这种方式特别适合
需要多维度分析的场景——比如"这个商业计划可行吗？"可以经济学家、
工程师、市场营销三个角色分别分析。

.. code-block:: python

   class RoleBasedDebate:
       """
       多角色模拟辩论。
       每个 Agent 扮演一个专业角色，从特定视角分析问题。
       """
       def __init__(self, role_agents: dict, judge_agent):
           self.role_agents = role_agents
           self.judge = judge_agent

       def debate(self, question: str) -> dict:
           opinions = {}
           for role, agent in self.role_agents.items():
               opinions[role] = agent.run(
                   f"你是一名{role}。请从你的专业角度分析：{question}"
               )
           summary = self.judge.run(
               f"综合以下各方的专业意见：\n{opinions}"
           )
           return {"opinions": opinions, "summary": summary}

辩论的成本和效果权衡
=====================

辩论不是免费的。每次辩论意味着多次 LLM 调用。

.. code-block:: text

   一场 3 轮辩论的成本：
   - 2 个 Agent × 3 轮辩论 = 6 次 LLM 调用
   - 1 次评审 Agent 调用
   - 总计：7 次 LLM 调用

   相比单 Agent 的直接回答（1 次调用），成本是 7 倍。
   但准确率提升通常在 10-30% 之间。

什么时候值得付出这 7 倍的成本？

- **高风险决策** （代码审查、合同分析）：值得——一个错误可能造成很大损失
- **事实核查** （涉及关键信息准确性）：值得——幻觉可能误导决策
- **日常问答** （"今天天气怎么样？"）：不值得——单 Agent 就够了

.. admonition:: 辩论的实用建议
   :class: tip

   1. **角色差异化**：确保各 Agent 的知识背景不同，否则辩论没有意义
   2. **轮次控制**：3 轮左右效果最佳，过多轮次易导致观点趋同或偏离
   3. **评审独立**：评审 Agent 不应参与辩论，保持中立
   4. **证据要求**：要求每个观点附带证据来源，减少无依据的争论
   5. **成本意识**：只在关键任务上启用辩论机制

参考文献
============

- Du et al., "Improving Factuality and Reasoning in Language Models through Multi-Agent Debate", 2023
- Liang et al., "Encouraging Divergent Thinking in Large Language Models", 2023
