.. _chapter-04-debate:

===============================
多 Agent 辩论
===============================

辩论（Debate）是多 Agent 系统中提升推理质量的重要机制。两个或多个 Agent
围绕同一问题展开辩论，通过互相质疑和反驳来逼近正确答案。这在复杂决策、
事实核查和多角度分析场景中尤其有效。

辩论的核心价值
====================

.. list-table::
   :header-rows: 1

   * - 问题
     - 单 Agent 回答
     - 多 Agent 辩论
   * - 偏见放大
     - 模型偏见在推理链中不断放大
     - 对方 Agent 会指出偏见并修正
   * - 确认偏误
     - 倾向于支持自己已有的观点
     - 对方提出相反证据，迫使重新审视
   * - 幻觉隐藏
     - 错误事实被包装在流畅文字中
     - 对方 Agent 发现矛盾并质疑
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

经典辩论系统实现
====================

.. code-block:: python

   class DebateSystem:
       def __init__(self, agents, judge_agent=None, rounds=3):
           self.agents = agents  # [agent_a, agent_b, ...]
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
                   # 收集其他 Agent 的观点
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

       def _format_others(self, others):
           return "\n".join(
               f"专家 {i}: {pos}" for i, pos in others.items()
           )

       def _final_judgment(self, question, positions):
           debate_log = "\n\n".join(
               f"--- 第 {i+1} 轮 ---\n{p}"
               for i, p in enumerate(self.history)
           )
           return self.judge.run(
               f"问题：{question}\n\n辩论记录：\n{debate_log}\n\n"
               f"请基于辩论中的论据质量，给出最可靠的最终答案。"
           )

       def _consensus_fallback(self, positions):
           # 无评审 Agent 时，让所有 Agent 投票
           agreement = all(
               self._are_aligned(p1, p2)
               for p1 in positions.values()
               for p2 in positions.values()
           )
           if agreement:
               return list(positions.values())[0]
           # 分歧时返回多数方观点
           return max(set(positions.values()), key=list(positions.values()).count)

辩论变体
============

1. 苏格拉底式辩论
------------------------------

不以胜负为目标，而是通过持续追问暴露推理链条中的薄弱环节。

.. code-block:: python

   class SocraticDebate:
       """苏格拉底式追问"""
       def __init__(self, agent, questioner=None, max_questions=5):
           self.agent = agent
           self.questioner = questioner  # 追问 Agent
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

为每个 Agent 分配特定角色（经济学家、工程师、伦理学家），从不同角度分析问题。

.. code-block:: python

   class RoleBasedDebate:
       def __init__(self, role_agents: dict, judge_agent):
           self.role_agents = role_agents  # {"经济学家": agent_a, ...}
           self.judge = judge_agent

       def debate(self, question: str) -> dict:
           opinions = {}
           for role, agent in self.role_agents.items():
               opinions[role] = agent.run(
                   f"你是一名{role}。请从你的专业角度分析：{question}"
               )
           # 评审综合
           summary = self.judge.run(
               f"综合以下各方的专业意见：\n{opinions}"
           )
           return {"opinions": opinions, "summary": summary}

.. admonition:: 辩论的实用建议
   :class: tip

   1. **角色差异化**：确保各 Agent 的知识背景或提示词方向不同，避免同质化
   2. **轮次控制**：3 轮左右效果最佳，过多轮次易导致观点趋同或偏离
   3. **评审独立**：评审 Agent 不应参与辩论，保持中立
   4. **证据要求**：要求每个观点附带证据来源，减少无依据的争论

参考文献
============

- Du et al., "Improving Factuality and Reasoning in Language Models through Multi-Agent Debate", 2023
- Liang et al., "Encouraging Divergent Thinking in Large Language Models", 2023
