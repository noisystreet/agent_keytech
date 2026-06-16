.. _chapter-04-debate:

===============================
多 Agent 辩论
===============================

辩论（Debate）是多 Agent 系统中提升推理质量的重要机制。两个或多个 Agent 围绕
同一问题展开辩论，通过互相质疑和反驳来逼近正确答案。

.. code-block:: python

   class DebateSystem:
       """两个 Agent 就同一问题展开辩论"""
       def __init__(self, agent_a, agent_b, rounds=3):
           self.a = agent_a
           self.b = agent_b
           self.rounds = rounds

       def debate(self, question: str) -> str:
           a_pos = self.a.run(f"{question}\n请给出你的初步观点")
           b_pos = self.b.run(f"{question}\n请给出你的初步观点")

           for r in range(self.rounds):
               a_pos = self.a.run(
                   f"对方的观点是：{b_pos}\n请反驳并提出你的修正观点"
               )
               b_pos = self.b.run(
                   f"对方的观点是：{a_pos}\n请反驳并提出你的修正观点"
               )

           # 最终综合双方观点
           summary = self.a.run(
               f"总结辩论要点：\nAgent A: {a_pos}\nAgent B: {b_pos}"
           )
           return summary
