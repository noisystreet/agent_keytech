.. _chapter-03-self-consistency:

===============================
自一致性（Self-Consistency）
===============================

自一致性由 Wang et al. (2022) 提出，通过多次采样推理路径并取多数结果来提升准确率。
这对 Agent 系统非常实用——单次推理可能产生幻觉，但多次推理的一致性可以提供置信度信号。

.. code-block:: python

   class SelfConsistency:
       def __init__(self, llm, n_samples=5, temperature=0.7):
           self.llm = llm
           self.n_samples = n_samples
           self.temperature = temperature

       def answer(self, question: str) -> str:
           # 多次采样推理路径
           candidates = []
           for _ in range(self.n_samples):
               response = self.llm.generate(
                   question, temperature=self.temperature
               )
               answer = self.extract_answer(response)
               candidates.append(answer)

           # 投票选择最一致的答案
           from collections import Counter
           votes = Counter(candidates)
           most_common = votes.most_common(1)[0][0]

           # 投票分布可以作为置信度指标
           confidence = votes.most_common(1)[0][1] / self.n_samples
           return most_common, confidence

.. admonition:: Agent 场景中的应用
   :class: application

   自一致性在 Agent 中有两个典型用途：
   1. **工具调用纠错**：多次请求 LLM 决定调用哪个工具，取大多数结果
   2. **答案校验**：LLM 生成答案后，用自一致性验证答案的可靠性
   缺点是需要多次 LLM 调用，成本和时间会增加 n 倍。
