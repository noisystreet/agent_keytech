.. _chapter-03-cot:

===============================
思维链（Chain-of-Thought）
===============================

思维链（Chain-of-Thought, CoT）是提示 LLM 逐步推理而非直接给出答案的技术。
Wei et al. (2022) 发现，在提示词中加入"Let's think step by step"可以
显著提升复杂推理任务的准确率。

为什么 CoT 有效？
========================

CoT 的核心价值在于将隐式推理过程**显式化**。LLM 本质上是一个"next token predictor"，
直接给出答案时，模型需要在单次前向传播中完成所有隐含推理。而 CoT 通过
引导模型逐步展开推理过程，将复杂问题分解为多个简单步骤，每一步都更易生成正确 token。

.. list-table::
   :header-rows: 1

   * - 方面
     - 直接回答
     - CoT 逐步推理
   * - 推理过程
     - 隐含在单次生成中
     - 每一步显式输出
   * - 错误定位
     - 只看到错误答案
     - 可以定位到哪一步出错
   * - 人类可解释
     - 差
     - 好
   * - 复杂推理
     - 困难
     - 显著提升
   * - Token 消耗
     - 少
     - 多（但值得）

三种 CoT 变体
===================

1. Zero-shot CoT
------------------------------

最简单的实现方式——在问题后追加"逐步推理"指令。

.. code-block:: python

   def zero_shot_cot(llm, question: str) -> str:
       prompt = f"""
       问题：{question}

       逐步推理（Let's think step by step）：
       """
       reasoning = llm.generate(prompt, temperature=0.0)
       return reasoning

2. Few-shot CoT
------------------------------

提供推理示例（Chain-of-Thought Prompting），引导模型模仿正确推理模式。

.. code-block:: python

   def few_shot_cot(llm, question: str) -> str:
       prompt = """
       问题：教室有 5 排椅子，每排 6 把，坐了 12 个人，还有多少个空位？
       推理：5 排 × 6 把 = 30 把椅子。30 - 12 = 18 个空位。答案是 18。

       问题：商店有 23 个苹果，卖出 7 个后又进货 15 个，现在有多少个？
       推理：
       """
       reasoning = llm.generate(prompt, temperature=0.0)
       return reasoning

3. Auto-CoT
------------------------------

为解决手动设计 Few-shot 示例的人力成本，Auto-CoT（Zhang et al., 2022）
自动从数据集中选取多样化示例并生成推理链。

.. code-block:: python

   class AutoCoT:
       def __init__(self, llm, example_pool: list):
           self.llm = llm
           self.example_pool = example_pool  # 预置示例池

       def generate(self, question: str, k: int = 3) -> str:
           # 选择与当前问题最相关的 k 个示例
           selected = self._select_diverse_examples(question, k)
           example_text = "\n\n".join(
               f"问题：{ex['question']}\n推理：{ex['reasoning']}\n答案：{ex['answer']}"
               for ex in selected
           )
           prompt = f"{example_text}\n\n问题：{question}\n推理："
           return self.llm.generate(prompt, temperature=0.0)

       def _select_diverse_examples(self, question, k):
           # 使用向量相似度选择多样化的示例
           return self.example_pool[:k]  # 简化实现

.. admonition:: 在实际 Agent 中的应用
   :class: application

   Agent 场景中，CoT 有三个典型用途：
   1. **工具调用决策**：让 LLM 逐步分析应该调用哪个工具
   2. **多步推理**：复杂任务拆解为可执行的子步骤
   3. **错误回溯**：当 Agent 执行失败时，用 CoT 分析失败原因

Limitations
============

.. list-table::
   :header-rows: 1

   * - 局限
     - 说明
     - 缓解方案
   * - 成本增加
     - CoT 生成大量中间 token，延迟和费用上升
     - 简单问题走直接回答，复杂问题走 CoT
   * - 虚假推理
     - LLM 可能生成"看起来合理"但实际错误的推理链
     - 结合工具验证每一步
   * - 不适用于所有任务
     - 对常识推理帮助不大，主要用于数学、逻辑、规划
     - 根据任务类型选择策略
   * - 推理链过长
     - 超过一定长度后，中间步骤质量下降
     - 分段推理 + 摘要

参考文献
============

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", 2022
- Kojima et al., "Large Language Models are Zero-Shot Reasoners", 2022
- Zhang et al., "Automatic Chain of Thought Prompting in Large Language Models", 2022
