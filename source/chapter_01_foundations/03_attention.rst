.. _chapter-01-attention:

===============================
Attention 机制
===============================

Attention 是 Transformer 的核心机制，决定了 LLM 如何关注输入序列中的不同位置。
对 Agent 开发者而言，理解 Attention 有助于设计更有效的提示词和上下文管理策略。

自注意力（Self-Attention）
============================

.. code-block:: python

   # Attention 的数学本质
   # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

   def attention(query, key, value):
       """简化版 Attention 计算"""
       d_k = query.shape[-1]
       scores = query @ key.T / (d_k ** 0.5)
       weights = softmax(scores, dim=-1)
       output = weights @ value
       return output

Agent 中的 Attention 直觉
=============================

Attention 的权重分布解释了 Agent 行为中一些反直觉的现象：

- **长上下文问题**：Softmax 将概率分配给整个序列中的 token，当上下文很长时，
  靠近中间的 token 容易被"稀释"。这意味着 Agent 的"记忆"在长对话中会自然衰减
- **注意力迷失**：Agent 在长推理链中可能"忘记"初始指令，因为早期 token 的
  注意力权重会被后续大量 token 覆盖

.. admonition:: 线性 Attention 与 Agent 的关系
   :class: application

   标准 Attention 的计算复杂度是 O(n²)，n 是序列长度。这就是为什么 LLM 的上下文
   窗口存在硬限制——每增加一倍上下文，计算量增加四倍。Agent 的长期记忆问题（如
   MySQL 表结构太长导致 Agent 无法理解）根源就在于此。**Flash Attention** 等优化
   技术通过分块和近似计算，在不显著影响质量的前提下降低了 Attention 的计算复杂度。
