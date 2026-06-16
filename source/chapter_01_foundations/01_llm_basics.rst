.. _chapter-01-llm-basics:

===============================
LLM 原理与架构
===============================

大语言模型（Large Language Model, LLM）是基于 Transformer 架构的神经网络模型，
通过海量文本数据预训练获得通用的语言理解和生成能力。

为什么从 LLM 原理讲起？
============================

Agent 的能力天花板由底层 LLM 决定。理解以下概念对构建 Agent 至关重要：

- **自回归生成**：LLM 逐个 token 生成输出，这个特性决定了 Agent 推理的延迟和不确定性
- **上下文窗口**：LLM 能"看到"的最大 token 数，直接影响 Agent 记忆策略的设计
- **涌现能力**：当模型规模超过某个阈值后，突然出现的推理、规划等高阶能力

.. admonition:: 缩放定律：越大越好？
   :class: story

   Kaplan 等人（2020）提出的缩放定律（Scaling Law）指出：模型性能与参数量、数据量、
   计算量呈幂律关系。但 Chinchilla（Hoffmann et al., 2022）修正了这一观点——对于给定
   的计算预算，最优方案是同时扩大模型和数据，而不是盲目增加参数量。这意味着：**训练
   Agent 使用的底座模型，不必追求最大，而应追求最适合任务**。

核心架构
============

现代 LLM 大多基于 **Decoder-only Transformer** 架构：

.. mermaid::

   flowchart LR
       Input[输入文本] --> Tokenize[Token化]
       Tokenize --> Embed[Token Embedding]
       Embed --> Pos[Position Encoding]
       Pos --> Attn[Masked Self-Attention]
       Attn --> FFN[Feed-Forward]
       FFN --> Norm[Layer Norm]
       Norm --> LM[LM Head]
       LM --> Output[输出 Token]
       Attn --> Attn
       FFN --> Attn

自回归生成
=============

.. code-block:: python

   # 自回归生成的本质：逐个预测下一个 token
   def generate(model, prompt, max_tokens=100):
       tokens = tokenize(prompt)
       for _ in range(max_tokens):
           logits = model(tokens)           # 前向传播
           next_token = sample(logits[-1])  # 采样下一个 token
           tokens.append(next_token)
           if next_token == EOS_TOKEN:      # 遇到结束符停止
               break
       return detokenize(tokens)

.. caution::

   自回归生成决定了 Agent 的 **"慢思考"** 特性：每生成一个 token 都需要一次完整的前向传播。
   这意味着 Agent 的推理延迟 = token 数量 × 单次前向时间。在实际 Agent 系统中，优化输出
   token 数量（如通过更短的思维链格式）可以直接降低延迟。
