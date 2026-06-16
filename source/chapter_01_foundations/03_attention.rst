.. _chapter-01-attention:

===============================
Attention 机制
===============================

Attention 是 Transformer 的"心脏"。如果你只理解 Transformer 的一个组件，
那应该就是 Attention。它对 Agent 开发者的影响远超你想象——注意力机制的
行为决定了你的 Agent 能"记住"多少信息、会不会在长对话中"迷失"、
以及推理链能做多长。

但我要先澄清一个常见的误解：很多人把 Attention 想象成"模型在关注什么"，
好像模型有一双眼睛在看输入文本的不同位置。这个类比不太准确。

更准确的类比是**信息检索系统**：Q（Query）是你当前关心的问题，
K（Key）是所有候选信息的索引，V（Value）是信息本身。Attention 就是在
你关心的问题和所有信息之间做一次"检索"，找出最相关的信息来使用。

.. code-block:: python

   # Attention 的直观理解
   def attention_as_retrieval(query, keys, values):
       """
       query: 当前 token 的"问题"
       keys: 所有 token 的"索引"
       values: 所有 token 的"信息内容"

       步骤：
       1. 用 query 去匹配每个 key，得到注意力分数
       2. 将分数转化为权重（softmax）
       3. 用权重加权组合所有 value
       """
       scores = [dot_product(query, key) for key in keys]  # 匹配
       weights = softmax(scores)                            # 归一化
       output = sum(w * v for w, v in zip(weights, values)) # 加权
       return output

为什么这对 Agent 重要？
=========================

Agent 的每一步推理都依赖 Attention 去"回顾"之前的上下文。
当上下文变长时，Attention 的行为会发生一些反直觉的变化。

.. list-table::
   :header-rows: 1

   * - 上下文长度
     - Attention 行为
     - 对 Agent 的影响
   * - 短（< 2K）
     - 每个 token 都能获得足够的注意力
     - Agent 能准确"记住"用户指令
   * - 中等（2K-8K）
     - 开头和结尾的 token 获得更多注意力
     - Agent 可能"遗忘"中间的工具返回结果
   * - 长（8K-64K）
     - 注意力在大量 token 间均摊
     - Agent 在长推理链中"丢失"初始目标
   * - 超长（> 64K）
     - 信号淹没在噪声中
     - 召回率显著下降，需要外部记忆补偿

这个现象被 Liu et al. 总结为 "Lost in the Middle"（2023）：
当相关信息位于输入中间位置时，LLM 的召回率比放在开头或结尾时低很多。
这对 Agent 意味着：

1. **把最重要的指令放在 System Prompt（开头）**
2. **最近的工具返回结果放在对话末尾**
3. **中间放较不重要的历史对话**

等等，这和 Attention 的数学有什么关系？让我们来看。

Attention 的数学本质
=======================

.. code-block:: python

   # 缩放点积注意力：
   # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

   def scaled_dot_product_attention(query, key, value):
       """
       Q: [batch, seq_len, d_k]   查询
       K: [batch, seq_len, d_k]   键
       V: [batch, seq_len, d_v]   值
       """
       d_k = query.shape[-1]
       # Q @ K^T: [batch, seq_len, seq_len] 注意力矩阵
       scores = query @ key.transpose(-2, -1)  # 点积相似度
       scores = scores / (d_k ** 0.5)          # 缩放，防止 softmax 进入饱和区
       weights = softmax(scores, dim=-1)       # 行归一化
       output = weights @ value                 # 加权求和
       return output

为什么除以 sqrt(d_k)？如果你对数值计算敏感，可能会注意到这个问题。
假设 d_k=1024，不缩放时 Q@K^T 的方差大约是 d_k 的量级（因为点积是
d_k 个元素的和），这意味着分数的范围可能达到 [-100, 100] 甚至更大。
在这个范围上做 softmax，所有概率会极端集中在最大值附近，其他位置的
梯度几乎为零——模型学不动了。除以 sqrt(d_k) 把方差拉回到 1 左右，
梯度才能正常传播。

因果掩码（Causal Masking）
===============================

LLM 在生成文本时，**不能看到未来的 token**。这就是因果掩码的作用。

.. code-block:: python

   def causal_attention(query, key, value):
       """
       Decoder-only 的因果注意力。
       mask 是一个上三角矩阵：当前位置只能看到自己和之前的位置。
       """
       d_k = query.shape[-1]
       scores = query @ key.transpose(-2, -1) / (d_k ** 0.5)

       # 创建因果掩码：上三角为 -inf
       seq_len = scores.shape[-1]
       mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
       scores = scores.masked_fill(mask == 1, float("-inf"))

       weights = softmax(scores, dim=-1)
       return weights @ value

.. mermaid::

   flowchart LR
       subgraph 因果掩码示例
           direction LR
           T1[token 1] --> T1
           T2[token 2] --> T1
           T2 --> T2
           T3[token 3] --> T1
           T3 --> T2
           T3 --> T3
       end

虽然训练时使用的是因果掩码，但推理时所有的 prompt token 都是可见的。
这意味着 Agent 的完整 System Prompt + 对话历史在推理时是一次性"看到"的。
Attention 的权重分布决定了 Agent 会重点关注哪些信息。

多头注意力（Multi-Head Attention）
========================================

多头注意力的设计原理比大多数人理解的更精妙。

.. code-block:: python

   class MultiHeadAttention:
       """
       多头注意力：用多组 Q/K/V 权重并行计算不同层面的注意力。
       每个"头"可以关注不同方面的语义关系。

       比如在 Agent 推理中：
       - 头 1：关注当前工具调用的参数
       - 头 2：关注原始用户指令
       - 头 3：关注前一步的推理结果
       - 头 4：关注工具返回的结果摘要
       """
       def __init__(self, d_model=1024, n_heads=16):
           self.d_model = d_model
           self.n_heads = n_heads
           self.d_head = d_model // n_heads

           # 每个头有自己的 Q/K/V 权重
           self.W_q = [nn.Linear(d_model, self.d_head) for _ in range(n_heads)]
           self.W_k = [nn.Linear(d_model, self.d_head) for _ in range(n_heads)]
           self.W_v = [nn.Linear(d_model, self.d_head) for _ in range(n_heads)]

       def forward(self, x):
           # 每个头独立计算注意力
           heads = []
           for i in range(self.n_heads):
               q = self.W_q[i](x)
               k = self.W_k[i](x)
               v = self.W_v[i](x)
               head = scaled_dot_product_attention(q, k, v)
               heads.append(head)

           # 拼接所有头的输出
           concat = torch.cat(heads, dim=-1)
           return concat

多头设计中有一个有趣的发现（来自 Anthropic 的 Transformer 电路分析）：
**不同的头有不同的"分工"**。有些头关注语法关系（主语-谓语），有些关注
位置关系（前一个 token），有些关注语义相似性。在 Agent 场景中，
可能有些头专门负责跟踪"当前在哪一步"，有些负责"用户的要求是什么"。

KV Cache：推理的加速器
=========================

KV Cache 是 LLM 推理中最关键的优化。理解它才能理解为什么 Agent 的
每一步推理需要多长时间。

.. code-block:: python

   class KVCache:
       """
       Key-Value 缓存：自回归生成时，缓存已计算的 K 和 V。

       为什么需要 KV Cache？
       生成 token t 时，需要计算所有前序 token 对 t 的注意力。
       但前序 token 之间的注意力在上一步已经算过了——
       没必要重算！只需要计算当前 token 对之前所有 token 的注意力。

       没有 KV Cache：每步计算量和序列长度成正比 → O(n²)
       有 KV Cache：  每步计算量恒定 → O(n)
       """
       def __init__(self, n_layers=32):
           # 每层缓存一组 K 和 V
           self.k_cache = [{} for _ in range(n_layers)]
           self.v_cache = [{} for _ in range(n_layers)]

       def update(self, layer, k, v):
           """添加当前 step 的 K 和 V"""
           step = len(self.k_cache[layer])
           self.k_cache[layer][step] = k
           self.v_cache[layer][step] = v

       def get(self, layer):
           """获取到当前 step 为止的所有 K 和 V"""
           return self.k_cache[layer], self.v_cache[layer]

这就是为什么 Agent 的第一步推理（处理所有 prompt token）是最慢的，
后续每一步（只生成一个 token）要快得多。因为第一步没有 KV Cache
可用，需要计算所有 token 之间的注意力。

Flash Attention：让长上下文成为可能
=======================================

标准 Attention 的计算复杂度是 O(n²)，n 是序列长度。这意味着：
- 8K 上下文：8K² = 64M 次运算
- 32K 上下文：32K² = 1B 次运算
- 128K 上下文：128K² = 16B 次运算

Flash Attention 通过**分块计算**和**重计算**将复杂度降到了接近 O(n)
的实际水平。它的核心思路是：不一次性计算整个注意力矩阵，而是分块计算，
每块只保留必要的输出，中间结果在反向传播时重新计算（用计算换显存）。

.. code-block:: python

   # Flash Attention 的核心思路（伪代码）
   def flash_attention(Q, K, V, block_size=128):
       """
       Q: [seq_len, d_k]  但实际是分块处理的
       标准 attention 需要存储 [seq_len, seq_len] 的注意力矩阵。
       Flash attention 只需要存储 block_size × block_size。
       """
       output = zeros_like(Q)
       for q_block in split(Q, block_size):
           for k_block in split(K, block_size):
               # 计算这个小块的注意力
               scores = q_block @ k_block.T / sqrt(d_k)
               weights = softmax(scores)
               # 累加结果
               output[q_block_idx] += weights @ v_block
       return output

对于 Agent 开发者来说，Flash Attention 意味着：**你可以用更大的上下文窗口
而无需担心 GPU 显存爆炸**。这也是为什么 2024 年后的模型支持 128K+
上下文窗口——技术上早就可行，但如果没有 Flash Attention，
128K 的注意力计算会让 GPU 显存和算力都吃不消。

注意力模式的可视化
======================

如果你用工具（如 BertViz、AttentionViz）可视化 Attention 权重，
会看到一些有趣的模式：

.. code-block:: text

   输入: "查一下 [北京] 的天气"

   Head 3 的注意力权重（关注"北京"）：
   查  → 北京: 0.15
   一  → 北京: 0.10
   下  → 北京: 0.05
   北  → 北京: 0.60  ← 重点关注当前位置
   京  → 北京: 0.50
   的  → 北京: 0.30
   天  → 北京: 0.20
   气  → 北京: 0.15

观察 Attention 模式可以帮助你诊断 Agent 的问题。比如，如果你发现
Agent 经常"忘记"用户指令，可能是 Attention 在长序列中的权重分布
过于分散。解决方案不是改模型，而是调整你的提示词结构——
把关键指令放在开头和结尾，避开"注意力中间地带"。

对 Agent 开发者的实践建议
==============================

1. **提示词结构影响注意力分布**——最重要的指令放在 System Prompt（开头）
   和最末尾的 user message（结尾）
2. **中间位置是注意力"洼地"**——历史对话和中间推理步骤放中间，
   但不要放关键信息
3. **长的工具返回结果会稀释注意力**——工具返回尽量精简，长结果用
   摘要替代
4. **Attention 有头分工**——不要期望一个提示词覆盖所有需求，
   分开写不同职责的提示词段
5. **KV Cache 不等于记忆**——KV Cache 只是缓存在当前上下文中，
   不等于长期记忆。跨会话的记忆需要外部存储
6. **Flash Attention 让你能用更大窗口**——但大窗口不等于好结果，
   信息密度比窗口大小更重要
