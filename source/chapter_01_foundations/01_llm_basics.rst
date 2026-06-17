.. _chapter-01-llm-basics:

===============================
LLM 原理与架构
===============================

大语言模型（Large Language Model, LLM）是 Agent 的"大脑"。不管你用的是什么
Agent 框架，底层模型的推理能力直接决定了 Agent 的上限。理解 LLM 的基本原理，
能帮你做出更好的工程决策

.. admonition:: 从"下一个词预测"到涌现出智能
   :class: story

   LLM 的训练目标听起来简单得令人失望：预测下一个词。给模型看
   "今天的天气真"，让它预测下一个最可能的词是"好"。就是这么简单的任务，
   当模型参数达到数十亿、训练数据达到数万亿 token 时，却涌现出了
   **推理、规划、编码、翻译** 等高阶能力。

   这就像学书法——你只练习怎么写好每一个笔画（预测下一个 token），
   但写着写着你突然就能写出有风格、有感情的整篇文章了。
   "涌现"是 LLM 最迷人也最令人不安的特性：没有人设计推理能力，
   它是从"预测下一个词"这个简单任务中自己长出来的。

这一节不从数学公式讲起——那些在 Attention 小节已经覆盖了。我们聚焦于
**对 Agent 开发者真正有用的 LLM 知识**：自回归生成、上下文窗口、
涌现能力和缩放定律。

为什么从 LLM 原理讲起？
============================

Agent 的能力天花板由底层 LLM 决定。以下是几个最直接的例子：

.. list-table::
   :header-rows: 1

   * - LLM 特性
     - 对 Agent 的影响
     - 工程决策
   * - 自回归生成
     - 每步推理需逐个 token 生成，
       延迟 = token 数 × 前向传播时间
     - 控制输出长度（短格式 vs 完整格式）
   * - 上下文窗口
     - Agent 能"记住"的信息量受限于
       LLM 的单次处理上限
     - 设计记忆系统时需考虑窗口预算
   * - 涌现能力
     - 模型过小（<7B）时，
       Agent 的推理/规划能力较弱
     - 选择底座模型时需平衡成本与能力
   * - 缩放定律
     - 更大的模型不一定更适合 Agent，
       任务适配更重要
     - 不要盲目追求大模型，
       先用小模型验证

自回归生成
================

LLM 的生成方式是**自回归**的：每次生成一个 token，然后把新 token
拼回到输入中，再生成下一个。这个过程听起来简单，但它的两个特性
直接影响 Agent 的工程实现。

.. code-block:: python

   # 自回归生成的本质
   def autoregressive_generate(model, prompt, max_tokens=100):
       """
       LLM 生成的核心循环。

       每步只做一件事：基于所有已生成的 token，预测下一个最可能的 token。
       然后把新 token 拼回去，继续预测下一个。

       这就是为什么 LLM 生成"看似突然停住"——它不知道什么时候该停，
       直到生成 EOS（End-of-Sequence）token。
       """
       tokens = tokenize(prompt)

       for _ in range(max_tokens):
           logits = model(tokens)           # 前向传播：算一次
           next_token = sample(logits[-1])  # 采样：选一个
           tokens.append(next_token)

           if next_token == EOS_TOKEN:      # 结束符
               break

       return detokenize(tokens)

特性一：**生成延迟正比于输出长度**

如果模型生成一个 token 需要 50ms，100 个 token 的输出就需要 5 秒。
这对 Agent 意味着：你的 System Prompt 越长，每次推理的"首 token 延迟"越大；
你的输出越详细，流式等待的时间越久。

特性二：**推理成本正比于输入 + 输出 token 数**

Agent 的每次工具调用，都需要把历史对话重新过一遍。如果你选了按 token
计费的 API，多轮对话的成本会快速累积——不是线性增长，因为每一步的输入
都在变大。

.. code-block:: python

   # Agent 多轮对话的 token 消耗累积
   # 第 1 步：System Prompt (2000) + 用户输入 (50) → 2050
   # 第 2 步：System Prompt (2000) + 用户输入 (50) + 上一步对话 (2100) → 4150
   # 第 3 步：System Prompt (2000) + 用户输入 (50) + 前两步对话 (6250) → 8300
   # ...
   # 第 10 步：约 50,000 tokens — 这就是为什么 Agent 对话不能无限持续
   # 要么截断历史，要么做摘要压缩

上下文窗口
================

上下文窗口是 LLM 单次能处理的 token 上限。它不是一个"被动的限制"——
它直接影响 Agent 的架构设计。

.. mermaid::

   flowchart LR
       subgraph 上下文窗口内容
           System[System Prompt<br>1K-4K tokens] --> History[对话历史<br>2K-50K tokens]
           History --> Context[RAG 上下文<br>1K-15K tokens]
           Context --> Reasoning[推理中间步骤<br>1K-5K tokens]
           Reasoning --> Output[输出预留<br>1K-8K tokens]
       end

模型窗口大小决定了你能在"一张"推理中塞入多少信息。

.. list-table::
   :header-rows: 1

   * - 窗口大小
     - 代表模型
     - Agent 能力
   * - 4K-8K
     - Llama 2, Mistral 7B
     - 只能支持单步问答，复杂任务需要频繁压缩
   * - 16K-32K
     - Llama 3, Mistral Large
     - 支持多步 Agent 循环 + 少量 RAG
   * - 128K-200K
     - GPT-4 Turbo, Claude 3, Llama 3.1
     - 可承载完整 Agent + RAG + 历史
   * - 1M-10M
     - Gemini 1.5 Pro, 前沿研究
     - 可加载整个代码库

但这里有一个非常反直觉的现象：**更大的窗口不等于更好的 Agent 表现。**

.. admonition:: Lost in the Middle
   :class: caution

   Liu et al.（2023）发现，LLM 对输入中间的 token 召回率远低于开头和
   结尾。当相关信息位于输入中间位置时，检索准确率从 ~90%（开头）下降到
   ~50%（中间）。

   这意味着：

   - 窗口大了，但如果 Agent 需要的信息被淹没在大量无关上下文里，反而会表现更差
   - 关键信息放两端，不重要信息放中间
   - 不要为了"充分利用大窗口"而塞入无关内容

涌现能力
================

涌现能力（Emergent Abilities）是 LLM 领域最有趣的现象之一。当模型规模
超过某个阈值后，一些能力会"突然出现"——不是慢慢变好，而是从无到有。

.. code-block:: python

   # 涌现能力的直观理解
   model_sizes = {
       "1B":   {"reasoning": "差",   "translation": "中",   "coding": "差"},
       "7B":   {"reasoning": "中",   "translation": "好",   "coding": "中"},
       "13B":  {"reasoning": "中",   "translation": "好",   "coding": "中"},
       "70B":  {"reasoning": "好",   "translation": "好",   "coding": "好"},
       ">100B": {"reasoning": "好",  "translation": "好",   "coding": "好"},
   }

   # 关键规律：
   # 1. 规模在 7B 以下时，复杂推理几乎不可能
   # 2. 7B-13B 是"够用"的边界——可以做简单的 Agent 任务
   # 3. 70B 以上才能稳定处理多步推理
   # 4. 涌现不是渐进的，当你跨过阈值时能力"突然出现"

对 Agent 开发者的实践意义：
- **小模型（<7B）**：适合简单指令跟随、单一工具调用
- **中等模型（7B-13B）**：适合多步 Agent 循环，但可能需要更清晰的提示词
- **大模型（>70B）**：适合复杂推理、多步规划、模糊指令

缩放定律
================

Kaplan et al.（2020）发现模型性能与三个因素呈幂律关系：参数量、数据量和
计算量。但 Chinchilla（Hoffmann et al., 2022）的重要修正是：**不是参数
越多越好，模型和数据应该等比例缩放**。

这对 Agent 开发者意味着：

.. code-block:: text

.. code-block:: python

   # 为 Agent 任务选择模型的决策流程
   def select_model_for_agent(task_complexity: str, budget: float) -> str:
       """
       根据任务复杂度和预算选择模型。
       """
       if task_complexity == "simple" and budget < 0.01:
           return "qwen-2.5-7b"       # 简单任务，小模型
       elif task_complexity == "simple":
           return "gpt-4o-mini"        # 简单任务，平衡
       elif task_complexity == "medium":
           return "claude-3-haiku"     # 中等任务，成本可控
       elif task_complexity == "complex":
           return "gpt-4o"             # 复杂任务，强模型
       else:
           return "claude-opus-4"      # 最复杂任务，最强模型

Transformer 架构速览
=====================

现代 LLM 几乎都使用 Decoder-only Transformer。它的核心结构是：

.. mermaid::

   flowchart LR
       Input[输入文本] --> Tokenize[Token 化]
       Tokenize --> Embed[Token Embedding]
       Embed --> Pos[位置编码]
       Pos --> Block[Transformer Block ×N]
       Block --> LM[LM Head]
       LM --> Output[输出 Token]

       subgraph Block [每个 Transformer Block]
           direction TB
           MHA[多头自注意力<br>Multi-Head Self-Attention] --> FFN[前馈网络<br>Feed-Forward]
           FFN --> Norm[层归一化<br>Layer Norm]
       end

Decoder-only 与 Encoder-Decoder（如 T5）的核心区别：Decoder-only 是**因果的**
（当前 token 只能看之前的内容），而 Encoder-Decoder 的编码器可以双向看。
对于 Agent 场景，Decoder-only 更合适——你不需要看到"未来的工具返回结果"。

训练与推理的差异
====================

一个经常被忽略的知识点：训练时和推理时的行为模式完全不同。

.. list-table::
   :header-rows: 1

   * - 维度
     - 训练阶段
     - 推理阶段（Agent 使用）
   * - 输入方式
     - 一次输入完整序列
     - 逐步生成
   * - KV Cache
     - 不需要
     - 必须使用，否则极慢
   * - 批处理
     - 批次内并行
     - 逐 token 串行
   * - 内存需求
     - 高（需要存梯度）
     - 中（只需要存 KV Cache）
   * - 精度
     - FP16/FP32（需要梯度精度）
     - 可量化到 INT4（推理不需要高精度）
   * - Token 消费
     - 固定（每个样本消费固定 token 数）
     - 动态（输出长度不确定）

这个差异在 Agent 部署中很实用：训练你需要高端 GPU，推理你可以在消费级
硬件上跑——尤其是量化后的模型。

对 Agent 开发者的三个核心启示
================================

1. **延迟的优化重点是输出长度，不是模型速度**

   .. code-block:: python

       # 坏示例：让 Agent 输出完整推理过程
       agent.run("分析这个需求...")  # 输出 2000 tokens → 20 秒

       # 好示例：让 Agent 只输出关键结论
       agent.run("分析这个需求，只输出结论和证据")  # 输出 200 tokens → 2 秒

2. **上下文窗口决定了你的记忆策略**

   128K 窗口不意味着你能塞 128K 信息。有效内容受限于 Lost in the Middle。
   对抗策略：把最重要信息放两端，中间放次要信息。

3. **模型选型应该从评估开始，不是直觉**

   用评估集（Eval Set）测试不同模型在 Agent 任务上的表现，选择性价比
   最高的那个。不要让"直觉"帮你选模型——你可能会高估小模型或低估大模型。
