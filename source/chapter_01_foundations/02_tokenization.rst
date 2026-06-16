.. _chapter-01-tokenization:

===============================
Tokenization 与嵌入
===============================

Tokenization 是将自然语言切分为模型可处理的基本单元（Token）的过程。
我见过的 Agent 开发者中，十个有九个低估了 Tokenization 的影响。
他们以为这只是一个"预处理步骤"，但实际上 Token 的选择会影响你的
Agent 能处理多长的文本、成本和速度，甚至回答质量。

为什么？因为 LLM 的上下文窗口是按 token 计数的，而不是按字数。
不同的 Tokenizer 对同一段中文的切分方式不同，导致"看起来同长度的文本"
实际消耗的 token 可以差 2-3 倍。这意味着如果你选错了 Tokenizer，
你的 Agent 的"有效上下文"可能只有别人的一半。

Token 的概念
================

Token 是 LLM 处理文本的基本单元。它**不是**单词，也不是字，而是一个
经过统计优化的"中间单位"。

.. code-block:: python

   # 同一个句子，不同语言的 token 消耗差异
   中文: "我是一个学生"
   英文: "I am a student"

   # GPT-4 Tokenizer 的处理结果：
   # 中文 "我是一个学生" → ["我", "是", "一个", "学生"] = 4 tokens
   # 英文 "I am a student" → ["I", " am", " a", " student"] = 4 tokens

   # 但如果是长文本：
   # 中文 1000 字 → ≈ 1500 tokens（每个汉字约 1.5 token）
   # 英文 1000 词 → ≈ 750 tokens（每个英文词约 0.75 token）

这个差异有实际后果：同样 128K 的上下文窗口，如果给一个全中文的 Agent
提示词，有效内容只有大约 85K 的"信息量"；如果提示词是中英文混合的，
这个比例会更复杂。

常见分词算法
================

1. BPE（Byte Pair Encoding）
------------------------------

GPT 系列使用的 Tokenizer。从单个字符开始，逐对合并出现频率最高的字符对，
直到达到预定的词汇表大小。

.. code-block:: python

   def bpe_train(corpus, vocab_size=50000):
       """
       BPE 训练过程（简化版）：
       1. 把所有文本拆成单个字符
       2. 统计所有相邻字符对的频率
       3. 把最频繁的字符对合并成一个新 token
       4. 重复直到词汇表达到目标大小
       """
       # 初始化：把所有词拆成字符序列
       words = list(corpus)
       vocab = set(words)

       while len(vocab) < vocab_size:
           # 统计相邻字符对的频率
           pairs = {}
           for i in range(len(words) - 1):
               pair = (words[i], words[i+1])
               pairs[pair] = pairs.get(pair, 0) + 1

           # 找出最频繁的一对
           most_freq = max(pairs, key=pairs.get)
           # 合并这对
           new_token = most_freq[0] + most_freq[1]
           vocab.add(new_token)

           # 替换原序列中的这对
           new_words = []
           for w in words:
               new_words.append(w)
               if w == most_freq[0]:
                   continue  # 简化处理
           words = new_words

       return vocab

   # 实际效果
   # 初始："h", "e", "l", "l", "o"
   # 第一轮合并 "l" + "l" → "ll"
   # → "h", "e", "ll", "o"
   # 第二轮合并 "h", "e" → "he"
   # → "he", "ll", "o"
   # 最终："he", "ll", "o" → 3 tokens（原来是 5 个）

2. WordPiece
------------------------------

BERT 使用的 Tokenizer。和 BPE 类似，但合并依据不是频率，而是**概率增益**。

3. SentencePiece
------------------------------

Llama 系列使用的 Tokenizer。和前两者的核心区别：**不依赖空格分割**。
这对于中文、日文等不使用空格的语言特别重要。

.. code-block:: python

   # SentencePiece 的优势
   # BPE 需要先按空格分词： "我是一个学生" 先变成 ["我", "是", "一个", "学生"]
   # SentencePiece 直接处理原始文本，不依赖空格
   # 所以 SentencePiece 对中文更友好，token 效率更高

Token 对 Agent 成本的影响
=============================

.. code-block:: python

   def estimate_agent_cost(agent_run, price_per_1k_input=0.01, price_per_1k_output=0.03):
       """
       估算一次 Agent 执行的成本。
       包含多步推理中的输入和输出 token。
       """
       total_input_tokens = 0
       total_output_tokens = 0

       for step in agent_run.steps:
           total_input_tokens += step.input_tokens  # 每步的 System Prompt + 历史
           total_output_tokens += step.output_tokens

       cost = (total_input_tokens / 1000 * price_per_1k_input +
               total_output_tokens / 1000 * price_per_1k_output)
       return cost

   # 实际例子：一个 5 步的 Agent 执行
   # 每步输入：~4000 tokens（System Prompt + 对话历史）
   # 每步输出：~200 tokens（模型回复）
   # 总成本 = (5 * 4000 / 1000 * 0.01) + (5 * 200 / 1000 * 0.03)
   #        = 0.20 + 0.03 = $0.23
   #
   # 如果每天 10000 次调用：$2300/天
   # 这就是为什么 token 优化在 Agent 场景中如此重要

token 优化的几个实用技巧：

1. **精简 System Prompt**：每条规则都占用 token。如果一个规则的效果不确定，
   先不加
2. **工具描述剪裁**：工具描述的 name 和 description 都会占用 token，
   description 控制在 1-2 句话
3. **历史对话摘要**：不要保留完整对话历史，用 LLM 生成摘要
4. **工具返回裁剪**：工具返回结果只保留关键字段

Token 对上下文窗口的影响
============================

Agent 的核心约束是上下文窗口有限。但"上下文窗口大小"不是固定的——

.. code-block:: python

   # 不同 Tokenizer 下，同样 1000 个汉字的 token 消耗
   tokenizers = {
       "GPT-4 (BPE)": 1500,  # 中文约 1.5 token/字
       "Llama 3 (SentencePiece)": 1200,  # 对中文更高效
       "Claude (自家)": 1300,
   }

   # 如果上下文窗口是 128K：
   # 用 GPT-4：能容纳约 85K 中文
   # 用 Llama 3：能容纳约 106K 中文
   # 差距近 20K token——这意味着你的 Agent 能多记一轮对话

这就是为什么有些 Agent 在 Llama 3 上表现"更好"——不是模型能力更强，
而是 Tokenizer 效率更高，同一个上下文窗口里能容纳更多信息。

词汇表大小的权衡
====================

.. list-table::
   :header-rows: 1

   * - 词汇表大小
     - 优点
     - 缺点
     - 代表模型
   * - 32K
     - 模型体积小，推理快
     - 需要多个 token 表示罕见词
     - 早期 GPT 模型
   * - 100K
     - 常见词直接用 1 token
     - 模型嵌入层更大
     - GPT-4 (100K)、Claude (100K)
   * - 128K+
     - 罕见领域词也能编码
     - 更大的嵌入矩阵
     - Llama 3 (128K)

大的词汇表意味着**更少的 token 表示同样信息** ，但代价是嵌入矩阵更大
（词汇表大小 × 嵌入维度）。这是一个存储效率和推理效率之间的权衡。

Tokenization 对 Agent 的隐藏影响
====================================

.. admonition:: 中文 Agent 多花 50% 的 token
   :class: caution

   如果你的 Agent 面向中文用户，System Prompt 是中文的，工具描述的
   注释也是中文的，那么同样的功能比英文 Agent 多花约 50% 的 token。
   在设计 System Prompt 时要有这个意识——精简中文提示词比精简英文更关键。

.. admonition:: 数字和代码的 Token 效率
   :class: tip

   "1234567890" 可能被切为多个 token（因为 Tokenizer 不常看到这个组合）。
   但 "2024" 可能就是一个 token（因为常见）。所以如果你的 Agent 需要
   处理大量数字，尽量用常见的表达方式。

.. code-block:: python

   # Token 检查工具
   def check_tokens(text, tokenizer):
       """检查任意文本的 token 消耗"""
       tokens = tokenizer.encode(text)
       print(f"文本：{text[:50]}...")
       print(f"Token 数量：{len(tokens)}")
       print(f"Token 列表：{tokens[:10]}...")
       return len(tokens)

   # 在 Agent 开发中应该经常做这个检查
   # 你会发现哪些提示词设计浪费了大量 token
   # 然后针对性地优化它们
