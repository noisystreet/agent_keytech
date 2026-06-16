.. _chapter-01-tokenization:

===============================
Tokenization 与嵌入
===============================

Tokenization 是将自然语言文本切分为模型可处理的基本单元（Token）的过程。
这是 LLM 处理文本的第一步，也是影响 Agent 性能的隐藏因素。

.. admonition:: Token 不是单词
   :class: funfact

   一个中文汉字可能被切分为 1-2 个 token，而 "Agent" 这个英文单词通常为 1 个 token。
   GPT-4 的词汇表大小约为 100k token，覆盖了主流语言和代码。这意味着 Agent 在
   处理中英文混杂的输入时，**实际的上下文容量会因语言不同而变化**——全中文时
   能容纳的内容比全英文少约 50%。

常见分词算法
================

- **BPE（Byte Pair Encoding）**：GPT 系列使用，从字符开始逐对合并高频 token
- **WordPiece**：BERT 使用，基于概率而非频率合并
- **SentencePiece**：Llama 系列使用，不依赖空格分割，原生支持中文

影响 Agent 的 Token 细节
=============================

.. code-block:: python

   # Token 计数直接影响成本
   def estimate_cost(text, price_per_1k=0.01):
       tokens = len(tokenizer.encode(text))
       cost = tokens / 1000 * price_per_1k
       return tokens, cost

   # Agent 系统提示词通常占据大量 token
   system_prompt = """
   你是 AI 助手。你有以下工具可用：
   - search(query): 搜索互联网
   - calculate(expr): 计算数学表达式
   - write_file(path, content): 写入文件
   每次回复必须包含 reasoning 和 action。
   """
   tokens, cost = estimate_cost(system_prompt)
   print(f"系统提示占用 {tokens} token，每次调用成本 ${cost:.4f}")
