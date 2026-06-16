.. _chapter-05-chunking:

===============================
文档分块
===============================

文档分块（Chunking）是将长文档切分为可检索的小块。你可能会觉得这只是一个
"分段"的技术活，但分块策略的好坏可以直接让你的 RAG 检索质量差 20-30%。
分块太粗，一个块里塞了太多主题，检索时精度下降；分块太细，跨块的上下文
被切断，Agent 看不到完整信息。

分块的核心矛盾
================

Agent 的上下文窗口是有限的。你不能把整本《三体》塞进一次推理——Model Context Protocol
窗口再大也装不下。但你也不能把一句话切成一个块——Agent 需要看到足够的上下文才能
理解这句"不要回答！不要回答！！不要回答！！！"是在什么情景下说的。

所以分块的本质是一个**粒度权衡**：

.. list-table::
   :header-rows: 1

   * - 粒度
     - 优势
     - 劣势
     - 适合场景
   * - 粗粒度（500-1000 tokens）
     - 上下文完整，Agent 容易理解
     - 块内信息混杂，检索精度低
     - 叙事性文档、长文本
   * - 中粒度（200-500 tokens）
     - 检索精度和上下文完整性的平衡
     - 可能切断语义边界
     - 大多数场景
   * - 细粒度（50-200 tokens）
     - 检索精度高，命中即相关
     - 上下文碎片化，需要多块拼接
     - FAQ、代码片段

五种分块策略
================

1. 固定大小分块
------------------------------

最简单直接：按 token 数均分。

.. code-block:: python

   class FixedSizeChunker:
       def __init__(self, chunk_size=500, overlap=50):
           self.chunk_size = chunk_size
           self.overlap = overlap

       def chunk(self, text: str) -> list:
           tokens = self._tokenize(text)
           chunks = []
           start = 0
           while start < len(tokens):
               end = start + self.chunk_size
               chunk_tokens = tokens[start:end]
               chunks.append(self._detokenize(chunk_tokens))
               start = end - self.overlap  # 重叠部分防止切断关键信息
           return chunks

固定分块的优点是**实现简单**（20 行代码搞定）。缺点是**不尊重语义边界**——
一个段落可能被从中间切断，导致检索时看到的信息不完整。

2. 语义分块
------------------------------

按自然语义边界（段落、标题、章节）切分。

.. code-block:: python

   class SemanticChunker:
       def __init__(self, max_chunk_size=500):
           self.max_chunk_size = max_chunk_size

       def chunk(self, text: str) -> list:
           # 1. 按标题分割
           sections = self._split_by_headings(text)
           chunks = []

           for section in sections:
               # 2. 按段落分割
               paragraphs = section.split("\n\n")
               current_chunk = ""

               for para in paragraphs:
                   if len(self._tokenize(current_chunk + para)) > self.max_chunk_size:
                       if current_chunk:
                           chunks.append(current_chunk)
                       current_chunk = para
                   else:
                       current_chunk += "\n\n" + para

               if current_chunk:
                   chunks.append(current_chunk)

           return chunks

       def _split_by_headings(self, text) -> list:
           """按 Markdown 标题或章节标记分割"""
           import re
           sections = re.split(r'\n#{1,3}\s', text)
           return [s.strip() for s in sections if s.strip()]

3. 递归分块
------------------------------

从语义边界（标题 → 段落 → 句子）逐级尝试，找到不超过最大大小的最优块。

.. code-block:: python

   class RecursiveChunker:
       def __init__(self, chunk_size=500, overlap=50):
           self.chunk_size = chunk_size
           self.overlap = overlap
           self.separators = ["\n\n", "\n", "。", "；", "，"]

       def chunk(self, text: str) -> list:
           chunks = []
           self._recursive_split(text, self.separators, chunks)
           return chunks

       def _recursive_split(self, text, separators, result):
           if len(self._tokenize(text)) <= self.chunk_size:
               result.append(text)
               return

           if not separators:
               # 已经没有分隔符可用了，强制按 token 切分
               result.extend(self._force_split(text))
               return

           sep = separators[0]
           segments = text.split(sep)
           current = ""

           for seg in segments:
               if len(self._tokenize(current + sep + seg)) > self.chunk_size:
                   if current:
                       result.append(current)
                   # 如果单个片段仍然太大，用更细粒度的分隔符继续切
                   if len(self._tokenize(seg)) > self.chunk_size:
                       self._recursive_split(seg, separators[1:], result)
                   else:
                       current = seg
               else:
                   current = seg

           if current:
               result.append(current)

4. Agent 感知分块
------------------------------

这是专门为 Agent 场景设计的策略。Agent 经常需要调用工具，不同工具的
输出格式不同，需要不同的分块策略。

.. code-block:: python

   class AgentAwareChunker:
       """
       根据内容类型选择分块策略。
       代码、表格、自然语言用不同的分割方式。
       """
       def __init__(self):
           self.strategies = {
               "code": self._chunk_code,
               "prose": self._chunk_prose,
               "table": self._chunk_table,
               "json": self._chunk_json,
           }

       def chunk(self, text: str, content_type: str = "prose") -> list:
           strategy = self.strategies.get(content_type, self._chunk_prose)
           return strategy(text)

       def _chunk_code(self, code: str) -> list:
           """代码按函数/类定义切分"""
           import re
           # 按函数定义分割
           functions = re.split(r'\n(def |class |async def )', code)
           return [f.strip() for f in functions if f.strip()]

       def _chunk_prose(self, text: str) -> list:
           return SemanticChunker().chunk(text)

       def _chunk_table(self, table: str) -> list:
           """表格按行分块，保留表头"""
           lines = table.strip().split("\n")
           header = lines[0]
           chunks = []
           for i in range(1, len(lines), 20):
               chunk = "\n".join([header] + lines[i:i+20])
               chunks.append(chunk)
           return chunks

5. 层级分块
------------------------------

为大文档同时维护粗粒度和细粒度的表示。

.. code-block:: python

   class HierarchicalChunker:
       """
       层级分块：粗块用于检索筛选，细块用于精细阅读。
       Agent 先定位到粗块，再在粗块内搜索细块。
       """
       def chunk(self, text: str) -> dict:
           # 大块：章节级别
           coarse = self._split_by_section(text)

           # 小块：段落级别
           fine = {}
           for section in coarse:
               fine[section["title"]] = self._split_by_paragraph(section["content"])

           return {"coarse": coarse, "fine": fine}

       def retrieve(self, query: str, index) -> list:
           # 先检索粗块
           coarse_results = index.search_coarse(query, k=3)
           # 在命中的粗块内检索细块
           return index.search_fine(query, coarse_results)

分块策略的评估
================

哪个分块策略最好？不是理论推导出来的，而是测出来的。

.. code-block:: python

   def evaluate_chunker(chunker, documents, test_queries):
       """
       评估分块策略的检索质量。
       核心指标：每个 query 是否能检索到包含答案的块。
       """
       chunks = chunker.chunk(documents)
       hits = 0

       for query, answer in test_queries:
           # 检索最相关的块
           relevant_chunk = retrieve(query, chunks)
           # 检查答案是否在检索到的块中
           if answer in relevant_chunk:
               hits += 1

       return hits / len(test_queries)

   # 在不同数据上测试不同的分块策略
   results = {
       "fixed_size": evaluate_chunker(FixedSizeChunker(), docs, tests),
       "semantic": evaluate_chunker(SemanticChunker(), docs, tests),
       "recursive": evaluate_chunker(RecursiveChunker(), docs, tests),
       "agent_aware": evaluate_chunker(AgentAwareChunker(), docs, tests),
   }

在生产环境中，**没有通用的最佳分块策略**，只有最适合你数据的分块策略。
测试是关键。
