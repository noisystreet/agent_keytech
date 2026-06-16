.. _chapter-05-retrieval:

===============================
检索策略
===============================

检索是 RAG 的第一步，决定了 Agent 能看到哪些信息。"检索质量直接影响 Agent
回答的准确性和相关性"这句话，Agent 开发者应该每天念三遍。它不是一个
可以用"调个向量数据库"就解决的问题。

很多人都觉得向量检索是"银弹"——把文档转成向量，查相似度，完事。
但如果你在生产环境里调试过 Agent 回答为什么不对，有相当大概率会发现：
不是 Agent 推理错了，是它根本没看到正确的信息。检索出来的文档
跟问题毫不相干。

这就是检索策略要解决的核心问题：**如何在海量文档中找到 Agent 真正需要的那几段。**

三种检索范式
================

1. 稀疏检索（Sparse Retrieval）
------------------------------

传统的关键词匹配方法。BM25 是其中最经典的实现。

.. code-block:: python

   class BM25Retriever:
       """
       BM25：基于词频和逆文档频率的排序函数。
       核心思路：在一个文档中频繁出现、但在整个语料中少见的词更重要。
       """
       def __init__(self, documents):
           self.documents = documents
           self._build_index()

       def _build_index(self):
           # 对每个文档统计词频
           self.doc_freq = {}  # 每个词出现在多少个文档中
           self.term_freq = [] # 每个文档中每个词的频率
           self.avg_doc_len = 0

           total_len = 0
           for doc in self.documents:
               terms = self._tokenize(doc)
               total_len += len(terms)

               # 统计当前文档的词频
               tf = {}
               for t in set(terms):
                   tf[t] = terms.count(t)
                   self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
               self.term_freq.append(tf)

           self.avg_doc_len = total_len / len(self.documents)
           self.N = len(self.documents)  # 文档总数

       def search(self, query: str, k: int = 10) -> list:
           query_terms = self._tokenize(query)
           scores = []

           for i, doc in enumerate(self.documents):
               score = 0
               for term in query_terms:
                   if term not in self.term_freq[i]:
                       continue
                   tf = self.term_freq[i][term]
                   df = self.doc_freq.get(term, 0)
                   doc_len = sum(self.term_freq[i].values())

                   # BM25 核心公式
                   k1, b = 1.5, 0.75
                   idf = log((self.N - df + 0.5) / (df + 0.5) + 1)
                   tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / self.avg_doc_len))
                   score += idf * tf_norm

               scores.append(score)

           # 返回 top-k
           top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
           return [(self.documents[i], scores[i]) for i in top_indices]

BM25 的优势是**精确、可解释、零成本**（不需要模型）。当你搜索一个具体的
订单号 "ORD-2024-001" 时，BM25 能精确命中，而向量检索可能因为"语义相似"
返回一堆无关的"ORD"相关文档。

但 BM25 的局限也很明显：如果用户的 query 和文档用词不同（比如 query 说"车"，
文档写"汽车"），BM25 就无法匹配。这就是语义鸿沟问题。

2. 稠密检索（Dense Retrieval）
------------------------------

.. code-block:: python

   class DenseRetriever:
       """
       稠密检索：用嵌入模型将 query 和文档映射到同一向量空间，
       通过向量相似度检索。
       """
       def __init__(self, embedder, documents):
           self.embedder = embedder
           self.documents = documents
           self.doc_vectors = [self.embedder.embed(doc) for doc in documents]

       def search(self, query: str, k: int = 10) -> list:
           q_vec = self.embedder.embed(query)

           # 计算与所有文档的余弦相似度
           scores = []
           for doc_vec in self.doc_vectors:
               sim = cosine_similarity(q_vec, doc_vec)
               scores.append(sim)

           top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
           return [(self.documents[i], scores[i]) for i in top_indices]

稠密检索解决了语义鸿沟问题——"车"和"汽车"在向量空间中的距离很近。
但它引入了新问题：**冷启动**。你需要在你的数据上测试嵌入模型是否有效，
如果效果不好，需要换模型甚至微调。

3. 混合检索（Hybrid Retrieval）
------------------------------

混合检索结合稀疏和稠密两种方式，取长补短。

.. code-block:: python

   class HybridRetriever:
       """
       混合检索：BM25 + 向量检索，加权融合。
       alpha 控制两者的权重：alpha=0 时纯向量，alpha=1 时纯 BM25。
       """
       def __init__(self, bm25_index, vector_db, alpha=0.5):
           self.bm25 = bm25_index
           self.vector_db = vector_db
           self.alpha = alpha

       def search(self, query: str, k: int = 10) -> list:
           # 两种检索各自打分
           bm25_results = dict(self.bm25.search(query, k=k * 3))
           vector_results = dict(self.vector_db.similarity_search(query, k=k * 3))

           # 归一化融合
           all_docs = set(list(bm25_results.keys()) + list(vector_results.keys()))
           fused = {}
           for doc in all_docs:
               bm25_score = self._normalize(bm25_results.get(doc, 0), bm25_results)
               vec_score = self._normalize(vector_results.get(doc, 0), vector_results)
               fused[doc] = self.alpha * bm25_score + (1 - self.alpha) * vec_score

           return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

       def _normalize(self, score, all_scores):
           """Min-Max 归一化到 [0, 1]"""
           if not all_scores:
               return 0
           values = list(all_scores.values())
           min_v, max_v = min(values), max(values)
           if max_v == min_v:
               return 0.5
           return (score - min_v) / (max_v - min_v)

这里有个实用的经验值：对于大多数 Agent 场景，alpha 取 0.3-0.5 效果最好。
也就是说，向量检索的权重略高于或等于 BM25。但如果你的场景涉及大量
精确匹配（代码搜索、订单查询），alpha 应该调到 0.7 以上。

索引结构：暴力搜索 vs ANN
============================

当文档数量超过百万级时，暴力搜索（遍历所有文档）就太慢了。这时需要
近似最近邻搜索（ANN）。

.. code-block:: python

   # 暴力搜索 vs ANN 的权衡
   comparison = {
       "exact_search": {
           "latency": "O(n·d)",      # n=文档数, d=维度
           "accuracy": "100%",
           "best_for": "数据量 < 10 万"
       },
       "hnsw": {
           "latency": "O(log n)",     # 层级图遍历
           "accuracy": "95-99%",      # 近似，但足够准确
           "best_for": "数据量 10 万以上"
       },
       "ivf": {
           "latency": "O(sqrt(n))",   # 倒排文件
           "accuracy": "90-95%",
           "best_for": "需要快速建索引"
       },
   }

HNSW（Hierarchical Navigable Small World）是目前最主流的 ANN 算法。
它的工作原理像一个多层次的导航地图：顶层是"高速公路"（稀疏连接），
底层是"小巷"（密集连接）。

.. code-block:: python

   # HNSW 的建索引参数
   hnsw_params = {
       "M": 16,        # 每层最大连接数——越大精度越高但索引越大
       "ef_construction": 200,  # 建索引时的搜索范围——越大质量越高
       "ef_search": 50,         # 检索时的搜索范围——越大越准但越慢
   }

   # ef_search 的调参经验
   # ef_search = 50：  召回率 ≈ 90%
   # ef_search = 100： 召回率 ≈ 95%
   # ef_search = 200： 召回率 ≈ 98%
   # ef_search 每增加一倍，延迟增加约 30% 但召回率提升递减

查询重写（Query Rewriting）
============================

用户的问题往往不适合直接检索。比如用户问"它是什么时候成立的？"，
"它"指代什么？如果没有上下文，检索结果肯定不对。

.. code-block:: python

   class QueryRewriter:
       """
       将用户原始问题改写为更适合检索的形式。
       这是 Agent RAG 中最容易被忽略但效果提升最大的环节。
       """
       def rewrite(self, original_query: str, conversation_history: list = None) -> str:
           prompt = f"""
           用户的问题是：{original_query}

           对话历史：{conversation_history}

           请将用户问题改写为一个独立的、完整的检索查询，以便在知识库中搜索。
           要求：
           - 补全代词（"它" → "具体实体"）
           - 去掉口语化的语气词
           - 保留问题的核心意图

           改写后的查询：
           """
           return llm.generate(prompt, temperature=0.0)

   # 改写前后的对比
   # 原始："它支持多少种语言？"
   # 改写："OpenAI GPT-4 API 支持的语言数量"
   #
   # 原始："帮我找一下那篇关于 attention 的论文"
   # 改写："Attention Is All You Need 论文内容"

查询路由（Query Routing）
============================

不同类型的 query 应该路由到不同的检索策略。

.. code-block:: python

   class QueryRouter:
       """
       根据查询类型路由到不同的检索策略。
       """
       def __init__(self, retrievers: dict):
           self.retrievers = retrievers  # {"keyword": bm25, "semantic": dense, "hybrid": hybrid}
           self.routes = {
               "code": "keyword",      # 代码搜索→BM25
               "general": "hybrid",    # 通用搜索→混合
               "research": "semantic", # 学术搜索→向量
               "factual": "hybrid",    # 事实查询→混合
           }

       def route(self, query: str) -> list:
           query_type = self._classify(query)
           retriever = self.retrievers[self.routes[query_type]]
           return retriever.search(query)

       def _classify(self, query: str) -> str:
           prompt = f"分类查询类型（code/general/research/factual）：{query}"
           return llm.generate(prompt, temperature=0.0).strip().lower()

.. admonition:: 检索策略选型建议
   :class: tip

   1. **MVP 阶段**：BM25 就够了，零成本、可解释
   2. **遇到语义问题**：加向量检索，做混合搜索
   3. **检索质量遇到瓶颈**：加查询重写（效果最明显）
   4. **数据量 > 100 万**：上 ANN 索引（HNSW 优先）
   5. **多场景混合**：加查询路由，不同类型走不同检索
