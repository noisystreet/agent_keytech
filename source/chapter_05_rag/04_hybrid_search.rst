.. _chapter-05-hybrid-search:

===============================
混合搜索
===============================

混合搜索结合了关键词搜索（BM25）和向量搜索的优势，是 RAG 系统达到生产级
检索质量的关键。如果说 BM25 是"精确但不聪明"，向量搜索是"聪明但不精确"，
那混合搜索就是"又聪明又精确"。

但混合搜索也不是"装上去就变好"。它引入了一个关键问题：**如何融合两种
完全不同的分数？** BM25 的分数和向量相似度分数的分布特征完全不同，
直接相加会出问题。

分数归一化：打通 BM25 和向量
===============================

.. code-block:: python

   from rank_bm25 import BM25Okapi
   from sentence_transformers import SentenceTransformer

   class HybridSearchEngine:
       def __init__(self, documents):
           self.documents = documents
           self.bm25 = BM25Okapi(documents)
           self.embedder = SentenceTransformer('BAAI/bge-large-zh')
           self.vectors = self.embedder.encode(documents)

       def search(self, query, k=10, alpha=0.3):
           # BM25 得分
           bm25_scores = self.bm25.get_scores(query.split())
           # 向量得分
           q_vec = self.embedder.encode([query])
           vec_scores = cosine_similarity(q_vec, self.vectors)[0]
           # 加权融合
           combined = alpha * normalize(bm25_scores) \
                      + (1 - alpha) * normalize(vec_scores)
           top_k = np.argsort(combined)[-k:][::-1]
           return [(self.documents[i], combined[i]) for i in top_k]

这里 normalize 函数是关键。如果不做归一化，BM25 的分数范围可能是 [0, 20]，
而向量相似度是 [-1, 1]，不加归一化的话混合结果基本由 BM25 主导。

三种常见融合策略
======================

1. 加权平均融合
------------------------------

最直接的方法：对两种分数做 min-max 归一化或 z-score 归一化，然后加权平均。

.. code-block:: python

   def weighted_fusion(bm25_scores, vec_scores, alpha=0.5):
       """
       加权平均融合。
       alpha = 0.3 → 向量主导（推荐）
       alpha = 0.7 → BM25 主导（适合精确匹配）
       """
       bm25_norm = minmax_normalize(bm25_scores)
       vec_norm = minmax_normalize(vec_scores)
       return alpha * bm25_norm + (1 - alpha) * vec_norm

2. RRF（Reciprocal Rank Fusion）
------------------------------

RRF 不依赖分数本身，而是基于**排序位置**计算融合得分。

.. code-block:: python

   def rrf_fusion(bm25_ranked, vec_ranked, k=60):
       """
       RRF：基于排序位置的融合方法。
       不依赖原始分数，对分数分布不敏感，更鲁棒。

       score(d) = sum(1 / (k + rank_i(d)))
       k 是平滑参数，k=60 是经验值。
       """
       scores = {}
       for i, doc in enumerate(bm25_ranked):
           scores[doc] = 1 / (k + i + 1)
       for i, doc in enumerate(vec_ranked):
           scores[doc] = scores.get(doc, 0) + 1 / (k + i + 1)
       return sorted(scores.items(), key=lambda x: x[1], reverse=True)

RRF 的优势在于它**不关心 BM25 和向量相似度的分数分布是否一致**。
只要两种检索各自给出了排序，RRF 就能融合。这让它在实践中比加权平均更稳定。

.. admonition:: RRF vs 加权平均
   :class: tip

   加权平均适合"两种检索的分数都是可靠的"；RRF 适合"不清楚分数分布"。
   建议 MVP 阶段用 RRF（参数少、不需调试），线上稳定后尝试加权平均。

3. Score Fusion with Dynamic Alpha
------------------------------

不固定 alpha，而是根据查询类型动态调整。

.. code-block:: python

   class DynamicHybridSearch:
       """
       动态 alpha 的混合搜索。
       对于精确查询（ID、代码）增加 BM25 权重，
       对于语义查询增加向量权重。
       """
       def __init__(self, bm25, vector_db):
           self.bm25 = bm25
           self.vector_db = vector_db

       def search(self, query: str, k: int = 10) -> list:
           alpha = self._estimate_alpha(query)
           bm25_results = self.bm25.search(query, k=k * 2)
           vec_results = self.vector_db.similarity_search(query, k=k * 2)
           return self._rrf_fusion(bm25_results, vec_results, alpha)

       def _estimate_alpha(self, query: str) -> float:
           """
           估计 alpha。
           - 包含 ID、代码、数字 → alpha 大（BM25 主导）
           - 自然语言 → alpha 小（向量主导）
           """
           if any(c.isdigit() for c in query):
               return 0.7  # 精确匹配
           if any(c in query for c in ["#", "-", "_", "/"]):
               return 0.6  # 代码/路径
           return 0.3  # 自然语言

.. code-block:: python

   # 完整实现：动态混合搜索引擎
   class CompleteHybridEngine:
       def __init__(self, documents):
           self.bm25 = BM25Okapi(documents)
           self.embedder = SentenceTransformer('BAAI/bge-large-zh')
           self.vectors = self.embedder.encode(documents)
           self.documents = documents

       def search(self, query: str, k=10) -> list:
           # BM25 检索
           bm25_scores = self.bm25.get_scores(query.split())

           # 向量检索
           q_vec = self.embedder.encode([query])
           vec_scores = cosine_similarity(q_vec, self.vectors)[0]

           # RRF 融合
           bm25_ranked = sorted(
               range(len(bm25_scores)),
               key=lambda i: bm25_scores[i], reverse=True
           )
           vec_ranked = sorted(
               range(len(vec_scores)),
               key=lambda i: vec_scores[i], reverse=True
           )

           rrf_scores = defaultdict(float)
           for i, idx in enumerate(bm25_ranked):
               rrf_scores[idx] += 1.0 / (60 + i)
           for i, idx in enumerate(vec_ranked):
               rrf_scores[idx] += 1.0 / (60 + i)

           top_k = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
           return [(self.documents[idx], score) for idx, score in top_k]

混合搜索的常见误区
======================

.. admonition:: 误区 1：向量检索比 BM25 "更好"
   :class: caution

   它们解决的是不同的匹配问题。向量检索擅长"语义相似"，BM25 擅长"精确匹配"。
   Agent 场景中两者都需要。比如查询"删除 user_id=100 的记录"——
   精确保留的 BM25 能找到正确的文档，语义检索可能返回一堆关于"删除用户"的
   无关内容。

.. admonition:: 误区 2：alpha 是固定的
   :class: tip

   alpha 应该在 0.2-0.8 之间根据查询类型动态调整。Agent 框架可以在
   调用检索前先分类查询，选择不同的混合策略。这种动态调整可以让
   检索质量提升 10-20%。
