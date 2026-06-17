.. _chapter-05-hybrid-search:

===============================
混合搜索
===============================

混合搜索结合了关键词搜索（BM25）和向量搜索的优势，是 RAG 系统达到生产级
检索质量的关键。如果说 BM25 是"精确但不聪明"，向量搜索是"聪明但不精确"，
那混合搜索就是"又聪明又精确"。

但混合搜索也不是"装上去就变好"。它引入了一个关键问题：**如何融合两种
完全不同的分数？** BM25 的分数范围是 [0, 20+] 且分布严重偏斜（少数文档
得分远高于其他），而向量相似度范围是 [-1, 1] 且分布相对均匀。不加处理
直接相加，结果基本由 BM25 主导。

分数归一化：打通 BM25 和向量
===============================

.. code-block:: python

   from rank_bm25 import BM25Okapi
   from sentence_transformers import SentenceTransformer
   import numpy as np

   class HybridSearchEngine:
       def __init__(self, documents):
           self.documents = documents
           self.bm25 = BM25Okapi(documents)
           self.embedder = SentenceTransformer('BAAI/bge-large-zh')
           self.vectors = self.embedder.encode(documents)

       def search(self, query, k=10, alpha=0.3):
           bm25_scores = self.bm25.get_scores(query.split())
           q_vec = self.embedder.encode([query])
           vec_scores = cosine_similarity(q_vec, self.vectors)[0]

           # 归一化是关键！否则 BM25 主导结果
           combined = alpha * self._normalize(bm25_scores) \
                      + (1 - alpha) * self._normalize(vec_scores)

           top_k = np.argsort(combined)[-k:][::-1]
           return [(self.documents[i], combined[i]) for i in top_k]

       def _normalize(self, scores):
           """Min-Max 归一化到 [0, 1]"""
           min_v, max_v = scores.min(), scores.max()
           if max_v == min_v:
               return np.zeros_like(scores)
           return (scores - min_v) / (max_v - min_v)

三种常见融合策略
======================

1. 加权平均融合
------------------------------

最直接的方法：对两种分数做归一化，然后按权重混合。

.. code-block:: python

   def weighted_fusion(bm25_scores, vec_scores, alpha=0.5):
       """
       加权平均融合。

       alpha 的选择：
       - alpha=0.3：向量主导（适合语义搜索）
       - alpha=0.7：BM25 主导（适合精确匹配）
       - alpha=0.5：均衡
       """
       bm25_norm = minmax_normalize(bm25_scores)
       vec_norm = minmax_normalize(vec_scores)
       return alpha * bm25_norm + (1 - alpha) * vec_norm

加权平均的优势是简单、可解释。缺点是它对两种分数的"可靠性"一视同仁——
如果 BM25 的结果很烂（比如查询"苹果手机"，BM25 返回了字面匹配但语义无关的
结果），它还是会占 50% 的权重。

2. RRF（Reciprocal Rank Fusion）
----------------------------------

RRF 不依赖原始分数，而是基于**排序位置**计算融合得分。这是目前工业界
最推荐的混合检索方法。

.. code-block:: python

   def rrf_fusion(bm25_ranked, vec_ranked, k=60):
       """
       RRF 的核心公式：score(d) = sum(1 / (k + rank_i(d)))

       k 的作用：平滑参数。k 越大，排名靠后的文档越有机会出现。
       k=60 是经验值，在大多数数据集上表现稳定。
       """
       from collections import defaultdict
       scores = defaultdict(float)

       for rank, doc in enumerate(bm25_ranked):
           scores[doc] += 1.0 / (k + rank + 1)

       for rank, doc in enumerate(vec_ranked):
           scores[doc] += 1.0 / (k + rank + 1)

       return sorted(scores.items(), key=lambda x: x[1], reverse=True)

RRF 的优势：它不关心分数分布，只关心排序位置。这让它比加权平均更稳定。
你不需要调试归一化参数，RRF 开箱即用。

.. admonition:: RRF vs 加权平均
   :class: tip

   加权平均适合"两种检索的分数都是可靠的"场景；RRF 适合"不清楚分数分布"。
   建议 MVP 阶段用 RRF（参数少、不需调试），线上稳定后可以尝试加权平均
   来微调。

3. Score Fusion with Dynamic Alpha
--------------------------------------

在加权平均的基础上，根据查询类型动态调整融合权重。

.. code-block:: python

   class DynamicHybridSearch:
       """
       动态 alpha：精确查询时增加 BM25 权重，
       语义查询时增加向量权重。

       核心假设：不同查询类型需要不同的检索策略。
       """
       def search(self, query: str, k: int = 10) -> list:
           alpha = self._estimate_alpha(query)
           bm25_results = self.bm25.search(query, k=k * 2)
           vec_results = self.vector_db.similarity_search(query, k=k * 2)
           return self._rrf_fusion(bm25_results, vec_results, alpha)

       def _estimate_alpha(self, query: str) -> float:
           """
           估计 alpha。

           判断逻辑很简单：
           - 包含数字 → 可能是 ID/订单号 → BM25 优先 (alpha=0.7)
           - 包含特殊字符 → 可能是代码/路径 → BM25 优先 (alpha=0.6)
           - 其他 → 自然语言 → 向量优先 (alpha=0.3)
           """
           if any(c.isdigit() for c in query):
               return 0.7
           if any(c in query for c in ["#", "-", "_", "/"]):
               return 0.6
           return 0.3

完整实现
============

.. code-block:: python

   class CompleteHybridEngine:
       """
       生产级的混合搜索引擎。

       流程：
       1. BM25 和向量检索并行执行
       2. 双方各自返回 top-k 候选
       3. RRF 融合排序
       4. 返回最终结果
       """
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
           bm25_ranked = np.argsort(bm25_scores)[::-1]
           vec_ranked = np.argsort(vec_scores)[::-1]

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
   Agent 场景中两者都需要。比如查询"删除 user_id=100 的记录"——BM25 能精确
   命中 user_id=100 的文档，语义检索可能返回一堆关于"删除用户"的无关内容。

.. admonition:: 误区 2：alpha 是固定的
   :class: tip

   alpha 应该在 0.2-0.8 之间根据查询类型动态调整。Agent 框架可以在调用检索前
   先分类查询，选择不同的混合策略。这种动态调整可以让检索质量提升 10-20%。

.. admonition:: 误区 3：RRF 的 k 值不重要
   :class: caution

   k 值影响排名靠后文档的"复活机会"。k 越小，排名靠前的文档权重越大；
   k 越大，排名靠后的文档越有机会被重新发现。经验值 k=60，但在短文本
   搜索场景（如代码搜索），k=20 效果更好。
