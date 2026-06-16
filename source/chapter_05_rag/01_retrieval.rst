.. _chapter-05-retrieval:

===============================
检索策略
===============================

检索是 RAG 的第一步，决定了 Agent 能看到哪些信息。检索质量直接影响 Agent
回答的准确性和相关性。

检索策略对比
================

.. list-table::
   :header-rows: 1

   * - 方法
     - 原理
     - 优点
     - 缺点
   * - 稀疏检索（BM25）
     - 关键词匹配
     - 精确、可解释
     - 语义差异不识别
   * - 稠密检索（向量检索）
     - 语义相似度
     - 理解语义
     - 需要训练、冷启动
   * - 混合检索
     - 两者加权结合
     - 兼顾精确与语义
     - 实现复杂度高

.. code-block:: python

   class HybridRetriever:
       def __init__(self, bm25_index, vector_db, alpha=0.5):
           self.bm25 = bm25_index
           self.vector_db = vector_db
           self.alpha = alpha

       def retrieve(self, query: str, k: int = 10) -> List[Document]:
           # 两种检索各自打分
           bm25_results = self.bm25.search(query, k=k)
           vector_results = self.vector_db.similarity_search(query, k=k)

           # 加权融合
           scores = defaultdict(float)
           for doc, score in bm25_results:
               scores[doc.id] += self.alpha * score
           for doc, score in vector_results:
               scores[doc.id] += (1 - self.alpha) * score

           # 排序返回 Top-k
           ranked = sorted(scores.items(), key=lambda x: -x[1])
           return [docs[id] for id, _ in ranked[:k]]
