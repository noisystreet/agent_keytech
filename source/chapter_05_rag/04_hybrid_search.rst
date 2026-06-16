.. _chapter-05-hybrid-search:

===============================
混合搜索
===============================

混合搜索结合了关键词搜索（BM25）和向量搜索的优势，是 RAG 系统达到生产级
检索质量的关键。

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
