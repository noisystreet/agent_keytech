.. _chapter-05-reranking:

===============================
重排序（Reranking）
===============================

检索结果中，排在前面的文档不一定是最相关的。**重排序（Reranking）** 在初步检索后
使用更精确的模型对候选文档重新打分，是提升 RAG 系统检索质量的关键环节。

为什么需要重排序？
====================

.. list-table::
   :header-rows: 1

   * - 检索阶段
     - 召回策略
     - 特点
     - 延迟
   * - 第一阶段（Retrieval）
     - 稠密/稀疏检索
     - 速度快，召回率高
     - ~10ms
   * - 第二阶段（Reranking）
     - 交叉编码器（Cross-Encoder）
     - 精度高，但计算量大
     - ~100-500ms

两者的关系是：**第一阶段尽可能多召回（高 Recall），第二阶段精确排序（高 Precision）**。
典型做法是检索 top-100 文档，重排序后只取 top-5。

交叉编码器 vs 双编码器
==========================

.. mermaid::

   flowchart LR
       subgraph Bi-Encoder [双编码器 - 第一阶段]
           Q1[查询] --> E1[编码器]
           D1[文档] --> E2[编码器]
           E1 --> V1[向量]
           E2 --> V2[向量]
           V1 --> Sim[余弦相似度]
           V2 --> Sim
       end

       subgraph Cross-Encoder [交叉编码器 - 第二阶段]
           Q2[查询] --> Concat[拼接]
           D2[文档] --> Concat
           Concat --> CE[共享编码器]
           CE --> Score[相关性分数 0-1]
       end

.. code-block:: python

   from sentence_transformers import CrossEncoder

   class Reranker:
       def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
           # 交叉编码器，同时编码查询和文档
           self.model = CrossEncoder(model_name)

       def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
           # 构造 (query, doc) 对
           pairs = [[query, doc] for doc in documents]

           # 交叉编码器打分
           scores = self.model.predict(pairs)

           # 按分数排序
           scored = list(zip(documents, scores))
           scored.sort(key=lambda x: x[1], reverse=True)

           return scored[:top_k]

   # 使用示例
   reranker = Reranker()
   retrieved_docs = retriever.search("什么是 Agent？", k=20)
   reranked = reranker.rerank("什么是 Agent？", retrieved_docs, top_k=3)

.. admonition:: 重排序的收益
   :class: tip

   在生产 RAG 系统中，增加一个重排序环节通常可以将最终回答质量提升 **10-30%**，
   尤其是当知识库中存在语义相似但实际无关的文档时（例如"苹果"指水果还是公司）。
   由于只对 top-k 结果重排序，计算成本可控。

常用重排序模型
==================

.. list-table::
   :header-rows: 1

   * - 模型
     - 特点
     - 推荐场景
   * - BAAI/bge-reranker-v2-m3
     - 支持多语言，轻量
     - 中文场景首选
   * - Cohere Rerank
     - API 调用，零部署成本
     - 快速验证
   * - BAAI/bge-reranker-v2-minicpm
     - 更轻量，推理快
     - 延迟敏感场景
   * - cross-encoder/ms-marco-MiniLM-L6-v2
     - 英文效果优秀
     - 英文场景

重排序在 Agent 中的特殊价值
==============================

Agent 场景中，检索结果的质量直接影响 Agent 的后续决策。重排序在以下场景
尤其重要：

1. **工具选择**：从大量 API 描述中精确匹配用户意图对应的工具
2. **记忆检索**：从长期记忆中召回最相关的历史交互记录
3. **Few-shot 示例筛选**：从示例池中挑选最佳示例用于上下文学习

.. code-block:: python

   # Agent 中的重排序：选择最佳 Few-shot 示例
   class ExampleSelector:
       def __init__(self, examples, reranker):
           self.examples = examples
           self.reranker = reranker

       def select(self, query: str, k: int = 3) -> list:
           # 先用 BM25 粗召回
           candidates = bm25_retrieve(query, self.examples, k=20)
           # 再用重排序精筛
           return self.reranker.rerank(query, candidates, top_k=k)
