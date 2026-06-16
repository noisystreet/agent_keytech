.. _chapter-05-reranking:

===============================
重排序（Reranking）
===============================

检索结果中，排在前面的文档不一定是最相关的。**重排序（Reranking）** 在
初步检索后使用更精确的模型对候选文档重新打分，是 RAG 系统中"投入产出比"
最高的优化。

如果你已经部署了一个 RAG 系统，但对它的检索质量不满意，**重排序应该是你
尝试的第一个优化方向**。它不需要更换嵌入模型、不需要重新索引数据，
只需要在检索流水线上增加一个环节，就能将最终回答质量提升 10-30%。

为什么需要重排序？
====================

第一阶段的检索（稠密/稀疏检索）追求的是"快速召回"——在最短时间内
从海量文档中找到可能相关的候选。它用的是双编码器架构：查询和文档
各自编码，然后用余弦相似度匹配。

这种架构的优点是速度快（可以预先编码所有文档），但缺点是精度有限。
因为查询和文档是**独立编码**的，它们之间的交互只发生在最后的向量
比较这个瞬间。模型无法在编码时"看到"对方的语义。

重排序使用**交叉编码器（Cross-Encoder）**：把查询和文档拼接在一起，
用一个共享的 Transformer 同时编码它们。这样，模型在编码时就能
看到查询和文档之间的微妙关系。

.. list-table::
   :header-rows: 1

   * - 检索阶段
     - 召回策略
     - 特点
     - 延迟
   * - 第一阶段（Retrieval）
     - 稠密/稀疏检索
     - 速度快，召回率高
     - ~10ms（百万级）
   * - 第二阶段（Reranking）
     - 交叉编码器
     - 精度高，但计算量大
     - ~100-500ms（百级）

典型做法是先检索 top-100 文档，重排序后只取 top-5。这样既有高召回率
（第一阶段），又有高精度（第二阶段）。

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

   在一个电商客服 Agent 的数据中，不使用重排序时 top-5 准确率为 72%，
   增加重排序后提升到 89%。这意味着 Agent 看到正确信息的概率增加了 17%，
   直接体现在回答准确率的提升上。

重排序的候选数量选择
======================

一个常见的工程问题是：第一阶段应该召回多少候选文档交给重排序？

.. code-block:: python

   # 候选数量对效果和延迟的影响
   tradeoff = {
       20:  {"accuracy": "基准", "latency": "50ms", "建议": "起步用这个"},
       50:  {"accuracy": "+5%",  "latency": "120ms", "建议": "生产推荐"},
       100: {"accuracy": "+8%",  "latency": "250ms", "建议": "高精度场景"},
       200: {"accuracy": "+9%",  "latency": "500ms", "建议": "收益递减"},
   }

50-100 是大多数场景的黄金区间。超过 100 后，精度提升非常有限，但延迟
线性增长。这是因为重排序模型只需要找到前几个最相关的文档——文档排名
越靠后，被选中的概率越低，投入的计算资源就越浪费。

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
   * - BAAI/bge-reranker-v2-minicpm
     - 更轻量，推理快
     - 延迟敏感场景
   * - Cohere Rerank
     - API 调用，零部署成本
     - 快速验证
   * - cross-encoder/ms-marco-MiniLM-L6-v2
     - 英文效果优秀
     - 英文场景
   * - jina-reranker-v2-base-multilingual
     - 多语言，8192 长上下文
     - 长文档重排序

重排序的两阶段策略实现
======================

.. code-block:: python

   class TwoStageRetriever:
       """
       两阶段检索：召回 + 重排序。
       第一阶段用 BM25 或向量检索快速召回候选。
       第二阶段用交叉编码器精确排序。
       """
       def __init__(self, stage1_retriever, reranker_model="BAAI/bge-reranker-v2-m3"):
           self.stage1 = stage1_retriever
           self.reranker = CrossEncoder(reranker_model)

       def search(self, query: str, k: int = 5, candidates: int = 50) -> list:
           # 第一阶段：快速召回大量候选
           stage1_results = self.stage1.search(query, k=candidates)

           # 第二阶段：精确重排序
           pairs = [[query, doc] for doc, _ in stage1_results]
           scores = self.reranker.predict(pairs)

           # 排序并返回 top-k
           scored = list(zip(stage1_results, scores))
           scored.sort(key=lambda x: x[1], reverse=True)
           return scored[:k]

重排序在 Agent 中的特殊价值
==============================

Agent 场景中，检索结果的质量直接影响 Agent 的后续决策。
重排序在以下场景尤其重要：

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

重排序 vs Embedding 降维
============================

一个容易混淆的点：重排序和嵌入降维解决的是不同的问题。

- **嵌入降维**（如 PCA）：减少向量的维度，降低存储和搜索成本。
  但它不改变检索的**精度上限**。

- **重排序**：用更强的模型重新评估候选文档，直接提升检索的
  **精度上限**。

如果你只能做一个优化，选重排序。它的回报更直接。
