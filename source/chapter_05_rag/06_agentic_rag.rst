.. _chapter-05-agentic-rag:

===============================
Agentic RAG 与 Graph RAG
===============================

传统的 RAG 是"检索→生成"的单向流水线，而 **Agentic RAG** 将 Agent 的规划与
决策能力引入 RAG 流程中，让检索不再是"一次性的"而是"迭代的"和"自适应的"。

从流水线到智能体
====================

.. list-table::
   :header-rows: 1

   * - 维度
     - 传统 RAG
     - Agentic RAG
   * - 检索次数
     - 一次检索
     - 多轮迭代检索
   * - 查询构造
     - 直接使用用户 query
     - 重写、分解、扩写
   * - 信息融合
     - 简单拼接
     - 根据上下文判断
   * - 决策能力
     - 无
     - 判断是否需补充检索
   * - 错误恢复
     - 无
     - 检索不足时自动重试

.. mermaid::

   flowchart TD
       Q[用户问题] --> Analyze[Agent 分析需求]
       Analyze --> Decompose{是否需要分解?}
       Decompose -- 是 --> Subs[拆分为子问题]
       Subs --> Retrieve1[检索子问题 1]
       Subs --> Retrieve2[检索子问题 2]
       Subs --> Retrieve3[检索子问题 3]
       Decompose -- 否 --> Retrieve[直接检索]
       Retrieve1 --> Eval{信息是否充足?}
       Retrieve2 --> Eval
       Retrieve3 --> Eval
       Retrieve --> Eval
       Eval -- 不充足 --> Rewrite[重写查询<br>再次检索]
       Rewrite --> Retrieve
       Eval -- 充足 --> Generate[生成最终回答]

.. code-block:: python

   class AgenticRAG:
       def __init__(self, llm, retriever, max_iterations=3):
           self.llm = llm
           self.retriever = retriever
           self.max_iterations = max_iterations

       def answer(self, question: str) -> str:
           context = []
           query = question

           for i in range(self.max_iterations):
               # 检索
               docs = self.retriever.search(query, k=5)
               context.extend(docs)

               # Agent 判断是否还需要更多信息
               prompt = (
                   f"原问题：{question}\n"
                   f"当前已收集的信息：\n{self._format_context(context)}\n\n"
                   f"基于当前信息，能否回答原问题？"
                   f"如果不能，需要进一步搜索什么？"
               )
               decision = self.llm.generate(prompt, temperature=0.0)

               if "能回答" in decision or "可以回答" in decision:
                   break

               # 根据 Agent 的反馈重写查询
               query = decision.strip()

           # 最终生成
           final_prompt = (
               f"基于以下信息回答问题：\n{self._format_context(context)}\n\n"
               f"问题：{question}\n\n请给出准确、简洁的回答。"
           )
           return self.llm.generate(final_prompt, temperature=0.0)

       def _format_context(self, docs):
           return "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))

Graph RAG
=============

Graph RAG（Microsoft, 2024）将知识图谱引入 RAG 流程，解决传统 RAG 对
**多跳关系** 和 **全局性问题** 处理能力不足的缺陷。

.. list-table::
   :header-rows: 1

   * - 特性
     - 传统 RAG
     - Graph RAG
   * - 知识组织
     - 扁平文档片段
     - 实体-关系图
   * - 检索粒度
     - 文本块相似度
     - 实体 + 关系 + 社区
   * - 跨文档推理
     - 弱（依赖 LLM 拼接）
     - 强（图结构天然支持）
   * - 全局问题
     - 差
     - 好（通过社区摘要）
   * - 实现复杂度
     - 低
     - 高

Graph RAG 的核心流程：

.. mermaid::

   flowchart LR
       Docs[原始文档] --> Extract[LLM 提取实体和关系]
       Extract --> Graph[知识图谱]
       Graph --> Community[社区检测与摘要]
       Community --> Index[向量索引]
       Index --> Search[检索: 实体+社区+文本]
       Search --> Answer[生成回答]

.. code-block:: python

   # Graph RAG 的核心思路（简化示例）
   class GraphRAG:
       def __init__(self, llm, vector_store):
           self.llm = llm
           self.vector_store = vector_store
           self.graph = {}  # {entity: [(relation, target_entity)]}

       def build_graph(self, documents: list):
           """从文档中提取实体关系构建知识图谱"""
           for doc in documents:
               entities = self.llm.generate(
                   f"从以下文本中提取实体和关系（JSON 格式）：\n{doc}"
               )
               # 解析并存入图结构
               self._parse_and_store(entities, doc)

       def retrieve(self, query: str) -> list:
           # 向量检索相关文本块
           text_results = self.vector_store.similarity_search(query, k=5)

           # 提取查询中的实体
           entities = self.llm.generate(
               f"从问题中提取关键实体：{query}"
           )

           # 图检索：查找关联实体和关系
           graph_results = []
           for entity in entities:
               if entity in self.graph:
                   graph_results.extend(self.graph[entity])

           # 合并结果
           return text_results + graph_results

.. admonition:: 如何选择？
   :class: tip

   - **简单问答场景**：传统 RAG 足够，不需要 Agentic RAG
   - **复杂多步推理**：Agentic RAG 更适合（如"对比 A 和 B 公司的产品策略"）
   - **全局/综述性问题**：Graph RAG 更优（如"XX 领域的研究现状"）
   - **生产环境建议**：从传统 RAG 起步，根据检索质量瓶颈逐步升级

参考文献
============

- Microsoft Research, "GraphRAG: Unlocking LLM Discovery on Narrative Private Data", 2024
- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey", 2023
