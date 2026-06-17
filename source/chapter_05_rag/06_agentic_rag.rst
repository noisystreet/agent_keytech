.. _chapter-05-agentic-rag:

===============================
Agentic RAG 与 Graph RAG
===============================

传统的 RAG 是"检索→生成"的单向流水线，而 **Agentic RAG** 将 Agent 的规划与
决策能力引入 RAG 流程中，让检索不再是"一次性的"而是"迭代的"和"自适应的"。

传统 RAG 有一句话概括就是"一次性检索，一次性生成"。这种模式对简单问题有效，
但面对一个需要多角度分析的问题（比如"对比 A 公司和 B 公司的产品策略差异"），
单次检索往往无法找到所有需要的信息。Agentic RAG 解决的就是这个问题——
让检索过程本身变得智能。

从流水线到智能体
====================

.. list-table::
   :header-rows: 1

   * - 维度
     - 传统 RAG
     - Agentic RAG
   * - 检索次数
     - 一次检索，不管够不够
     - 多轮迭代，直到信息充足
   * - 查询构造
     - 直接使用用户 query
     - 重写、分解、扩写、翻译
   * - 信息融合
     - 简单拼接所有结果
     - 根据上下文选择性整合
   * - 决策能力
     - 无（检索完就生成）
     - 判断是否需补充检索
   * - 错误恢复
     - 无（检索错就生成错）
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
       """
       Agentic RAG：Agent 驱动的迭代检索。

       关键区别：
       1. Agent 判断是否需要补充检索（而不是预设固定步数）
       2. Agent 重写查询以获取更精确的结果
       3. Agent 在信息充足时自动停止
       """
       def __init__(self, llm, retriever, max_iterations=3):
           self.llm = llm
           self.retriever = retriever
           self.max_iterations = max_iterations

       def answer(self, question: str) -> str:
           context = []
           query = question

           for i in range(self.max_iterations):
               # 检索当前 query
               docs = self.retriever.search(query, k=5)
               context.extend(docs)

               # Agent 判断是否还需要更多信息
               decision = self._assess_information(question, context)

               if decision["sufficient"]:
                   break

               # 根据分析结果重写查询
               query = decision["next_query"]

           # 最终生成
           return self._generate_answer(question, context)

       def _assess_information(self, question, context) -> dict:
           """
           Agent 判断当前信息是否充足。

           这个判断是 Agentic RAG 的核心——Agent 需要知道
           "我已经知道什么"和"我还需要知道什么"。
           """
           prompt = f"""
           原问题：{question}

           当前已收集的信息：
           {self._format_context(context)}

           请分析：
           1. 当前信息能否完整回答原问题？（能/不能）
           2. 如果不能，还需要搜索什么？
           3. 请给出下一步的搜索查询

           输出格式：
           充足: 能/不能
           下一步查询: ...
           """
           response = self.llm.generate(prompt, temperature=0.0)
           return self._parse_decision(response)

       def _generate_answer(self, question, context):
           prompt = f"""
           基于以下信息回答问题。只使用提供的信息，不要编造。

           信息：
           {self._format_context(context)}

           问题：{question}
           """
           return self.llm.generate(prompt, temperature=0.0)

查询分解：处理复杂问题
============================

面对复杂问题，Agentic RAG 的第一步往往是把问题拆成子问题。

.. code-block:: python

   class QueryDecomposition:
       """将复杂问题分解为多个子问题，分别检索"""
       def decompose(self, question: str) -> list:
           prompt = f"""
           将以下复杂问题拆解为 2-4 个独立的子问题。
           每个子问题应该能通过单次检索找到答案。

           问题：{question}

           输出格式（每行一个子问题）：
           """
           response = self.llm.generate(prompt, temperature=0.0)
           return [q.strip() for q in response.split("\n") if q.strip()]

   # 使用示例
   decomposer = QueryDecomposition()
   question = "对比 OpenAI 和 Anthropic 在 AI 安全方面的不同策略"
   sub_questions = decomposer.decompose(question)
   # 输出：
   # 1. OpenAI 的 AI 安全策略是什么？
   # 2. Anthropic 的 AI 安全策略是什么？
   # 3. 两者的安全策略有哪些主要差异？

Graph RAG
=============

Graph RAG（Microsoft, 2024）将知识图谱引入 RAG 流程，解决传统 RAG 对
**多跳关系** 和**全局性问题** 处理能力不足的缺陷。

传统 RAG 把文档切成碎片，每个碎片独立检索。但当问题需要跨多个文档
整合信息时（比如"这本书中所有提到 AI 风险的段落有哪些共同点？"），
碎片化的检索就很难处理。Graph RAG 通过构建实体-关系图来解决这个问题。

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
     - 弱（依赖 LLM 拼接信息）
     - 强（图结构天然支持关联）
   * - 全局问题
     - 差（每个片段只有局部信息）
     - 好（通过社区摘要获取全局视图）
   * - 实现复杂度
     - 低
     - 高（需要建图、社区检测）

Graph RAG 的完整工作流程：

.. mermaid::

   flowchart LR
       Docs[原始文档] --> Extract[LLM 提取实体和关系]
       Extract --> Graph[知识图谱]
       Graph --> Community[社区检测<br>Leiden 算法]
       Community --> Index[向量索引<br>社区摘要]
       Index --> Search[检索<br>实体 + 社区 + 文本]
       Search --> Answer[生成回答]

.. code-block:: python

   class GraphRAG:
       """
       Graph RAG 的简化实现。

       核心步骤：
       1. 从文档中提取实体和关系
       2. 构建知识图谱
       3. 检测社区（相关实体的聚类）
       4. 检索时同时搜索实体、社区和原始文本
       """
       def __init__(self, llm, vector_store):
           self.llm = llm
           self.vector_store = vector_store
           self.graph = {"entities": {}, "relations": []}
           self.communities = []

       def build_graph(self, documents: list):
           """从文档中提取实体关系构建知识图谱"""
           for doc in documents:
               # LLM 提取实体和关系
               extracted = self.llm.generate(f"""
                   从以下文本中提取所有实体和它们之间的关系。
                   输出 JSON 格式：
                   {{"entities": ["实体1", "实体2", ...],
                     "relations": [["实体1", "关系", "实体2"], ...]}}

                   文本：{doc}
               """)
               parsed = json.loads(extracted)
               self._add_to_graph(parsed)
               self.vector_store.add(doc)

           # 社区检测（识别紧密相关的实体组）
           self._detect_communities()

       def retrieve(self, query: str, k: int = 5) -> list:
           results = []

           # 1. 向量检索相关文本
           text_results = self.vector_store.similarity_search(query, k=k)

           # 2. 提取查询中的实体，做图检索
           entities = self.llm.generate(
               f"从问题中提取关键实体（逗号分隔）：{query}"
           ).split(",")

           for entity in entities:
               entity = entity.strip()
               if entity in self.graph["entities"]:
                   # 查找关联实体和社区信息
                   related = self._get_related(entity)
                   results.extend(related)

           # 3. 合并去重
           seen = set()
           unique_results = []
           for r in text_results + results:
               if r not in seen:
                   seen.add(r)
                   unique_results.append(r)

           return unique_results[:k]

如何选择？
============

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐方案
     - 原因
   * - 简单事实问答
     - 传统 RAG
     - 一次检索就够了，不需要额外开销
   * - 多步推理
     - Agentic RAG
     - 需要迭代检索和查询重写
   * - 跨文档分析
     - Graph RAG
     - 实体关系天然支持跨文档推理
   * - 全局综述
     - Graph RAG
     - 社区摘要能提供全局视角
   * - 生产环境建议
     - 从传统 RAG 起步
     - 根据检索质量瓶颈逐步升级

.. admonition:: 不要过早升级
   :class: tip

   很多团队一上来就上 Graph RAG，结果发现：
   - 建图成本高（大量 LLM 调用）
   - 对于简单问题，效果和传统 RAG 差不多
   - 维护图结构的复杂度远超预期

   建议：从传统 RAG 开始，当遇到"跨文档推理"或"全局性问题"的瓶颈时，
   再逐步引入 Agentic RAG 和 Graph RAG。
