.. _chapter-02-memory:

===============================
记忆系统
===============================

记忆是 Agent 区别于无状态 API 调用的关键特性。Agent 的记忆系统通常分为三个层次：

.. list-table::
   :header-rows: 1

   * - 记忆层次
     - 类比人类
     - 实现方式
     - 容量
   * - 短期记忆
     - 工作记忆
     - 上下文窗口（in-context）
     - 有限（~128k token）
   * - 长期记忆
     - 记忆存储
     - 向量数据库 + RAG
     - 理论上无限
   * - 程序记忆
     - 本能反射
     - 微调后的模型权重
     - 永久

.. code-block:: python

   class HierarchicalMemory:
       def __init__(self, llm, vector_db):
           self.short_term = []  # 当前对话历史
           self.long_term = vector_db
           self.llm = llm

       def get_context(self, query: str) -> str:
           # 1. RAG 检索相关长期记忆
           memories = self.long_term.similarity_search(query, k=5)

           # 2. 组合检索结果 + 短期记忆
           context = f"""
           长期记忆（相关片段）：
           {format_memories(memories)}

           当前对话：
           {format_short_term(self.short_term)}
           """
           return context

       def add(self, role: str, content: str):
           self.short_term.append({"role": role, "content": content})
           # 自动摘要并存入长期记忆
           if len(self.short_term) > THRESHOLD:
               summary = self.llm.summarize(self.short_term)
               self.long_term.add(summary)
               self.short_term = self.short_term[-KEEP_LAST:]  # 只保留最近 K 轮
