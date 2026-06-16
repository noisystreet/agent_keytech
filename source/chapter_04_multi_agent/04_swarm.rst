.. _chapter-04-swarm:

===============================
集群与生态
===============================

集群（Swarm）是一种去中心化的多 Agent 架构，灵感来自蜂群、蚁群的自然现象。
每个 Agent 独立决策、局部感知，但整体表现出涌现的智能行为。

集群特性
============

- **去中心化**：没有单一控制节点，鲁棒性高
- **局部信息**：每个 Agent 只感知局部环境，不需要全局信息
- **涌现行为**：简单个体 + 简单规则 → 复杂整体行为

.. code-block:: python

   class SwarmAgent:
       def __init__(self, role, llm, local_tools):
           self.role = role
           self.llm = llm
           self.tools = local_tools
           self.messages = []  # 本地消息队列

       def step(self, global_context):
           # 感知局部环境
           local_context = self.filter_relevant(global_context)
           # 决定行动
           action = self.llm.generate(f"""
               你是 {self.role}。当前上下文：{local_context}
               决定你的下一步行动。
           """)
           return self.execute(action)

.. admonition:: 集群的工程挑战
   :class: caution

   集群模式虽然优雅，但在工程实践中面临巨大挑战：
   - 通信开销：n 个 Agent 的通信量是 O(n²)
   - 收敛困难：缺乏全局协调可能导致死循环
   - 调试困难：涌现行为难以预测和复现
   生产环境更常用的是"混合架构"：分层编排 + 局部集群。
