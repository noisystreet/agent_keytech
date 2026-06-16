.. _chapter-04-orchestration:

===============================
编排模式
===============================

编排（Orchestration）是多 Agent 系统的"导演"，决定了 Agent 之间的工作流和交互顺序。

.. mermaid::

   flowchart LR
       Orc[编排器] --> A1[分析师 Agent]
       Orc --> A2[搜索 Agent]
       Orc --> A3[摘要 Agent]
       A1 --> A2
       A2 --> A3
       A3 --> Orc

三种经典编排模式
====================

- **顺序链**：Agent A → Agent B → Agent C，每个 Agent 处理前一阶段的输出
- **路由**：编排器根据任务类型分发到不同 Agent
- **分层**：高级 Agent 分解任务，低级 Agent 执行子任务

.. code-block:: python

   class SequentialOrchestrator:
       def __init__(self, agents: List[Agent]):
           self.agents = agents

       def run(self, task: str) -> str:
           result = task
           for agent in self.agents:
               result = agent.run(result)
           return result
