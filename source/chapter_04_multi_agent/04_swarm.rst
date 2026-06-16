.. _chapter-04-swarm:

===============================
集群与生态
===============================

集群（Swarm）是一种去中心化的多 Agent 架构，灵感来自蜂群、蚁群等
自然智能现象。每个 Agent 独立决策、局部感知，但整体表现出涌现的智能行为。
与编排模式不同，集群没有中央控制节点。

集群 vs 编排
================

.. list-table::
   :header-rows: 1

   * - 维度
     - 编排模式
     - 集群模式
   * - 控制方式
     - 集中式（中央编排器）
     - 去中心化（各自决策）
   * - 决策粒度
     - 全局优化
     - 局部最优
   * - 通信模式
     - 编排器分发任务
     - 局部消息传递
   * - 伸缩性
     - 随 Agent 数量线性下降
     - 近线性伸缩
   * - 单点故障
     - 编排器故障 → 系统崩溃
     - 无单点故障
   * - 适用规模
     - 3-10 个 Agent
     - 10-100+ 个 Agent

集群 Agent 实现
====================

.. code-block:: python

   class SwarmAgent:
       def __init__(self, role, llm, local_tools):
           self.role = role
           self.llm = llm
           self.tools = local_tools
           self.messages = []        # 本地消息队列
           self.local_state = {}     # 局部状态
           self.neighbors = []       # 相邻 Agent

       def step(self, global_context: str) -> str:
           # 1. 感知局部环境
           local_context = self._filter_relevant(global_context)

           # 2. 处理邻居消息
           neighbor_signals = self._process_messages()

           # 3. 决定行动
           action = self.llm.generate(f"""
               你是 {self.role}。
               当前上下文：{local_context}
               邻居信号：{neighbor_signals}
               你的局部状态：{self.local_state}

               决定你的下一步行动。使用可用工具。
           """)

           # 4. 执行并更新状态
           result = self._execute_action(action)
           self.local_state["last_action"] = action
           self.local_state["last_result"] = result

           return result

       def _filter_relevant(self, global_context) -> str:
           """从全局上下文中提取与自身角色相关的部分"""
           return global_context  # 简化实现

       def send_to_neighbor(self, neighbor, message):
           """向相邻 Agent 发送消息"""
           neighbor.messages.append({
               "from": self.role,
               "content": message,
               "timestamp": time.time()
           })

       def _process_messages(self) -> str:
           """处理消息队列中的邻居消息"""
           signals = []
           while self.messages:
               msg = self.messages.pop(0)
               signals.append(f"[来自 {msg['from']}]: {msg['content']}")
           return "\n".join(signals)

集群协调模式
================

1. 领导者选举（Leader Election）
------------------------------

集群中动态选举协调者，避免永久中心化节点。

.. code-block:: python

   class LeaderElection:
       def __init__(self, agents: list):
           self.agents = agents
           self.leader = None

       def elect_leader(self, task: str) -> SwarmAgent:
           """根据任务特点选举最合适的领导者"""
           scores = []
           for agent in self.agents:
               score = self._score_fitness(agent, task)
               scores.append((score, agent))
           self.leader = max(scores, key=lambda x: x[0])[1]
           return self.leader

       def _score_fitness(self, agent, task) -> float:
           prompt = f"Agent 角色：{agent.role}\n任务：{task}\n适合度（0-1）："
           return float(llm.generate(prompt, temperature=0.0))

2. 声望系统（Reputation）
------------------------------

Agent 根据历史表现获得声望分，高分 Agent 有更高话语权。

.. code-block:: python

   class ReputationSystem:
       def __init__(self, decay=0.95):
           self.reputations = {}  # {agent_id: score}
           self.decay = decay

       def report_result(self, agent_id: str, success: bool):
           if agent_id not in self.reputations:
               self.reputations[agent_id] = 0.5
           adjustment = 0.1 if success else -0.1
           self.reputations[agent_id] = max(0, min(1,
               self.reputations[agent_id] + adjustment
           ))

       def get_weight(self, agent_id: str) -> float:
           return self.reputations.get(agent_id, 0.5)

3. 共识机制（Consensus）
------------------------------

集群中多个 Agent 通过投票达成一致决策。

.. code-block:: python

   class ConsensusMechanism:
       def __init__(self, agents, threshold=0.6):
           self.agents = agents
           self.threshold = threshold

       def reach_consensus(self, question: str) -> tuple:
           """返回 (是否达成共识, 共识结果, 支持率)"""
           votes = {}
           for agent in self.agents:
               vote = agent.run(f"{question}\n请仅回答：选项 A / 选项 B / 不确定")
               votes[agent.role] = vote

           support_ratio = sum(1 for v in votes.values() if v == "选项 A") / len(votes)
           consensus = support_ratio >= self.threshold
           decision = "选项 A" if support_ratio >= 0.5 else "选项 B"
           return consensus, decision, support_ratio

混合架构：分层 + 集群
========================

在实际生产中，纯粹的去中心化集群很少使用。更常见的是**混合架构**
——上层用分层编排进行任务分解，下层用集群进行并行处理。

.. mermaid::

   flowchart TD
       Director[总导演 Agent] --> Team1[任务组 1<br>集群协调者]
       Director --> Team2[任务组 2<br>集群协调者]
       Director --> Team3[任务组 3<br>集群协调者]
       Team1 --> W1[工人 Agent 1]
       Team1 --> W2[工人 Agent 2]
       Team1 --> W3[工人 Agent 3]
       Team2 --> W4[工人 Agent 4]
       Team2 --> W5[工人 Agent 5]
       Team3 --> W6[工人 Agent 6]
       Team3 --> W7[工人 Agent 7]

.. admonition:: 集群的工程挑战
   :class: caution

   集群模式虽然优雅，但在工程实践中面临巨大挑战：
   - **通信开销**：n 个 Agent 的通信量是 O(n²)
   - **收敛困难**：缺乏全局协调可能导致死循环
   - **调试困难**：涌现行为难以预测和复现
   - **一致性保证**：去中心化环境中难以保证数据一致性

   生产环境建议使用**混合架构**：分层编排（宏观） + 局部集群（微观）。

参考文献
============

- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior", 2023
- Wu et al., "Swarm Algorithms for Multi-Agent Systems", 2024
