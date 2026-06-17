.. _chapter-04-communication:

===============================
Agent 间通信模式
===============================

多 Agent 系统的核心挑战是通信。Agent 之间的消息需要结构化、可追溯且高效。
消息格式和通信模式直接影响系统的可扩展性和可靠性。

如果你构建过单 Agent 系统，你可能觉得 Agent 间通信只是一个"发消息"的问题。
但实际没那么简单。Agent 之间的通信和人与人之间的通信面临类似的问题：
消息会不会丢失？对方收到后怎么确认？如果对方回复慢了要不要等？
多个 Agent 同时发送消息会不会混乱？

标准消息协议
================

一个设计良好的 Agent 消息需要包含足够的信息让接收方理解上下文，
但不能太复杂以至于解析开销过大。

.. code-block:: python

   class AgentMessage:
       """Agent 间通信的标准消息格式"""
       def __init__(self, sender, receiver, msg_type, content, metadata=None):
           self.sender = sender          # 发送者 ID
           self.receiver = receiver      # 接收者 ID 或 "broadcast"
           self.msg_type = msg_type      # "request" / "response" / "broadcast" / "error"
           self.content = content        # 消息正文（结构化数据）
           self.metadata = metadata or {}
           self.timestamp = time.now()
           self.msg_id = str(uuid.uuid4())  # 唯一 ID，用于追溯

       def to_dict(self):
           return {
               "msg_id": self.msg_id,
               "sender": self.sender,
               "receiver": self.receiver,
               "type": self.msg_type,
               "content": self.content,
               "metadata": self.metadata,
               "timestamp": self.timestamp,
           }

message_id 是一个容易被忽略但极其重要的字段。没有唯一 ID，当 Agent B
收到两条内容相似的消息时，它没法判断是两条不同的请求还是一条重复发送。
这在分布式系统中是经典的"幂等性"问题。

三大通信模式
================

1. 广播模式（Broadcast）
------------------------------

一个 Agent 向所有其他 Agent 发布消息，适合任务分发和状态同步。

.. code-block:: python

   class BroadcastChannel:
       """广播模式：一个消息发送给所有 Agent"""
       def __init__(self):
           self.agents = {}
           self.message_log = []

       def register(self, agent_id: str, agent):
           self.agents[agent_id] = agent

       def broadcast(self, sender: str, msg: AgentMessage):
           for agent_id, agent in self.agents.items():
               if agent_id != sender:
                   agent.receive(msg)
           self.message_log.append(msg)

广播的优点是实现简单、传播快。缺点也很明显：
- 信息泛滥：n 个 Agent 各自发一条广播，总消息量是 O(n²)
- 无差异化：所有 Agent 都收到同样的消息，不管是否相关

2. 点对点模式（Peer-to-Peer）
------------------------------

两个 Agent 直接通信，适合协作任务。

.. code-block:: python

   class PeerToPeer:
       """点对点模式：两个 Agent 直接通信"""
       def __init__(self):
           self.connections = {}
           self.agents = {}

       def register(self, agent_id, agent):
           self.agents[agent_id] = agent

       def send(self, sender: str, receiver: str, msg: AgentMessage):
           key = (sender, receiver)
           if key not in self.connections:
               self.connections[key] = []
           self.connections[key].append(msg)

       def query(self, sender: str, receiver: str, question: str) -> str:
           """A 向 B 发起查询并等待回复"""
           msg = AgentMessage(
               sender=sender, receiver=receiver,
               msg_type="request", content=question
           )
           self.send(sender, receiver, msg)
           response = self.agents[receiver].process(msg)
           reply = AgentMessage(
               sender=receiver, receiver=sender,
               msg_type="response", content=response
           )
           self.send(receiver, sender, reply)
           return response

3. 黑板模式（Blackboard）
------------------------------

所有 Agent 读写共享的黑板，适合复杂系统中多个 Agent 的协作。

.. code-block:: python

   class Blackboard:
       """
       黑板模式：所有 Agent 共享一个工作空间。
       Agent 写入信息，其他 Agent 读取并处理。
       适合：松耦合、任务可分解的复杂系统。
       """
       def __init__(self):
           self.board = {}

       def write(self, agent_id: str, key: str, value: any):
           self.board[key] = {
               "value": value,
               "author": agent_id,
               "timestamp": time.time()
           }

       def read(self, key: str):
           entry = self.board.get(key)
           return entry["value"] if entry else None

       def search(self, query: str) -> list:
           """语义搜索黑板内容"""
           return [(key, entry) for key, entry in self.board.items()
                   if query.lower() in str(entry["value"]).lower()]

通信模式对比
================

.. list-table::
   :header-rows: 1

   * - 模式
     - 优点
     - 缺点
     - 适合场景
   * - 广播
     - 简单、信息传播快
     - 信息泛滥、O(n²) 通信量
     - 状态同步、通知
   * - 点对点
     - 精准、可追踪
     - 需要知道对方地址
     - 私密协作、委托任务
   * - 黑板
     - 松耦合、可扩展
     - 读写冲突、性能瓶颈
     - 复杂协作、知识共享

实际系统很少只用一种模式。常见的组合是：
- 上层用黑板（编排器写入任务，Worker Agent 读取并处理）
- 下层用点对点（Worker 之间交换中间结果）
- 异常时用广播（通知所有 Agent 停止工作）

Agent 通信协议概览
====================

2026 年主流的 Agent 通信协议：

.. list-table::
   :header-rows: 1

   * - 协议
     - 提出者
     - 定位
     - 适用场景
   * - MCP
     - Anthropic
     - Agent ↔ 工具
     - 工具调用标准化
   * - A2A
     - Google
     - Agent ↔ Agent
     - 跨 Agent 协作
   * - GSP
     - Google
     - Agent 协作框架
     - 多 Agent 工作流

选择建议：
- 小型系统（<5 个 Agent）：点对点通信就够了
- 中型系统（5-20 个 Agent）：黑板模式，耦合度低
- 大型系统（>20 个 Agent）：混合架构，不同层级用不同模式

通信可靠性保障
================

Agent 通信必须是可靠的。消息丢失可能导致整个任务失败。

.. code-block:: python

   class ReliableMessaging:
       """
       带确认和重试的可靠 Agent 通信。

       核心机制：
       1. 每条消息有唯一 ID（用于去重）
       2. 发送后等待 ACK
       3. 超时未收到 ACK 则重试
       4. 达到最大重试次数后报告失败
       """
       def __init__(self, max_retries=3, timeout=30):
           self.pending = {}
           self.retries = {}
           self.max_retries = max_retries
           self.timeout = timeout

       def send_with_ack(self, msg: AgentMessage) -> bool:
           """带确认的可靠发送"""
           for attempt in range(self.max_retries):
               try:
                   self._deliver(msg)
                   ack = self._wait_ack(msg.msg_id, self.timeout)
                   if ack:
                       return True
               except TimeoutError:
                   continue
           return False

       def _deliver(self, msg):
           """实际的消息发送逻辑"""
           pass
