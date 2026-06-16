.. _chapter-04-communication:

===============================
Agent 间通信模式
===============================

多 Agent 系统的核心挑战是通信。Agent 之间的消息需要结构化、可追溯且高效。
消息格式和通信模式直接影响系统的可扩展性和可靠性。

标准消息协议
================

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

三大通信模式
================

1. 广播模式（Broadcast）
------------------------------

一个 Agent 向所有其他 Agent 发布消息，适合任务分发和状态同步。

.. code-block:: python

   class BroadcastChannel:
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

       def receive(self, agent_id: str, msg: AgentMessage):
           """特定 Agent 接收消息"""
           self.message_log.append(msg)
           return self.agents[agent_id].process(msg)

2. 点对点模式（Peer-to-Peer）
------------------------------

两个 Agent 直接通信，适合协作任务。

.. code-block:: python

   class PeerToPeer:
       def __init__(self):
           self.connections = {}  # {(sender, receiver): [messages]}

       def send(self, sender: str, receiver: str, msg: AgentMessage):
           key = (sender, receiver)
           if key not in self.connections:
               self.connections[key] = []
           self.connections[key].append(msg)
           return f"消息已发送：{sender} → {receiver}"

       def query(self, sender: str, receiver: str, question: str) -> str:
           """A 向 B 发起查询"""
           msg = AgentMessage(
               sender=sender, receiver=receiver,
               msg_type="request", content=question
           )
           self.send(sender, receiver, msg)
           # 接收方处理并回复
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
       """所有 Agent 共享的工作空间"""
       def __init__(self):
           self.board = {}  # {key: {"value": any, "author": str, "timestamp": float}}

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
           results = []
           for key, entry in self.board.items():
               if query.lower() in str(entry["value"]).lower():
                   results.append((key, entry))
           return results

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

.. admonition:: Agent 通信协议对比
   :class: tip

   2026 年主流的 Agent 通信协议：
   - **MCP**（Model Context Protocol）：Claude Code/Cursor 使用的工具调用协议
   - **A2A**（Agent-to-Agent）：Google 提出的 Agent 间通信标准
   - **GSP**（Google Agent Collaboration）：多 Agent 协作框架

   选择建议：小型系统用点对点，中型系统用黑板，大型系统用混合架构。

通信可靠性保障
================

.. code-block:: python

   class ReliableMessaging:
       def __init__(self, max_retries=3, timeout=30):
           self.pending = {}   # msg_id → callback
           self.retries = {}
           self.max_retries = max_retries
           self.timeout = timeout

       def send_with_ack(self, msg: AgentMessage) -> bool:
           """带确认的可靠发送"""
           for attempt in range(self.max_retries):
               try:
                   self._deliver(msg)
                   # 等待确认
                   ack = self._wait_ack(msg.msg_id, self.timeout)
                   if ack:
                       return True
               except TimeoutError:
                   continue
           return False

       def _deliver(self, msg):
           pass  # 实际发送逻辑
