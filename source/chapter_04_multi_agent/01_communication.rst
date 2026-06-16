.. _chapter-04-communication:

===============================
Agent 间通信模式
===============================

多 Agent 系统的核心挑战是通信。Agent 之间的消息需要结构化、可追溯、且高效。

.. code-block:: python

   class AgentMessage:
       """Agent 间通信的标准消息格式"""
       def __init__(self, sender, receiver, msg_type, content, metadata=None):
           self.sender = sender          # 发送者 ID
           self.receiver = receiver      # 接收者 ID 或 "broadcast"
           self.msg_type = msg_type      # "request" / "response" / "broadcast"
           self.content = content        # 消息正文
           self.metadata = metadata or {}
           self.timestamp = time.now()

   # 广播模式：管理 Agent 广播任务信息
   # 点对点模式：两个 Agent 私下协作
   # 黑板模式：所有 Agent 写入共享黑板

通信模式
============

常见的有三种通信模式：**广播** 适合任务分发，**点对点** 适合协作，**黑板** 适合复杂系统。
