.. _chapter-02-context-window:

===============================
上下文窗口管理
===============================

上下文窗口（Context Window）是 LLM 单次能处理的 token 上限，也是 Agent 架构中
最核心的工程约束。理解和管理上下文窗口，直接决定了 Agent 的记忆深度、工具调用
能力和多轮对话质量。

上下文窗口是 Agent 的"工作台"
=================================

.. list-table::
   :header-rows: 1

   * - 组件
     - 占用 token 数（估算）
     - 说明
   * - System Prompt
     - 1k-4k
     - Agent 身份、行为规则、安全约束
   * - 对话历史
     - 2k-10k/轮
     - 每轮对话积累的历史
   * - 工具定义
     - 0.5k-2k/个
     - Function Calling 的 Schema
   * - 工具返回结果
     - 1k-20k/次
     - API 返回、RAG 检索结果
   * - RAG 上下文
     - 1k-5k/个文档
     - 检索到的相关文档片段
   * - 推理中间步骤
     - 0.5k-2k/步
     - CoT/ReAct 推理中间过程

.. mermaid::

   flowchart TD
       subgraph Budget [128K 窗口预算分配]
           Sys[System Prompt 3K] --> Remain1
           Remain1[剩余 125K] --> Tools[工具定义 4K]
           Tools --> Remain2[剩余 121K]
           Remain2 --> History[对话历史 20K]
           History --> Remain3[剩余 101K]
           Remain3 --> Retrieval[RAG 结果 15K]
           Retrieval --> Remain4[剩余 86K]
           Remain4 --> Reasoning[推理步骤 10K]
           Reasoning --> Free[剩余给模型输出]
       end

窗口管理策略
================

1. 滑动窗口（Sliding Window）
------------------------------

保留最近 N 轮对话，丢弃最早的历史。最简单、最常用的策略。

.. code-block:: python

   class SlidingWindow:
       def __init__(self, max_tokens=32000, reserve_ratio=0.3):
           self.max_tokens = max_tokens
           self.reserve = int(max_tokens * reserve_ratio)  # 为输出预留空间
           self.history = []

       def add(self, message: dict):
           self.history.append(message)
           self._trim()

       def _trim(self):
           total = sum(msg["tokens"] for msg in self.history)
           while total > self.max_tokens - self.reserve:
               dropped = self.history.pop(0)
               total -= dropped["tokens"]

       def get_context(self) -> list:
           return [msg["content"] for msg in self.history]

2. 摘要压缩（Summarization）
------------------------------

当历史超过阈值时，用 LLM 对早期对话生成摘要，替代原始轮次。

.. code-block:: python

   class SummarizationMemory:
       def __init__(self, llm, max_tokens=32000):
           self.llm = llm
           self.max_tokens = max_tokens
           self.summary = ""  # 压缩摘要
           self.recent = []    # 最近未压缩的历史

       def add(self, message: dict):
           self.recent.append(message)
           if self._total_tokens() > self.max_tokens * 0.7:
               self._compress()

       def _compress(self):
           prompt = f"请将以下对话压缩为 200 字以内的摘要，保留关键信息：\n{self.recent}"
           new_summary = self.llm.generate(prompt)
           self.summary = self.llm.generate(
               f"合并以下两段摘要：\n1. {self.summary}\n2. {new_summary}"
           )
           self.recent = []

       def get_context(self):
           return [
               {"role": "system", "content": f"对话摘要：{self.summary}"}
           ] + [msg["content"] for msg in self.recent[-5:]]

3. 关键信息优先（Priority-based）
--------------------------------------

为不同类型的信息分配不同的优先级和生存周期。核心假设：不是所有历史同等重要。

.. code-block:: python

   class PriorityMemory:
       PRIORITIES = {
           "user_goal": 100,      # 用户核心目标——永不丢弃
           "tool_result": 80,     # 工具返回的重要结果
           "user_preference": 90, # 用户偏好
           "chat_history": 30,    # 普通对话
           "greeting": 10,        # 问候语——最优先丢弃
       }

       def __init__(self, max_tokens=32000):
           self.max_tokens = max_tokens
           self.items = []  # [(priority, token_count, content)]

       def add(self, content: str, category: str, tokens: int):
           priority = self.PRIORITIES.get(category, 30)
           self.items.append((priority, tokens, content))
           self._evict()

       def _evict(self):
           # 按优先级升序排列，优先丢弃低优先级
           self.items.sort(key=lambda x: x[0])
           total = sum(item[1] for item in self.items)
           while total > self.max_tokens:
               dropped = self.items.pop(0)  # 丢弃最低优先级
               total -= dropped[1]

.. admonition:: 记忆 vs 上下文窗口
   :class: tip

   不要把上下文窗口管理等同于记忆系统。
   - **窗口管理** 解决的是"当前对话 LLM 能看到什么"
   - **记忆系统** 解决的是"Agent 跨会话能记住什么"

   窗口管理是短期的、token 预算导向的；记忆系统是长期的、语义导向的。
   实践中，两者配合使用：窗口管理保证当前推理质量，记忆系统通过 RAG 注入
   历史相关信息。

模型窗口大小对架构设计的影响
==============================

.. list-table::
   :header-rows: 1

   * - 窗口大小
     - 代表模型
     - 对 Agent 的影响
   * - 4K-8K
     - 早期 GPT-3.5
     - 只能支持极简 Agent，每轮需频繁摘要压缩
   * - 16K-32K
     - Mistral, Llama 3
     - 支持基本的工具调用 + 多轮对话
   * - 128K
     - GPT-4, Claude 3
     - 可承载完整 Agent 循环，含 RAG 上下文
   * - 200K-1M
     - Claude 4, Gemini 1.5 Pro
     - 可加载大型代码库、长文档，但不等于需要用完整个窗口
   * - 无限（Infini-Attn）
     - 前沿研究
     - 理论无上限，实践中仍是工程挑战

窗口越大越好吗？
==================

.. mermaid::

   flowchart LR
       A[大窗口] --> B[更多上下文]
       A --> C[更高的计算成本]
       A --> D[注意力稀释<br>Lost in the Middle]
       D --> E[相关信息被淹没<br>在大量无关内容中]
       D --> F[模型在新信息中<br>表现反而下降]

研究表明（Liu et al., "Lost in the Middle"），LLM 对位于上下文中间位置的信息
召回率远低于开头和结尾。这意味着：

- **不一定需要装满窗口**：大窗口的好处随利用率递减
- **关键信息放两端**：最重要的上下文放在 System Prompt（开头）或最后一条消息（结尾）
- **检索质量比检索数量重要**：RAG 返回 3 篇好文档优于 20 篇噪声文档

.. admonition:: 实践经验
   :class: tip

   在 Agent 生产中，建议：
   1. 设置软上限（例如 128K 窗口只用 80K），预留输出空间
   2. 关键指令放 System Prompt，工具调用结果放结尾
   3. 对话历史用摘要而非完整保留
   4. 工具返回结果仅保留关键字段，裁剪冗余信息
   5. 监控 token 消耗，在即将触及上限前主动做压缩

.. code-block:: python

   # token 预算监控器
   class TokenBudgetMonitor:
       def __init__(self, soft_limit=80000, hard_limit=128000):
           self.soft_limit = soft_limit
           self.hard_limit = hard_limit
           self.usage_log = []

       def check(self, current_tokens: int) -> str:
           ratio = current_tokens / self.hard_limit
           if ratio > 0.9:
               return "critical"   # 必须立即压缩
           elif ratio > 0.75:
               return "warning"    # 建议触发压缩
           elif ratio > 0.5:
               return "notice"     # 接近软上限
           return "ok"

       def estimate_tool_result(self, result: str) -> int:
           # 评估工具返回结果是否值得完整保留
           length = len(result)
           if length > 2000:
               return min(length, 500)  # 截断到 500 tokens
           return length
