.. _chapter-02-memory:

===============================
记忆系统
===============================

记忆是 Agent 区别于无状态 API 调用的关键特性。一个没有记忆的 Agent，
每次对话都是从零开始——它不认识你，不记得刚才说过什么，不记得三分钟前
工具返回了什么结果。

在日常使用中，你对 Agent 的"智能感"很大一部分来自它的记忆能力。
如果 Agent 记得你上周问过什么，记得你偏好的回答风格，记得哪些信息已经
确认过不需要再查——这些远比模型推理能力的提升更能改善用户体验。

Agent 开发者面对的挑战是：**LLM 的上下文窗口不是为"记忆"设计的**。
它是为"一次性处理"设计的。你需要构建专门的记忆系统，让 Agent 在
跨越会话、跨越工具调用边界时仍然能"记住"。

三层记忆架构
================

Agent 的记忆系统通常分为三个层次，对应不同的人类记忆机制：

.. list-table::
   :header-rows: 1

   * - 记忆层次
     - 类比人类
     - 实现方式
     - 容量
     - 持久性
   * - 短期记忆
     - 工作记忆（你在脑中默念的电话号码）
     - 上下文窗口（in-context）
     - 有限（~128k token）
     - 对话结束即消失
   * - 长期记忆
     - 记忆存储（你记得的童年往事）
     - 向量数据库 + RAG
     - 理论上无限
     - 跨会话持久
   * - 程序记忆
     - 本能反射（你会骑自行车但不记得怎么学会的）
     - 微调后的模型权重
     - 永久
     - 不随对话变化

了解这三种类型的差异，才能判断 Agent 的"记忆失灵"是哪种类型的问题。
比如，"Agent 不记得前三步推理的中间结果"——这是短期记忆管理的问题，
需要在上下文窗口中做压缩。"Agent 不记得上个对话月跟你聊过什么"——
这是长期记忆的问题，需要做 RAG 检索。

1. 短期记忆（Short-term Memory）
--------------------------------------

短期记忆实现最简单，就是 LLM 的上下文窗口本身。但"最简单"不意味着
"不需要管理"。如果不做主动管理，短期记忆会被无限制地膨胀填满。

.. code-block:: python

   class ShortTermMemory:
       """
       短期记忆：管理当前对话的上下文窗口。
       主要任务是在有限窗口内"装下"最重要的信息。
       """
       def __init__(self, max_tokens=32000, llm=None):
           self.history = []
           self.max_tokens = max_tokens
           self.llm = llm

       def add(self, message: dict):
           self.history.append(message)
           self._trim()

       def _trim(self):
           """上下文超限时做压缩"""
           total = self._count_tokens()
           if total <= self.max_tokens:
               return

           # 策略 1：丢弃最早的对话轮次（滑动窗口）
           while self._count_tokens() > self.max_tokens * 0.7:
               if len(self.history) <= 2:
                   break  # 至少保留当前轮次
               # 找到最早的用户-助手消息对并丢弃
               for i in range(len(self.history)):
                   if self.history[i].get("role") in ("user", "assistant"):
                       self.history.pop(i)
                       break

       def summarize_and_compress(self):
           """
           策略 2：用 LLM 生成摘要压缩早期对话。
           比滑动窗口更有信息保留，但多一次 LLM 调用。
           """
           if len(self.history) <= 4:
               return  # 太短不需要压缩

           # 保留最近的 2 轮
           keep = self.history[-4:]
           # 对前面的历史做摘要
           early = self.history[:-4]
           summary = self.llm.generate(
               f"将以下对话压缩为 100 字以内的摘要，保留关键事实和决策：\n{early}"
           )
           self.history = [
               {"role": "system", "content": f"[对话摘要] {summary}"}
           ] + keep

短期记忆管理的核心矛盾是：**你想保留更多信息以便 Agent 理解上下文，
但每多保留一轮对话，就消耗了工具返回结果和推理步骤的预算。**
经验值是工具调用结果 > 用户指令 > 历史对话——在裁剪时，优先保留前两者。

2. 长期记忆（Long-term Memory）
--------------------------------------

长期记忆将重要信息持久化存储，跨越不同的对话会话。实现的核心是
"写入→存储→检索→注入"的循环。

.. code-block:: python

   class LongTermMemory:
       """
       长期记忆：跨会话持久化记忆。
       使用向量数据库存储，通过 RAG 检索注入。

       关键设计选择：
       - 什么是"值得记住的"？
       - 检索时如何避免召回率不足？
       - 记忆存储的增长如何控制？
       """
       def __init__(self, llm, vector_db):
           self.llm = llm
           self.vector_db = vector_db  # 向量数据库

       def remember(self, conversation: list):
           """
           判断当前对话中哪些信息值得存入长期记忆。
           """
           # 1. 让 LLM 判断哪些信息重要
           important_facts = self.llm.generate(f"""
               分析以下对话，提取 3-5 条值得长期记住的信息
               （用户偏好、关键决策、重要事实）：

               {conversation}

               每条信息用一句话描述：
           """)

           # 2. 信息去重
           facts = self._deduplicate(important_facts)

           # 3. 存入向量数据库
           for fact in facts:
               embedding = self.embedder.embed(fact)
               self.vector_db.add(
                   text=fact,
                   embedding=embedding,
                   metadata={"timestamp": time.time()}
               )

       def recall(self, query: str, k: int = 5) -> str:
           """
           检索与当前问题相关的长期记忆。
           """
           memories = self.vector_db.similarity_search(query, k=k)
           if not memories:
               return ""

           return "\n".join([
               f"[记忆] {mem['text']}"
               for mem in memories
           ])

       def _deduplicate(self, facts_text: str) -> list:
           """
           语义去重：防止存储重复或矛盾的信息。
           """
           facts = facts_text.strip().split("\n")
           # 简化实现：只去除完全相同的
           return list(set(facts))

长期记忆的一个容易被忽略的问题：**检索的 query 怎么写？**

.. code-block:: python

   # 不要用用户的原话作为检索 query
   # 坏示例：
   query = "帮我查一下"
   # 这个 query 检索不到任何有用的长期记忆

   # 好示例：用 LLM 生成一个"检索意图"
   def generate_retrieval_query(user_input: str) -> str:
       """
       将用户输入转化为检索 query。
       目标：提取用户真正在问的实体和意图。
       """
       return llm.generate(
           f"用户说：{user_input}\n"
           f"请生成一个检索查询，用于在记忆库中找到相关信息："
       )

3. 程序记忆（Procedural Memory）
--------------------------------------

程序记忆通过微调（Fine-tuning）将行为模式固化到模型权重中。
一旦 Agent 学会了如何调用某个复杂的工具链，就不需要每次都在 System Prompt
中重新描述——这已经变成了它的"肌肉记忆"。

.. code-block:: python

   # 程序记忆的例子：
   # 微调前，你需要在 System Prompt 中写：
   SYSTEM_PROMPT = """
   当你需要对比两个产品时：
   1. 用 search 查找产品 A 的信息
   2. 用 search 查找产品 B 的信息
   3. 用 search 查找对比文章
   4. 综合结果

   每次都必须按这个顺序执行。不要跳步。
   """

   # 微调后，这些规则已经"内化"到模型权重中
   # 你只需要说：
   USER = "帮我对比产品 A 和产品 B"
   # Agent 就会自动按正确步骤执行

记忆系统的常见陷阱
======================

.. admonition:: 陷阱 1：长期记忆检索不到相关信息
   :class: caution

   最常见的问题。原因通常是：**存储的信息和检索的 query 不在同一语义空间。**
   比如存的时候是"用户喜欢 Python"，查的时候是"用户偏好的编程语言"。
   解决方案是存储时加多种表达方式的关键词，或使用混合搜索。

.. admonition:: 陷阱 2：短期记忆截断切掉了关键信息
   :class: caution

   滑动窗口策略的天然缺陷：最重要的信息可能是在第 3 轮出现的，但 "窗口"
   只保留了最近的 5 轮。解决方案：**优先级标记**——让 Agent 在对话中明确
   标记哪些信息是重要的（如 "记住：用户的项目截止日期是周五"）。
   系统在截断时优先保留标记过的信息。

.. admonition:: 陷阱 3：长期记忆污染
   :class: caution

   长期记忆存储了用户的错误假设或临时偏好。比如用户说"我不喜欢 Python"，
   但第二天又用了一天 Python。解决方案：**记忆的"软过期"机制**——
   给每条记忆加上时间戳和置信度分数，一段时间未被访问的记忆自动降权。

.. code-block:: python

   class MemoryWithConfidence:
       """带置信度和过期机制的长期记忆"""
       def __init__(self):
           self.memories = []  # [(text, confidence, timestamp)]

       def add(self, text: str, confidence: float = 0.5):
           self.memories.append((text, confidence, time.time()))

       def recall(self, query: str, k: int = 5) -> list:
           # 检索时考虑：语义相似度 × 置信度 × 时间衰减
           now = time.time()
           scored = []
           for text, conf, ts in self.memories:
               sim = semantic_similarity(query, text)
               time_decay = exp(-(now - ts) / (7 * 86400))  # 7 天半衰期
               score = sim * conf * time_decay
               scored.append((text, score))

           scored.sort(key=lambda x: x[1], reverse=True)
           return scored[:k]

      def reinforce(self, text: str):
          """当某条记忆被成功使用后，增加其置信度"""
          for i, (t, conf, ts) in enumerate(self.memories):
              if semantic_similarity(t, text) > 0.9:
                  self.memories[i] = (t, min(1.0, conf + 0.1), time.time())
                  break

三种记忆的协同工作
======================

.. code-block:: python

   class AgentMemorySystem:
       """
       完整的 Agent 记忆系统：三层记忆协同工作。
       """
       def __init__(self, llm, vector_db):
           self.short_term = ShortTermMemory(llm=llm)
           self.long_term = LongTermMemory(llm=llm, vector_db=vector_db)
           self.llm = llm

       def get_context(self, user_input: str) -> str:
           # 1. 从长期记忆中检索相关信息
           memories = self.long_term.recall(user_input)

           # 2. 从短期记忆中获取当前对话历史
           history = self.short_term.get_recent()

           # 3. 组合上下文
           context = ""
           if memories:
               context += f"## 关于你的信息\n{memories}\n\n"
           context += f"## 当前对话\n{history}"

           return context

       def after_response(self, user_input: str, agent_response: str):
           """每轮对话结束后，更新记忆系统"""
           # 更新短期记忆
           self.short_term.add({"role": "user", "content": user_input})
           self.short_term.add({"role": "assistant", "content": agent_response})

           # 判断是否需要存储长期记忆
           conversation = f"用户：{user_input}\n助手：{agent_response}"
           self.long_term.remember(conversation)
