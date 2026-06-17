.. _chapter-03-react:

===============================
ReAct 模式
===============================

ReAct（Reasoning + Acting）由 Yao et al. (2022) 提出，是 Agent 领域最具
影响力的决策模式之一。它让 LLM 在**推理**和**行动**之间交替进行，
每一步的推理指导下一步的行动，行动的结果又反馈给推理。

ReAct 可以看作 Agent 的"最小可行架构"——你只需要一个 LLM、一组工具、
一个循环，就能让模型从"只能聊天"变成"能做事情"。几乎所有现代 Agent
框架（LangChain、AutoGPT、OpenClaw）的核心循环都源自 ReAct。

ReAct 的核心循环
========================

.. mermaid::

   flowchart TD
       Start[收到任务] --> Think[思考：当前进度分析]
       Think --> Decide{下一步做什么？}
       Decide --> Act[行动：调用工具获取信息]
       Act --> Observe[观察：工具返回结果]
       Observe --> Check{任务完成？}
       Check -- 否 --> Think
       Check -- 是 --> Answer[输出最终答案]

.. code-block:: python

   class ReActAgent:
       """
       ReAct 的核心实现。每次循环做三件事：
       1. 思考（Thought）：分析当前状态，决定下一步
       2. 行动（Action）：调用工具获取信息
       3. 观察（Observation）：接收工具返回结果
       """
       def __init__(self, llm, tools: dict, max_steps=10):
           self.llm = llm
           self.tools = tools
           self.max_steps = max_steps

       def run(self, task: str) -> str:
           messages = [{"role": "user", "content": task}]

           for step in range(self.max_steps):
               thought = self.llm.generate(
                   self._build_react_prompt(messages)
               )
               messages.append({"role": "assistant", "content": thought})

               if self._is_final_answer(thought):
                   return self._extract_answer(thought)

               action = self._parse_action(thought)
               if action:
                   result = self._execute_tool(action)
                   messages.append({
                       "role": "tool",
                       "content": f"工具 {action['name']} 返回：{result}"
                   })
               else:
                   messages.append({
                       "role": "user",
                       "content": "请使用正确的行动格式。"
                   })

           return "已达最大步数，无法完成任务"

       def _build_react_prompt(self, messages) -> str:
           prompt = """请遵循以下格式交替思考和行动：

   思考：分析当前状态，决定下一步做什么
   行动：要调用的工具名称
   行动输入：工具的参数
   观察：工具返回的结果
   ...（重复思考和行动）
   最终答案：给出最终回答

   """
           for msg in messages:
               prompt += f"{msg['role']}: {msg['content']}\n"
           return prompt

为什么 ReAct 比纯推理更好？
============================

ReAct 的核心洞见是：**推理和行动互相增强**。这听起来像哲学，但实际效果
可以量化。

.. list-table::
   :header-rows: 1

   * - 场景
     - 纯 LLM 推理（CoT）
     - ReAct
     - 提升
   * - 事实问答
     - 准确率 70-80%（依赖训练数据）
     - 准确率 85-95%（通过工具验证）
     - +15%
   * - 数学计算
     - 准确率 60-70%
     - 准确率 90%+（用计算器工具）
     - +25%
   * - 实时信息
     - 无法获取（知识截止）
     - 可以获取（搜索工具）
     - 质变
   * - 错误纠正
     - 一步错步步错
     - 工具返回可纠正错误
     - 质变

ReAct 的"思考"格式设计
=========================

ReAct 的思考格式直接影响 Agent 的行为质量。下面两种写法的效果差异很大。

.. code-block:: python

   # 差：思考格式太模糊
   prompt = "思考并行动。"

   # 好：思考格式结构化，引导模型做正确的事
   prompt = """
   每次回复必须包含以下格式：

   思考：分析当前进度。我已经知道了什么？还需要什么？下一步最合理的操作是什么？

   行动：如果还需要更多信息，调用工具。格式：[工具名称(参数)]
   或
   最终答案：如果已有足够信息，直接给出答案。
   """

结构化思考格式的好处是：
1. 减少"幻觉工具调用"——模型不会凭空编造工具返回结果
2. 增加推理透明度——可以追踪模型在每个 step 的推理过程
3. 提升准确率——实验显示结构化思考格式比自由格式准确率高 8-12%

行动格式的标准化
====================

ReAct 中一个常见的工程问题是 LLM 输出的行动格式不符合预期。
标准化的行动格式可以大幅减少解析错误。

.. code-block:: python

   # 推荐的行动格式（JSON）
   ACTION_FORMAT = """
   {
       "thought": "分析当前状态、下一步的理由",
       "action": "工具名称或 null（如果直接回答）",
       "action_input": {"参数名": "参数值"},
       "answer": "最终答案或 null"
   }
   """

   # 坏示例（自由格式，难解析）：
   # "我现在需要查一下天气，用 get_weather 工具查北京天气"
   #
   # 好示例（结构化，易解析）：
   # {
   #   "thought": "需要查询北京天气",
   #   "action": "get_weather",
   #   "action_input": {"city": "北京"},
   #   "answer": null
   # }

   def parse_action(response: str) -> dict:
       """从 LLM 输出中解析行动指令"""
       import json
       # 尝试直接解析 JSON
       try:
           return json.loads(response)
       except json.JSONDecodeError:
           pass
       # 尝试从文本中提取 JSON 块
       import re
       match = re.search(r'\{.*\}', response, re.DOTALL)
       if match:
           return json.loads(match.group())
       return None

ReAct 的局限与改进
=====================

ReAct 不是银弹，它有几个明显的局限。

.. admonition:: 局限 1：循环深度有限
   :class: caution

   ReAct 每步的对话历史都在增长。10 步之后，上下文可能已经包含了
   上万 token 的历史。超过一定步数后，模型在"思考"阶段的表现会
   显著下降（注意力稀释）。解决方案：上下文压缩或 ReAct + 摘要。

.. admonition:: 局限 2：没有全局规划
   :class: caution

   ReAct 是"走一步看一步"。对需要全局规划的任务（如"查资料然后写一份报告"），
   ReAct 可能在前几步就偏离了方向。解决方案：先用 Planner 做规划，
   然后每步用 ReAct 执行。

.. admonition:: 局限 3：工具调用开销
   :class: caution

   每次 Action 都是一次工具调用。如果工具本身很慢（如搜索引擎 1-3 秒），
   多次串行调用会让用户体验很差。解决方案：并行工具调用或缓存。

ReAct 的三种变体
====================

1. Reflexion（2023，Shinn et al.）
----------------------------------

Reflexion 在 ReAct 的基础上增加了"自我反思"环节。当工具调用失败时，
Agent 不仅重试，还分析失败原因，把分析结果加入后续决策。

.. code-block:: python

   class ReflexionAgent(ReActAgent):
       """带自我反思的 ReAct Agent"""
       def run(self, task: str) -> str:
           memory = []  # 存储反思结果

           for attempt in range(3):  # 最多 3 次尝试
               result = super().run(task)
               if "错误" not in result and "失败" not in result:
                   return result

               # 反思：分析失败原因
               reflection = self.llm.generate(f"""
                   任务：{task}
                   执行结果：{result}

                   分析失败原因，并给出改进方案。
               """)
               memory.append(reflection)

2. ReAct + CoT 混合
----------------------------------

在每一步的"思考"阶段，先用 CoT 深入分析，再决定行动。

.. code-block:: python

   def react_with_cot_thought(step_context: str) -> dict:
       """在 ReAct 的思考阶段嵌入 CoT"""
       chain_of_thought = llm.generate(f"""
           当前任务进度：{step_context}

           请分步骤分析：
           1. 当前已经知道哪些信息？
           2. 还需要哪些信息才能完成任务？
           3. 哪个工具能提供这些信息？
           4. 调用这个工具时需要注意什么？

           基于以上分析，决定下一步行动。
       """)
       return parse_action(chain_of_thought)

3. ReAct with Tool Result Summarization
------------------------------------------

工具返回的结果很长时，让 LLM 先摘要再存入上下文。

.. code-block:: python

   def execute_and_summarize(tool, action):
       """执行工具调用并对结果做摘要"""
       raw_result = tool(**action["args"])
       # 如果结果太长，做摘要
       if len(raw_result) > 500:
           summary = llm.generate(
               f"将以下内容摘要为 50 字以内：{raw_result[:2000]}"
           )
           return f"[摘要] {summary}"
       return raw_result

参考文献
============

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", 2022
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", 2023
