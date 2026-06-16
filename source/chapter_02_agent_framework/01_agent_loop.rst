.. _chapter-02-agent-loop:

===============================
Agent 主循环
===============================

Agent 的核心是一个感知-推理-行动的循环。每次迭代，Agent 观察外部反馈，
进行内部推理，决定下一步行动，然后等待行动结果。

.. code-block:: python

   class Agent:
       def __init__(self, llm, tools, memory):
           self.llm = llm          # 底层大语言模型
           self.tools = tools      # 可用工具集合
           self.memory = memory    # 记忆系统

       def run(self, task: str) -> str:
           self.memory.add("user", task)
           max_steps = 10

           for step in range(max_steps):
               # 1. 推理：LLM 决定下一步
               response = self.llm.generate(self.memory.get_context())

               # 2. 解析：从输出中提取行动指令
               action = self.parse_action(response)

               # 3. 执行：调用工具或返回答案
               if action.type == "final_answer":
                   return action.content
               else:
                   result = self.tools.execute(action)
                   self.memory.add("observation", result)

           return "已达到最大步数限制"

.. admonition:: ReAct：推理与行动的同步
   :class: story

   2022 年 Google 提出的 ReAct 模式（Reasoning + Acting）是 Agent 循环的经典范式。
   它的核心洞察是：**让 LLM 生成思考过程后再决定行动，比直接输出行动更准确**。
   这是因为思考过程帮助 LLM 将长期目标分解为短期步骤，减少了"遗忘初始指令"的问题。
   今天大多数 Agent 框架（LangChain、AutoGPT 等）都采用了 ReAct 或其变体。
