.. _chapter-03-react:

===============================
ReAct 模式
===============================

ReAct（Reasoning + Acting）由 Yao et al. (2022) 提出，是 Agent 领域最具
影响力的决策模式之一。它让 LLM 在**推理**和**行动**之间交替进行，
每一步的推理指导下一步的行动，行动的结果又反馈给推理。

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
       def __init__(self, llm, tools: dict, max_steps=10):
           self.llm = llm
           self.tools = tools  # {"tool_name": callable}
           self.max_steps = max_steps

       def run(self, task: str) -> str:
           messages = [{"role": "user", "content": task}]

           for step in range(self.max_steps):
               # 思考：LLM 生成推理步骤
               thought = self.llm.generate(
                   self._build_react_prompt(messages)
               )
               messages.append({"role": "assistant", "content": thought})

               # 决定：是继续行动还是给出答案
               if self._is_final_answer(thought):
                   return self._extract_answer(thought)

               # 行动：解析并执行工具调用
               action = self._parse_action(thought)
               if action:
                   result = self._execute_tool(action)
                   messages.append({
                       "role": "tool",
                       "content": f"工具 {action['name']} 返回：{result}"
                   })
               else:
                   # 行动格式错误，提示修正
                   messages.append({
                       "role": "user",
                       "content": "请使用正确的行动格式：Action: tool_name\nAction Input: args"
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

       def _execute_tool(self, action: dict) -> str:
           tool = self.tools.get(action["name"])
           if not tool:
               return f"错误：未找到工具 '{action['name']}'"
           try:
               return tool(**action["args"])
           except Exception as e:
               return f"工具执行出错：{str(e)}"

ReAct 与 CoT 的对比
========================

.. list-table::
   :header-rows: 1

   * - 维度
     - CoT
     - ReAct
   * - 推理方式
     - 纯内部推理
     - 推理 + 外部信息交互
   * - 信息源
     - 仅依赖模型参数
     - 工具、数据库、互联网
   * - 幻觉风险
     - 高（无法验证事实）
     - 低（可交叉验证）
   * - 适用场景
     - 数学、逻辑推理
     - 需要外部信息的 Agent 任务
   * - 复杂任务处理
     - 可能偏离正确路径
     - 通过工具反馈纠正路径

在生产环境中，ReAct 和 CoT 通常**组合使用**——CoT 用于深入分析工具返回的结果，
ReAct 框架用于整体决策循环。

ReAct 模式的最佳实践
========================

.. admonition:: 行动格式标准化
   :class: tip

   使用固定的 JSON 格式或函数调用规范，减少 LLM 解析错误的概率：

   .. code-block:: python

       # 推荐的行动格式
       action_format = """
       {
           "thought": "分析当前状态",
           "action": "tool_name",
           "action_input": {"key": "value"}
       }
       """

       # 坏示例（自由格式）：
       # "我现在需要查一下天气，用天气工具查"
       # 好示例（结构化）：
       # {"thought": "需要查询北京天气", "action": "get_weather", "action_input": {"city": "北京"}}

ReAct 的错误恢复策略
========================

.. code-block:: python

   class RobustReActAgent(ReActAgent):
       def _execute_tool(self, action: dict) -> str:
           max_retries = 3
           for attempt in range(max_retries):
               try:
                   return super()._execute_tool(action)
               except TimeoutError:
                   if attempt < max_retries - 1:
                       continue  # 重试
                   return f"工具 {action['name']} 超时，已重试 {max_retries} 次"
               except RateLimitError:
                   import time
                   time.sleep(2 ** attempt)  # 指数退避
                   continue
               except Exception as e:
                   return f"工具 {action['name']} 执行失败：{str(e)}"
           return "工具执行失败，已耗尽所有重试次数"

参考文献
============

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", 2022
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", 2023
