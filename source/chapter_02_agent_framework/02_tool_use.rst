.. _chapter-02-tool-use:

===============================
工具调用
===============================

工具调用（Tool Use / Function Calling）是 Agent 与外部世界交互的唯一接口。
没有工具，Agent 只是一个会说话的模型——能聊天、能推理，但什么都做不了。
有了工具，Agent 才能查天气、发邮件、操作数据库、调用 API。

工具调用的核心流程很简单：开发者定义工具的 Schema，LLM 在推理时决定
是否调用以及传入什么参数，然后 Agent 框架执行工具调用并把结果返回给 LLM。

但"简单"的背后有很多工程细节。工具定义的描述写得好不好，直接决定 LLM
能不能在正确的时候调用正确的工具。工具调用的错误处理做得好不好，
直接决定 Agent 在多步任务中能不能稳定执行。

函数调用协议
================

OpenAI 风格的函数调用是事实标准。几乎所有主流模型都支持这种格式。

.. code-block:: python

   # OpenAI 风格的函数调用定义
   tools = [
       {
           "type": "function",
           "function": {
               "name": "search",
               "description": "搜索互联网获取信息，适合查找实时信息和最新动态",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "query": {
                           "type": "string",
                           "description": "搜索关键词，建议使用精确的关键词而非自然语言"
                       }
                   },
                   "required": ["query"]
               }
           }
       }
   ]

   # LLM 返回的调用请求（自动触发，无需额外提示词）
   response = llm.chat(messages, tools=tools)
   # response.tool_calls => [{"name": "search", "args": {"query": "..."}}]

.. note::

   主流模型均支持 Function Calling，包括 GPT-4o、Claude 3.5 Sonnet、Gemini 2.0、
   Qwen 2.5、DeepSeek V3 等。各家的参数格式略有差异，但核心模式一致。

工具描述的最佳实践
=====================

工具描述（description）的质量直接影响 LLM 选择工具的准确率。以下是几个关键点：

**描述要写"什么时候用"，而不是"是什么"。**

.. code-block:: python

   # 差：
   "description": "搜索工具，可以进行搜索"
   # 模型困惑：那我什么时候用搜索？

   # 好：
   "description": "搜索互联网获取最新信息，适合查询实时新闻、产品信息、事实类问题"
   # 模型清楚：需要实时信息时用这个

**参数描述要写明格式要求。**

.. code-block:: python

   # 差：
   "query": {"type": "string", "description": "搜索关键词"}

   # 好：
   "query": {"type": "string",
             "description": "搜索关键词。使用精确词而非自然语言。"
                            "如搜索'北京天气'而非'帮我查一下北京今天天气怎么样'"}

**一个工具只做一件事。**

如果有工具"search_and_calculate"，LLM 很难判断什么时候调用它。
一个工具一个函数，职责清晰，模型的选择准确率更高。

.. admonition:: 描述对准确率的影响
   :class: tip

   一个实验数据：当工具描述是 "搜索互联网" 时，准确调用率约 70%。
   当描述改为 "搜索互联网获取最新信息，适合查询实时新闻、产品价格、事实类问题"
   时，准确调用率提升到 85% 以上。好的描述值 15% 的准确率。

Tool Choice 控制
====================

OpenAI 的 API 提供了 `tool_choice` 参数来控制 LLM 调用工具的行为。

.. code-block:: python

   # 不强制调用工具（默认）：模型可以自己决定是否调用
   response = llm.chat(messages, tools=tools, tool_choice="auto")

   # 强制调用指定工具：模型必须调用 search 工具
   response = llm.chat(messages, tools=tools,
                       tool_choice={"type": "function", "function": {"name": "search"}})

   # 强制必须调用某个工具（不让模型直接回答）：适合每一步都必须用工具的场景
   response = llm.chat(messages, tools=tools, tool_choice="required")

   # 禁止调用工具：让模型直接回答
   response = llm.chat(messages, tools=tools, tool_choice="none")

强制调用在以下场景特别有用：
- **多步推理的第一步**：强制让 Agent 先搜索再回答
- **固定工作流**：强制 Agent 按预定步骤执行
- **Agent 的 reasoning 模式**：禁止工具调用时，Agent 只能纯推理

错误处理模式
================

工具调用可能因为各种原因失败：网络超时、API 返回错误、参数格式不对。
Agent 必须正确处理这些错误，否则多步推理在第三步崩了，前面的工作就白费了。

.. code-block:: python

   class RobustToolExecutor:
       """带错误处理的工具执行器"""
       def __init__(self, tools: dict, max_retries=2, timeout=10):
           self.tools = tools
           self.max_retries = max_retries
           self.timeout = timeout

       def execute(self, tool_name: str, arguments: dict) -> dict:
           """执行工具调用，含重试和超时逻辑"""
           if tool_name not in self.tools:
               return {"error": f"工具 {tool_name} 不存在", "success": False}

           for attempt in range(self.max_retries + 1):
               try:
                   result = self._call_with_timeout(tool_name, arguments)
                   return {"result": result, "success": True}

               except TimeoutError:
                   if attempt < self.max_retries:
                       continue
                   return {"error": f"工具 {tool_name} 超时", "success": False}
               except RateLimitError:
                   import time
                   time.sleep(2 ** attempt)  # 指数退避
                   continue
               except Exception as e:
                   return {"error": f"工具 {tool_name} 执行失败: {str(e)}",
                           "success": False}

       def _call_with_timeout(self, name, args):
           """带超时的工具调用"""
           import signal
           result = [None]

           def handler(signum, frame):
               raise TimeoutError()

           signal.signal(signal.SIGALRM, handler)
           signal.alarm(self.timeout)
           try:
               result[0] = self.tools[name](**args)
           finally:
               signal.alarm(0)
           return result[0]

工具调用的工程注意事项
========================

.. admonition:: 工具返回结果过长怎么办？
   :class: caution

   很多工具返回的结果很长（比如搜索返回 50 条结果），这些结果会全部
   塞进上下文中，迅速消耗 token 预算。

   策略：截断（只保留前 3 条）+ 摘要（让 LLM 压缩长结果）。在 Agent
   框架中，工具返回结果传给 LLM 前应该经过一层"结果处理器"。

.. code-block:: python

   def truncate_tool_result(result: str, max_tokens=500) -> str:
       """截断工具返回结果，控制在 max_tokens 以内"""
       tokens = tokenizer.encode(result)
       if len(tokens) <= max_tokens:
           return result
       # 保留开头和结尾，中间省略
       head = tokenizer.decode(tokens[:max_tokens // 2])
       tail = tokenizer.decode(tokens[-max_tokens // 2:])
       return f"{head}\n...（省略 {len(tokens) - max_tokens} tokens）...\n{tail}"

.. admonition:: 工具调用结果验证
   :class: tip

   工具返回后，不要直接喂给 LLM。先验证结果的格式和完整性。
   比如搜索工具返回空结果时，Agent 应该知道"没找到"而不是强行编造。

   .. code-block:: python

       def validate_and_format(tool_name, result):
           if result is None or result == "":
               return f"[工具 {tool_name} 返回空结果]"
           if isinstance(result, list) and len(result) == 0:
               return f"[工具 {tool_name} 未找到匹配内容]"
           return str(result)

Model Context Protocol (MCP)
=============================

MCP（Model Context Protocol）是 Anthropic 于 2024 年底推出的开放协议，
旨在为 LLM 应用提供标准化的工具和数据源接入方式。可以将其理解为 **"AI 应用的 USB-C 接口"**。

**为什么需要 MCP？**

在 MCP 之前，每个 Agent 框架都有自己的工具定义规范，工具开发者需要为不同
框架分别适配。MCP 统一了这一标准：

.. mermaid::

   flowchart LR
       subgraph Before [MCP 之前]
           A1[Agent A] --- Custom1[自定义工具]
           A2[Agent B] --- Custom2[自定义工具]
           A3[Agent C] --- Custom3[自定义工具]
       end

       subgraph After [MCP 之后]
           B1[Agent A] --- MCP[MCP 协议]
           B2[Agent B] --- MCP
           B3[Agent C] --- MCP
           MCP --- S1[搜索工具]
           MCP --- S2[数据库工具]
           MCP --- S3[文件工具]
       end

.. code-block:: python

   # MCP 服务器：提供工具和服务
   from mcp.server import Server, stdio_server
   from mcp.types import Tool, TextContent

   app = Server("weather-agent")

   @app.list_tools()
   async def list_tools():
       return [
           Tool(
               name="get_weather",
               description="查询指定城市的天气",
               input_schema={
                   "type": "object",
                   "properties": {
                       "city": {"type": "string", "description": "城市名称"}
                   },
                   "required": ["city"]
               }
           )
       ]

   @app.call_tool()
   async def call_tool(name: str, arguments: dict):
       if name == "get_weather":
           result = f"{arguments['city']} 的天气：晴，25°C"
           return [TextContent(type="text", text=result)]

MCP 的核心理念是**工具即服务**——工具不需要嵌入 Agent 框架中，
而是作为独立的服务运行，Agent 通过网络协议发现和调用它们。
这对于大型组织的工具治理特别重要：你可以让安全团队审批 MCP 服务器，
而不是逐个审核每个 Agent 的工具配置。

.. admonition:: MCP vs Function Calling
   :class: tip

   - **Function Calling** 是模型与开发者之间的约定——你定义函数，模型选择调用
   - **MCP** 是 Agent 与工具之间的网络协议——工具以服务形式注册，Agent 动态发现

   两者不是替代关系，而是互补：Function Calling 解决"模型怎么理解工具"，
   MCP 解决"工具怎么被 Agent 发现和连接"。实践中可以同时使用。
