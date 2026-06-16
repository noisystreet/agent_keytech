.. _chapter-02-tool-use:

===============================
工具调用
===============================

工具调用（Tool Use / Function Calling）是 Agent 与外部世界交互的唯一接口。
GPT-4、Claude 3、Llama 3 等主流模型都原生支持函数调用能力。

函数调用协议
================

.. code-block:: python

   # OpenAI 风格的函数调用
   tools = [
       {
           "type": "function",
           "function": {
               "name": "search",
               "description": "搜索互联网获取信息",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "query": {"type": "string", "description": "搜索关键词"}
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

**MCP 架构：**

.. code-block:: python

   # MCP 服务器：提供工具和服务
   # 使用官方 Python SDK

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

   if __name__ == "__main__":
       stdio_server.run(app)

.. code-block:: python

   # MCP 客户端：Agent 通过 MCP 发现和调用工具

   from mcp import Client

   client = Client("weather-server")

   # 自动发现工具
   tools = client.list_tools()
   # => [Tool(name="get_weather", description="查询天气", ...)]

   # 调用工具
   result = client.call_tool("get_weather", {"city": "北京"})

.. admonition:: MCP vs Function Calling
   :class: tip

   - **Function Calling** 是模型与开发者之间的约定——你定义函数，模型选择调用
   - **MCP** 是 Agent 与工具之间的网络协议——工具以服务形式注册，Agent 动态发现

   两者不是替代关系，而是互补：Function Calling 解决"模型怎么理解工具"，
   MCP 解决"工具怎么被 Agent 发现和连接"。实践中可以同时使用。
