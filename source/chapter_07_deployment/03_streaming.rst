.. _chapter-07-streaming:

===============================
流式响应
===============================

流式响应（Streaming）让 Agent 在生成过程中逐步输出结果，显著改善用户体验。

.. code-block:: python

   def run_agent_streaming(agent, task):
       """Agent 的流式输出"""
       full_response = ""

       for chunk in agent.run_stream(task):
           if chunk.type == "thought":
               yield f"🤔 {chunk.content}\n"
           elif chunk.type == "action":
               yield f"🛠️  调用工具: {chunk.name}({chunk.args})\n"
           elif chunk.type == "observation":
               yield f"📊 工具返回: {chunk.content[:100]}...\n"
           elif chunk.type == "answer":
               yield f"✅ 答案: {chunk.content}\n"
           full_response += chunk.content

   # 使用示例
   for token in run_agent_streaming(my_agent, "查一下今天的新闻"):
       print(token, end="", flush=True)
