.. _chapter-07-streaming:

===============================
流式响应
===============================

流式响应（Streaming）让 Agent 在生成过程中逐步输出结果，而不是让用户
等全部处理完了才看到结果。你可能觉得这只是一个"用户体验"问题，
但在 Agent 场景中，它关乎一个更深层的考量——**信任**。

想象一下这个场景：你问 Agent 一个复杂问题，它沉默了一分钟，然后突然
吐出一大段文字。你心里会想："它是真的在认真思考，还是卡住了？这个
答案可靠吗？"

如果 Agent 用流式输出逐步展示它的推理过程——"我正在搜索……"，
"我找到了 5 条结果，正在分析……"，"综合来看……"——用户就能看到
Agent 在"干活"。即使最终答案不完全准确，用户也更容易理解和接受。

这就是流式响应的本质价值：**让 Agent 的思考过程可见**。

Agent 流式的分层设计
========================

Agent 的流式输出和普通 LLM 流式输出不同。普通流式只输出文字 token，
Agent 流式需要输出多种类型的信息：

.. code-block:: python

   def run_agent_streaming(agent, task):
       """Agent 流式输出，展示完整的推理-行动过程"""
       full_response = ""

       for chunk in agent.run_stream(task):
           if chunk.type == "thought":
               yield f"💭 {chunk.content}\n"
           elif chunk.type == "action":
               yield f"🔧 调用工具: {chunk.name}({chunk.args})\n"
           elif chunk.type == "observation":
               yield f"📊 工具返回: {chunk.content[:100]}...\n"
           elif chunk.type == "answer":
               yield f"✅ {chunk.content}\n"
           full_response += chunk.content

   # 使用示例
   for token in run_agent_streaming(my_agent, "查一下今天的新闻"):
       print(token, end="", flush=True)

如果你用过 Cursor 的 Composer 模式或 Claude Code，会发现它们的流式输出
已经进化到远超上面这个例子的程度——它们甚至可以实时展示正在修改的文件、
正在运行的命令、正在生成的 diff。这就是流式在 Agent 场景中的高级形态。

实现细节：Event Stream
=========================

底层实现使用 Server-Sent Events（SSE），和 ChatGPT 的流式输出机制一样。

.. code-block:: python

   from flask import Response, stream_with_context

   @app.route("/agent/run", methods=["POST"])
   def agent_run_stream():
       task = request.json["task"]

       def generate():
           for chunk in agent.run_stream(task):
               yield f"data: {json.dumps(chunk.to_dict())}\n\n"

       return Response(
           stream_with_context(generate()),
           mimetype="text/event-stream"
       )

这里需要注意一个容易被忽略的细节：**SSE 的编码问题**。如果 Agent 返回
中文或其他非 ASCII 字符，确保 `json.dumps()` 设置了 `ensure_ascii=False`，
否则前端看到的是 `\u4f60\u597d` 而不是"你好"。

流式对 Token 消耗的影响
==========================

流式和不流式在 Token 消耗上是**完全相同**的——输出的内容一样多。
唯一的区别是输出时间分布。

.. list-table::
   :header-rows: 1

   * - 对比维度
     - 非流式
     - 流式
   * - 用户体验
     - 等待期无反馈，用户容易放弃
     - 逐步反馈，用户感知延迟更低
   * - Token 消耗
     - 完全一致
     - 完全一致
   * - 延迟感知
     - 用户感觉"慢"（实际延迟可能相同）
     - 用户感觉"快"（首 token 显示更快）
   * - 中断能力
     - 无法中断
     - 用户可随时中断长回复
     - 节省不必要的 Token 消耗
   * - 实现复杂度
     - 低（一行 return）
     - 中（需要 Event Stream 处理）

最后一行提到中断能力，这是流式的一个隐藏优势。如果 Agent 正在输出一长段
内容，用户发现方向不对，可以立刻点"停止"——后续的 token 就不再生成了。
非流式模式下，整个回答生成完了你才能看到它，想停都来不及。

.. admonition:: 流式的最佳实践
   :class: tip

   1. **TTFB 比总时间更重要**——用户不介意总时长 5 秒，但介意前 3 秒没反应
   2. **Agent 的思考步骤也应该流式输出**——用户想看"它在想什么"
   3. **前端做平滑的流式渲染**——不要一个 chunk 刷新一次 DOM，
      用缓冲区做 100ms 的节流
   4. **保留中断能力**——让用户可以随时停止 Agent 的执行
