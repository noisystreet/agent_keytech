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

如果 Agent 用流式输出逐步展示它的推理过程——"正在搜索……"，
"找到了 5 条结果，正在分析……"，"综合来看……"——用户就能看到
Agent 在"干活"。即使最终答案不完全准确，用户也更容易理解和接受。

这就是流式响应的本质价值：**让 Agent 的思考过程可见**。

Agent 流式的分层设计
========================

Agent 的流式输出和普通 LLM 流式输出不同。普通流式只输出文字 token，
Agent 流式需要输出多种类型的信息，每种类型在前端需要不同的展示方式。

.. code-block:: python

   # Agent 流式输出的消息类型
   class AgentStreamChunk:
       """Agent 流式输出的一个数据块"""
       def __init__(self, chunk_type, content, metadata=None):
           self.type = chunk_type  # "thought" / "action" / "observation" / "answer" / "error"
           self.content = content
           self.metadata = metadata or {}

       def to_dict(self):
           return {
               "type": self.type,
               "content": self.content,
               "metadata": self.metadata,
           }

   def run_agent_streaming(agent, task):
       """Agent 流式输出，展示完整的推理-行动过程"""
       for chunk in agent.run_stream(task):
           if chunk.type == "thought":
               yield f"💭 {chunk.content}\n"
           elif chunk.type == "action":
               yield f"🔧 调用工具: {chunk.name}({chunk.args})\n"
           elif chunk.type == "observation":
               yield f"📊 工具返回: {chunk.content[:100]}...\n"
           elif chunk.type == "answer":
               yield f"✅ {chunk.content}\n"

如果你用过 Cursor 的 Composer 模式或 Claude Code，会发现它们的流式输出
已经进化到远超上面这个例子的程度——它们可以实时展示正在修改的文件、
正在运行的命令、正在生成的 diff。这就是流式在 Agent 场景中的高级形态。

前端展示策略
================

不同类型的流式数据块在前端应该有不同的展示方式。

.. code-block:: javascript

   // 前端 Agent 流式渲染（JavaScript 示例）
   function renderAgentChunk(chunk) {
       switch (chunk.type) {
           case "thought":
               // 思考过程用灰色斜体，不显眼
               appendToLog(`<div class="thought">🤔 ${chunk.content}</div>`);
               break;
           case "action":
               // 工具调用用高亮色，突出正在做什么
               appendToLog(`<div class="action">🔧 ${chunk.content}</div>`);
               break;
           case "observation":
               // 工具返回结果用等宽字体，展示原始数据
               appendToLog(`<pre class="observation">${chunk.content}</pre>`);
               break;
           case "answer":
               // 最终答案用大字号，主色
               appendToLog(`<div class="answer">✅ ${chunk.content}</div>`);
               break;
       }
       // 自动滚动到底部
       scrollToBottom();
   }

.. code-block:: python

   # 流式渲染的缓冲区策略
   class StreamBuffer:
       """
       前端渲染缓冲区。不要每个 chunk 都刷新 DOM，而是攒够一定量再刷新。
       这样可以避免频繁 DOM 操作导致的页面卡顿。
       """
       def __init__(self, flush_interval=0.1):
           self.buffer = []
           self.flush_interval = flush_interval  # 100ms
           self.last_flush = time.time()

       def add(self, chunk):
           self.buffer.append(chunk)
           if time.time() - self.last_flush > self.flush_interval:
               self.flush()

       def flush(self):
           if not self.buffer:
               return
           # 把缓冲区内容发给前端
           send_to_frontend(self.buffer)
           self.buffer = []
           self.last_flush = time.time()

实现细节：Event Stream
=========================

底层实现使用 Server-Sent Events（SSE）。它比 WebSocket 更简单——SSE 是
单向的（服务器推送到客户端），不需要握手协议。

.. code-block:: python

   from flask import Response, stream_with_context
   import json

   @app.route("/agent/run", methods=["POST"])
   def agent_run_stream():
       task = request.json["task"]

       def generate():
           for chunk in agent.run_stream(task):
               # SSE 格式：data: {json}\n\n
               yield f"data: {json.dumps(chunk.to_dict(), ensure_ascii=False)}\n\n"

       return Response(
           stream_with_context(generate()),
           mimetype="text/event-stream",
           headers={
               "Cache-Control": "no-cache",
               "Connection": "keep-alive",
               "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
           }
       )

需要注意两个容易被忽略的细节：

1. **中文编码**：`json.dumps()` 必须加 `ensure_ascii=False`，否则中文被转义
2. **代理缓冲**：Nginx 默认会缓冲 SSE 响应，需要用 `X-Accel-Buffering: no` 禁用

.. code-block:: python

   # 前端消费 SSE 的标准方式
   async function runAgent(task) {
       const response = await fetch("/agent/run", {
           method: "POST",
           headers: { "Content-Type": "application/json" },
           body: JSON.stringify({ task }),
       });

       const reader = response.body.getReader();
       const decoder = new TextDecoder();

       while (true) {
           const { done, value } = await reader.read();
           if (done) break;

           const text = decoder.decode(value);
           // SSE 数据以 "data: " 开头
           const lines = text.split("\n");
           for (const line of lines) {
               if (line.startsWith("data: ")) {
                   const chunk = JSON.parse(line.slice(6));
                   renderAgentChunk(chunk);
               }
           }
       }
   }

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
     - 用户可随时中断长回复，节省不必要的 Token 消耗
   * - 实现复杂度
     - 低（一行 return）
     - 中（需要 Event Stream + 前端处理）

中断能力是流式的一个隐藏优势。如果 Agent 正在输出一长段内容，
用户发现方向不对，可以立刻点"停止"——后续的 token 就不再生成了。
非流式模式下，整个回答生成完了你才能看到它，想停都来不及。

.. code-block:: python

   # 后端的中断实现
   class StoppableStream:
       """支持中断的流式 Agent 执行"""
       def __init__(self):
           self._should_stop = False

       def stop(self):
           self._should_stop = True

       async def run(self, task):
           for chunk in agent.run_stream(task):
               if self._should_stop:
                   yield AgentStreamChunk("error", "用户中断了执行")
                   break
               yield chunk

.. admonition:: 流式的最佳实践
   :class: tip

   1. **TTFB 比总时间更重要**——用户不介意总时长 5 秒，但介意前 3 秒没反应
   2. **Agent 的思考步骤也应该流式输出**——用户想看"它在想什么"
   3. **前端做平滑的流式渲染**——不要一个 chunk 刷新一次 DOM，
      用缓冲区做 100ms 的节流
   4. **保留中断能力**——让用户可以随时停止 Agent 的执行
   5. **区分消息类型**——思考、行动、观察、答案用不同的展示样式
