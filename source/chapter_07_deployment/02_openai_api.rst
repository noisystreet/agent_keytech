.. _chapter-07-api:

===============================
OpenAI 兼容 API
===============================

OpenAI 兼容 API 已经成为 LLM 服务的事实标准。设计一个 Agent 系统时，
如果你选择了 OpenAI 兼容接口，就意味着你可以在**不修改一行 Agent 代码**
的情况下在 OpenAI / vLLM / DeepSeek / 本地模型之间切换。

很多 Agent 框架在设计之初就选择了 OpenAI 兼容接口作为底层的 LLM 抽象。
这个选择不是偶然的——OpenAI 的接口设计最成熟、文档最完善、生态最丰富。
如果你在做一个新的 Agent 框架，也建议直接用 OpenAI 兼容接口，而不是
自己定义一套 LLM 抽象层。

为什么是 OpenAI 兼容？
=========================

市面上有几十种 LLM API，OpenAI、Anthropic、Google、DeepSeek、Mistral……
每家都有自己的 SDK 和接口格式。如果 Agent 直接调用这些 SDK，每换一个模型
就要改一遍代码。

OpenAI 兼容 API 提供了一层**接口抽象**：

.. code-block:: python

   import openai
   from openai import OpenAI

   class LLMBackend:
       """
       不管你背后是什么模型，调用方式都一样。
       base_url 决定了连接到哪个后端。
       """
       def __init__(self, base_url, model, api_key=""):
           self.client = OpenAI(base_url=base_url, api_key=api_key)
           self.model = model

       def chat(self, messages, tools=None, **kwargs):
           return self.client.chat.completions.create(
               model=self.model,
               messages=messages,
               tools=tools,
               **kwargs
           )

透明切换后端
================

.. code-block:: python

   # 所有后端的接口完全一致
   backends = {
       "openai":    LLMBackend("https://api.openai.com/v1",        "gpt-4o"),
       "vllm":      LLMBackend("http://localhost:8000/v1",         "llama-3.1-8b"),
       "deepseek":  LLMBackend("https://api.deepseek.com/v1",      "deepseek-chat"),
       "qwen":      LLMBackend("https://dashscope.aliyuncs.com/compatible-mode/v1",
                                                              "qwen-plus"),
       "local":     LLMBackend("http://192.168.1.100:8000/v1",     "qwen-2.5-7b"),
   }

   # Agent 的 LLM 参数只是一个配置项
   agent = Agent(
       llm=backends["deepseek"],   # 改这一行就够了
       tools=[search, calculator]
   )

这里有一个重要的工程判断：你不需要所有后端都达到 OpenAI GPT-4 的水平。
在 Agent 架构中，LLM 只是推理引擎，真正干活的是工具。一个弱模型 +
好的工具链，往往比强模型 + 无工具更实用。

什么时候用哪个？
==================

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐后端
     - 理由
   * - 开发调试
     - OpenAI / Claude
     - 最稳定、最少意外行为
   * - 生产推理
     - vLLM + 本地模型
     - 成本可控、延迟可预测
   * - 成本敏感
     - DeepSeek / Qwen
     - 性价比高，中文友好
   * - 隐私敏感
     - 本地 vLLM
     - 数据不出内网
   * - 高并发
     - vLLM + 多卡
     - 可用 tensor parallel 扩展

兼容 API 的"坑"
====================

OpenAI 兼容并不等于 100% 兼容。以下是在 Agent 中切换后端时最常踩的坑。

.. admonition:: 工具调用格式差异
   :class: caution

   有些兼容实现不支持 ``tool_choice=required`` （强制调用工具），
   或者对 tools 数组的字段名不一致。比如：
   - OpenAI 用 `parameters`，有些实现用 `input_schema`
   - DeepSeek 的 tool_choice 可能不支持 `"required"`
   在切换后端时，一定要跑通工具调用测试。

.. admonition:: 流式输出差异
   :class: caution

   Streaming 模式下，不同后端的事件格式可能略有不同：
   - OpenAI 在最后一个 chunk 里设置 `finish_reason`
   - 有些实现在每个 chunk 都带 `finish_reason: null`，只有最后一个是 `"stop"`
   - 有些实现完全不发 finish_reason

   解决方案：在 Agent 框架层统一处理流式输出，不依赖后端的 finish_reason。

.. admonition:: 速率限制差异
   :class: caution

   OpenAI 返回 429 时带 Retry-After 头，但很多兼容实现只是简单返回
   429 而不给等待时间。如果你的 Agent 依赖 Retry-After 做指数退避，
   在这些后端上会直接失败。

.. code-block:: python

   # 一个兼容性更好的 LLM 调用封装
   class RobustLLMBackend:
       """兼容多后端的 LLM 调用封装"""
       def __init__(self, base_url, model, api_key="sk-xxx"):
           self.client = OpenAI(base_url=base_url, api_key=api_key)
           self.model = model

       def chat_with_fallback(self, messages, tools=None, max_retries=3):
           """带重试和降级的 LLM 调用"""
           for attempt in range(max_retries):
               try:
                   return self._try_chat(messages, tools)
               except UnsupportedToolChoiceError:
                   # 如果不支持 tool_choice=required，降级为 auto
                   return self._try_chat(messages, tools, tool_choice="auto")
               except RateLimitError:
                   wait = 2 ** attempt  # 指数退避
                   time.sleep(wait)
               except Exception as e:
                   if attempt == max_retries - 1:
                       raise
                   time.sleep(1)
           raise RuntimeError("LLM 调用失败")

并发与速率控制
================

当 Agent 系统需要处理大量请求时，需要对 LLM 后端的调用做并发控制。

.. code-block:: python

   import asyncio
   from asyncio import Semaphore

   class RateLimitedBackend:
       """
       带速率限制的 LLM 后端封装。
       防止 Agent 的并发请求打满 API 配额。
       """
       def __init__(self, backend, max_concurrent=10, rpm=60):
           self.backend = backend
           self.semaphore = Semaphore(max_concurrent)
           self.rpm = rpm
           self.request_times = []

       async def chat(self, messages, tools=None):
           # 速率限制
           await self._wait_for_capacity()

           async with self.semaphore:
               return await self.backend.chat(messages, tools)

       async def _wait_for_capacity(self):
           """确保每分钟不超过 RPM 限制"""
           now = time.time()
           self.request_times = [t for t in self.request_times
                                if now - t < 60]
           if len(self.request_times) >= self.rpm:
               sleep_time = 60 - (now - self.request_times[0])
               if sleep_time > 0:
                   await asyncio.sleep(sleep_time)
           self.request_times.append(time.time())

多后端负载均衡
================

在生产环境中，你可能同时使用多个后端来分摊负载和成本。

.. code-block:: python

   class LoadBalancedBackend:
       """
       多后端负载均衡。根据优先级和当前负载分发请求。
       """
       def __init__(self, backends: dict):
           # backends = {"primary": backend1, "fallback": backend2}
           self.backends = backends
           self.priority = ["primary", "secondary", "fallback"]

       def chat(self, messages, tools=None):
           errors = []
           for name in self.priority:
               backend = self.backends.get(name)
               if not backend:
                   continue
               try:
                   return backend.chat(messages, tools)
               except Exception as e:
                   errors.append(f"{name}: {str(e)}")
                   continue
           raise RuntimeError(f"所有后端都失败: {errors}")

.. admonition:: 选择后端的经验法则
   :class: tip

   - **预算充足**：主力 OpenAI + 备用 vLLM（本地模型）
   - **成本敏感**：主力 DeepSeek/Qwen + 备用 OpenAI（关键任务）
   - **隐私优先**：纯本地 vLLM + 加密传输
   - **高可用**：多个后端做负载均衡 + 自动降级
