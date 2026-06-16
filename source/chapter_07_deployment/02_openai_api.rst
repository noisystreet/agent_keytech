.. _chapter-07-api:

===============================
OpenAI 兼容 API
===============================

OpenAI 兼容 API 已经成为 LLM 服务的事实标准。设计一个 Agent 系统时，
如果你选择了 OpenAI 兼容接口，就意味着你可以在**不修改一行 Agent 代码**
的情况下，在 OpenAI / vLLM / Claude / DeepSeek / 本地模型之间切换。

这听起来像是一个简单的技术选择，但它带来的灵活性比你想象的大得多。

为什么是 OpenAI 兼容？
=========================

市面上有几十种 LLM API，OpenAI、Anthropic、Google、DeepSeek、Mistral……
每家都有自己的 SDK 和接口格式。如果 Agent 直接调用这些 SDK，每换一个模型
就要改一遍代码。

OpenAI 兼容 API 提供了一层**接口抽象**：

.. code-block:: python

   # 不管你背后是什么模型，调用方式都一样
   class LLMBackend:
       def __init__(self, base_url, model, api_key="sk-xxx"):
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

兼容 API 的"坑"
====================

OpenAI 兼容并不等于 100% 兼容。几个常见的差异：

.. admonition:: 工具调用格式差异
   :class: caution

   有些兼容实现不支持 ``tool_choice=required`` （强制调用工具），
   或者对 tools 数组的长度有限制。在切换后端时，一定要跑通工具调用测试。

.. admonition:: 流式输出差异
   :class: caution

   Streaming 模式下，不同后端的事件格式可能略有不同。比如有些后端
   会把 finish_reason 放在最后一个 chunk 里，有的不发。

.. admonition:: 速率限制实现
   :class: caution

   OpenAI 返回 429 时带 Retry-After 头，但很多兼容实现只是简单返回
   429 而不给等待时间。Agent 的 retry 逻辑需要考虑到这一点。

.. code-block:: python

   # 一个兼容性更好的 LLM 调用封装
   class RobustLLMBackend:
       def __init__(self, base_url, model, api_key="sk-xxx"):
           self.client = OpenAI(base_url=base_url, api_key=api_key)
           self.model = model

       def chat_with_fallback(self, messages, tools=None, max_retries=3):
           for attempt in range(max_retries):
               try:
                   return self.chat(messages, tools)
               except Exception as e:
                   if attempt == max_retries - 1:
                       raise
                   # 指数退避等待
                   time.sleep(2 ** attempt)
