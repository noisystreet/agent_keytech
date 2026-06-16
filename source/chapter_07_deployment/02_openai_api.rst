.. _chapter-07-api:

===============================
OpenAI 兼容 API
===============================

OpenAI 兼容 API 已成为 LLM 服务的事实标准接口。使用这种接口可以让 Agent
无需修改代码就能切换不同的 LLM 后端。

.. code-block:: python

   # 统一的 Agent 调用接口
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

   # 透明切换后端
   backends = {
       "openai":   LLMBackend("https://api.openai.com/v1",        "gpt-4o"),
       "vllm":     LLMBackend("http://localhost:8000/v1",         "llama-3.1-8b"),
       "claude":   LLMBackend("https://api.anthropic.com/v1",     "claude-3-5-sonnet"),
       "deepseek": LLMBackend("https://api.deepseek.com/v1",      "deepseek-chat"),
   }

   agent = Agent(llm=backends["vllm"], tools=[search, calculator])
