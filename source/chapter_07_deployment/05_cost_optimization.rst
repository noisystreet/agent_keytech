.. _chapter-07-cost:

===============================
成本优化策略
===============================

Agent 的成本和传统 API 调用完全不同。一次 Agent 执行可能包含多次 LLM 调用、
多次工具调用、以及大量的上下文传输。如果不做优化，一个复杂的 Agent 任务
可能消耗几万甚至几十万 token——对应的成本可能从几分钱飙升到几块钱。

很多团队在 Agent 原型阶段用的是 GPT-4，每月花几百美元觉得还行。一旦上线，
用户量上来，成本可能变成几万美元——这时才想起来优化，就有点晚了。

成本去哪儿了？
================

.. list-table::
   :header-rows: 1

   * - 成本来源
     - 占比（估算）
     - 说明
   * - LLM 输入 token
     - 60-70%
     - System Prompt + 对话历史 + 工具返回结果，每次推理都重新传输
   * - LLM 输出 token
     - 15-25%
     - Agent 的 CoT 推理 + 最终回答
   * - 嵌入 API
     - 5-10%
     - RAG 检索的嵌入计算
   * - 外部工具 API
     - 5-10%
     - 搜索、数据库查询等

最大头的是 LLM 输入 token，占比超过 60%。每次 Agent 循环（思考→工具调用→
观察→再次思考），都需要把 System Prompt + 完整历史重新传给 LLM。

.. code-block:: python

   # 一个 5 步 Agent 执行的 token 消耗
   step_costs = [
       {"step": 1, "input_tokens": 2500, "output_tokens": 200, "cost": "$0.008"},
       {"step": 2, "input_tokens": 4500, "output_tokens": 150, "cost": "$0.014"},
       {"step": 3, "input_tokens": 6500, "output_tokens": 180, "cost": "$0.020"},
       {"step": 4, "input_tokens": 8500, "output_tokens": 160, "cost": "$0.026"},
       {"step": 5, "input_tokens": 10500, "output_tokens": 250, "cost": "$0.032"},
   ]
   total = sum(s["cost"] for s in step_costs)
   print(f"一个 5 步 Agent 任务成本: ${total:.3f}")  # ~$0.10
   # 如果有 10000 个用户每天用 5 次: $5000/天

看到没？输入 token 成本从第一步的 2500 涨到第五步的 10500——因为每一步
都要把之前的历史带进去。**Agent 的对话越长，单步成本越高。**

八大优化策略
================

1. System Prompt 瘦身
------------------------------

System Prompt 是每步推理的"固定开销"。精简它就是直接降低成本。

.. code-block:: python

   # 优化前：2000 tokens
   SYSTEM_PROMPT_VERBOSE = """
   你是一个智能助手。你有以下工具：search 可以搜索互联网。
   当你需要查找信息时使用 search 工具。search 接收一个
   query 参数。调用工具后等待结果再决定下一步。
   ...
   """

   # 优化后：400 tokens（减少 80%）
   SYSTEM_PROMPT_CONCISE = """
   你是助手。工具: search(query) 搜索互联网。
   每步: 思考→行动→观察。最终给出答案。
   """

2. 工具返回结果截断
------------------------------

工具返回的结果往往远大于 Agent 实际需要的信息量。

.. code-block:: python

   def truncate_tool_result(result: str, max_tokens=300) -> str:
       """截断工具返回结果"""
       tokens = tokenizer.encode(result)
       if len(tokens) <= max_tokens:
           return result
       # 只保留开头，中间用摘要替代
       head = tokenizer.decode(tokens[:max_tokens // 2])
       tail = tokenizer.decode(tokens[-max_tokens // 4:])
       return f"{head}\n...（共 {len(tokens)} tokens，已截断）...\n{tail}"

3. 对话历史摘要压缩
------------------------------

长对话不做压缩的话，每步成本线性增长。用 LLM 生成历史摘要可以
让后续步骤的输入大小保持稳定。

.. code-block:: python

   class ConversationCompressor:
       def __init__(self, llm, max_history_tokens=2000):
           self.llm = llm
           self.max_history = max_history_tokens
           self.summary = ""

       def add_and_compress(self, new_messages: list) -> list:
           if len(tokenize(str(new_messages))) < self.max_history:
               return new_messages

           # 生成摘要
           self.summary = self.llm.generate(
               f"压缩以下对话为 50 字摘要：{new_messages}"
           )
           # 返回压缩后的上下文
           return [{"role": "system", "content": f"历史摘要: {self.summary}"}]

4. 模型分级调用
------------------------------

不是每步都需要最强模型。简单步骤用小模型，复杂步骤用大模型。

.. code-block:: python

   class TieredLLM:
       """
       分级 LLM 调用：简单任务用小模型，复杂任务用大模型。
       目标：80% 的请求走便宜模型，只有 20% 走贵模型。
       """
       def __init__(self):
           self.cheap = LLMBackend("gpt-4o-mini", cost_per_1k="$0.00015")
           self.expensive = LLMBackend("gpt-4o", cost_per_1k="$0.0025")

       def chat(self, messages, tools=None):
           # 先让小模型判断复杂度
           complexity = self._estimate_complexity(messages)

           if complexity < 0.7:
               return self.cheap.chat(messages, tools)
           return self.expensive.chat(messages, tools)

       def _estimate_complexity(self, messages) -> float:
           prompt = f"评估复杂度（0-1）：{messages[-1]['content']}"
           result = self.cheap.generate(prompt, temperature=0.0)
           return float(result.strip())

5. 缓存重复请求
------------------------------

相同的用户输入直接返回缓存结果，不需要调用 LLM。

.. code-block:: python

   class AgentCache:
       def __init__(self, ttl=3600):
           self.cache = {}  # task → (result, timestamp)
           self.ttl = ttl

       def get(self, task: str) -> str:
           if task in self.cache:
               result, ts = self.cache[task]
               if time.time() - ts < self.ttl:
                   return result
           return None

       def set(self, task: str, result: str):
           self.cache[task] = (result, time.time())

6. 减少 CoT 输出长度
------------------------------

CoT 推理会生成大量中间 token。对于不需要复杂推理的步骤，关掉 CoT。

.. code-block:: python

   # 优化前：CoT 输出 ~500 tokens
   prompt = f"逐步推理：{question}"
   # 输出 500 tokens → 成本 x5

   # 优化后：直接输出 ~100 tokens
   prompt = f"直接给出答案：{question}"
   # 输出 100 tokens → 成本 x1

7. 批量处理
------------------------------

对于可以并行处理的任务，批量发送减少 API 调用次数。

.. code-block:: python

   def batch_agent_calls(tasks: list, agent) -> list:
       """批量处理多个独立任务"""
       from concurrent.futures import ThreadPoolExecutor

       with ThreadPoolExecutor(max_workers=5) as executor:
           # 批量提交，共享同一个 System Prompt
           results = list(executor.map(agent.run, tasks))
       return results

8. 选择合适的嵌入模型
------------------------------

嵌入模型的选择直接影响 RAG 成本。

.. code-block:: python

   # 嵌入模型成本对比（100 万条文档）
   embedding_costs = {
       "text-embedding-3-large": {"dim": 3072, "cost": "$130/M", "storage": "12 GB"},
       "text-embedding-3-small": {"dim": 1536, "cost": "$20/M",  "storage": "6 GB"},
       "bge-small-zh-v1.5":     {"dim": 512,  "cost": "免费",   "storage": "2 GB"},
   }

   # 对于大多数 Agent 场景，bge-small 就够用了
   # 除非你的检索质量要求非常高

成本优化的性价比排序
======================

.. list-table::
   :header-rows: 1

   * - 优先级
     - 策略
     - 成本节省（估算）
     - 实施难度
   * - 1
     - System Prompt 瘦身
     - 30-50%
     - 低
   * - 2
     - 工具结果截断
     - 20-40%
     - 低
   * - 3
     - 模型分级调用
     - 40-60%
     - 中
   * - 4
     - 对话历史压缩
     - 30-50%
     - 中
   * - 5
     - 缓存重复请求
     - 20-40%（取决于重复率）
     - 低
   * - 6
     - 减少 CoT 长度
     - 20-30%
     - 低
   * - 7
     - 批量处理
     - 10-20%
     - 低
   * - 8
     - 嵌入模型选型
     - 10-30%
     - 低

.. admonition:: 成本优化的黄金法则
   :class: tip

   **不要优化你不度量的事情。** 在上线前就做好 token 消耗的日志记录。
   上线后第一件事是看"哪个 Agent 任务的 token 消耗最高"，
   然后针对性地优化它。大多数情况下，20% 的任务消耗了 80% 的 token。
