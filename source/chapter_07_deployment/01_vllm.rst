.. _chapter-07-vllm:

===============================
vLLM 推理服务
===============================

vLLM 是目前最流行的 LLM 推理加速框架。很多人第一次接触它的时候，
以为它只是一个"更快地跑模型"的工具。其实 vLLM 解决了一个更根本的问题——
**显存管理**。

GPU 显存是推理部署中最稀缺的资源。一块 80G 的 A100，如果没有好的显存管理，
可能连一个 70B 的模型都跑不起来，更别说同时服务多个请求了。vLLM 的核心
贡献，就是像操作系统的虚拟内存一样，把显存的利用率从"凑合"提到了"接近极限"。

PagedAttention 解决了什么问题？
=================================

要理解 vLLM，首先要理解 Transformer 推理的显存开销来自哪里。

LLM 生成每个 token 时，都需要计算之前所有 token 的注意力分数。这些
中间状态叫做 **KV Cache** （Key-Value 缓存）。对于一个大模型来说，
KV Cache 的显存占用大到惊人：

::

   一个 70B 模型，处理一个 4096 token 的请求：
   KV Cache ≈ 2 × 4096 × 80 × 40 × 2 bytes ≈ 52 GB
               (层)  (头)  (维度) (精度)

你看，一个请求的 KV Cache 就占了 52GB，而一整张 A100 也只有 80GB。
如果 KV Cache 管理不好，你根本没法同时服务多个用户。

vLLM 的 PagedAttention 借鉴了操作系统**分页存储**的思路——不提前给每个
请求分配固定大小的 KV Cache，而是按需分配小块（pages）。这样：

- **内部碎片减少**：不需要为最短的请求预留最大空间
- **内存共享**：多个请求的相同前缀可以共享 KV Cache（比如共享 System Prompt）
- **按需分配**：请求越长，分配的 page 越多，不会浪费

.. code-block:: python

   # 对比传统方案和 vLLM 的显存利用率
   traditional = {
       "batch_size": 4,
       "max_seq_len": 4096,
       "allocated": 4 * 52,    # 208 GB — 3 张 A100 都不够
       "utilization": "低（大量预分配浪费）"
   }

   vllm_way = {
       "same_request": "每请求仅分配实际使用的显存",
       "memory_sharing": "共享前缀不重复占用",
       "utilization": "接近极限"
   }

启动和配置
==============

.. code-block:: bash

   # 基础启动
   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --tensor-parallel-size 2 \
       --max-model-len 8192

   # Agent 通过标准 OpenAI 客户端连接
   client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="not-needed"
   )

   # 通过 Agent 调用时，完全无感：
   agent = Agent(llm=client, tools=[search, calculator])
   agent.run("查一下今天的新闻")

几项关键配置的解读：

- ``--tensor-parallel-size 2``：把模型切到 2 张 GPU 上跑。只有在模型单卡
  放不下时才需要。8B 模型一张 24G 卡就够了，70B 才需要多卡。
- ``--max-model-len 8192``：限制最大上下文长度。这个值设得越大，每个请求
  的 KV Cache 占用就越高。如果发现显存不够，优先减小的是这个值，而不是换卡。
- ``--gpu-memory-utilization 0.9`` （默认值）：vLLM 会预留 90% 的显存给
  KV Cache，剩下 10% 给模型权重和前向计算。如果你的请求特别多，可以提到 0.95，
  但如果跑到 OOM，就降一点。

吞吐量优化：连续批处理
=========================

vLLM 的另一个杀手锏是**连续批处理** （Continuous Batching）。

传统批处理的问题：你必须等一批里所有请求都完成后，才能处理下一批。
假设同一批里有 1 个长请求和 3 个短请求，3 个短请求早就完事了，
但必须傻等着那个长请求跑完——GPU 在空转。

连续批处理的思路：**每处理完一个请求，立即从等待队列中拉一个新请求进来**。
这样 GPU 永远在处理可用的请求，不会空转。

.. code-block:: text

   传统批处理：
   [A B C D 全部完成] → [E F G H 全部完成] → GPU 有空闲等待期

   连续批处理：
   [A 完成] → 拉入 E
   [B C D 完成] → 拉入 F G H
   [E F G H 完成] → ...
   → GPU 几乎一直在满负荷运转

这个优化在 Agent 场景中特别重要。Agent 的每次工具调用之间都有等待时间
（调用外部 API），连续批处理可以在这段等待时间里处理其他 Agent 的请求。

部署实践建议
================

.. admonition:: 生产部署的几个要点
   :class: tip

   1. **不要用默认配置上生产**——至少调一下 max-model-len 和 gpu-memory-utilization
   2. **关注 TTFT（Time To First Token）和 ITL（Inter-Token Latency）**——
      TTFT 决定用户的第一印象，ITL 决定流式输出的流畅度
   3. **vLLM 搭配 Agent 时的典型瓶颈不是推理速度，而是工具调用延迟**——
      搜索引擎慢 2 秒比模型推理慢 200ms 影响大得多
   4. **可以使用 vLLM 的 --enable-prefix-caching 加速共享 prompt**——
      多个 Agent 如果共享 System Prompt，第一次后的请求会更快
