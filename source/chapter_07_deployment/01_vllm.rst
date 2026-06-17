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

一个请求的 KV Cache 就占了 52GB，而一整张 A100 也只有 80GB。
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

   # Agent 通过标准 OpenAI 客户端连接，完全无感：
   client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
   agent = Agent(llm=client, tools=[search, calculator])

关键配置项详解
================

几个关键参数的工程含义：

.. list-table::
   :header-rows: 1

   * - 参数
     - 作用
     - 调优建议
   * - ``--tensor-parallel-size``
     - 模型分到几张 GPU 上
     - 8B 模型 1 张 24G 卡够用，70B 需 4-8 张
   * - ``--max-model-len``
     - 最大上下文长度
     - 设得越大，每个请求的 KVCache 占用越高。显存不够时优先减小这个值
   * - ``--gpu-memory-utilization``
     - KV Cache 可用显存比例
     - 默认 0.9，高并发提到 0.95，OOM 时降到 0.8
   * - ``--max-num-seqs``
     - 最大并发请求数
     - 默认 256，内存不够时减小
   * - ``--enable-prefix-caching``
     - 共享前缀缓存
     - Agent 共享 System Prompt 时建议开启

.. code-block:: bash

   # Agent 场景推荐的 vLLM 配置
   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --max-model-len 16384 \
       --gpu-memory-utilization 0.95 \
       --max-num-seqs 64 \
       --enable-prefix-caching \
       --trust-remote-code

吞吐量优化：连续批处理
=========================

vLLM 的另一个杀手锏是**连续批处理** （Continuous Batching）。

传统批处理的问题：你必须等一批里所有请求都完成后，才能处理下一批。
假设同一批里有 1 个长请求和 3 个短请求，3 个短请求早就完事了，
但必须傻等着那个长请求跑完——GPU 在空转。

连续批处理的思路：**每处理完一个请求，立即从等待队列中拉一个新请求进来**。

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

Prefix Caching 对 Agent 的加速
=================================

Agent 的每次推理都包含相同的 System Prompt。如果启用了 prefix caching，
vLLM 会缓存 System Prompt 的 KV Cache，后续请求复用缓存。

.. code-block:: text

   没有 prefix caching：
   用户 A: [System Prompt 3K] + [用户输入] → 全量计算
   用户 B: [System Prompt 3K] + [用户输入] → 重新全量计算

   有 prefix caching：
   用户 A: [System Prompt 3K] + [用户输入] → 计算 System Prompt 3K（首次）
   用户 B: [System Prompt 3K] + [用户输入] → 复用缓存，只计算用户输入部分

   → 用户 B 的首 token 延迟降低 60-80%

性能监控指标
================

部署 vLLM 后，应该关注以下指标：

.. list-table::
   :header-rows: 1

   * - 指标
     - 含义
     - 正常值
     - 需要关注的信号
   * - TTFT
     - 首 token 延迟
     - <500ms
     - >2s：need检查模型大小或前缀缓存
   * - ITL
     - 每个 token 的间隔时间
     - <50ms
     - >100ms：批量太大或显存不够
   * - Throughput
     - 每秒输出 token 数
     - >2000 tokens/s（8B 模型）
     - <500：需要增加批量大小
   * - GPU 利用率
     - GPU 计算时间占比
     - >85%
     - <60%：瓶颈在数据传输而非计算

.. admonition:: Agent 部署 vLLM 的经验总结
   :class: tip

   1. **不要用默认配置上生产**——至少调一下 max-model-len 和 gpu-memory-utilization
   2. **vLLM 瓶颈通常不在推理**——先在工具调用侧找优化空间
   3. **prefix caching 对 Agent 几乎总是有益的**——固定 System Prompt
   4. **如果有足够的请求并发** ，增加 max-num-seqs 提升吞吐量
   5. **监控 TTFT 和 ITL** ，而非总延迟——这两个指标告诉你到底慢在哪
