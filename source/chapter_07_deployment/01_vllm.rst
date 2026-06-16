.. _chapter-07-vllm:

===============================
vLLM 推理服务
===============================

vLLM 是目前最流行的 LLM 推理加速框架，通过 PagedAttention 和连续批处理
大幅提升推理吞吐量。

核心特性
============

- **PagedAttention**：高效管理 KV Cache，减少显存碎片
- **连续批处理**：动态组批，不需要等一批全部完成才能处理下一批
- **OpenAI 兼容 API**：直接作为 OpenAI API 的 drop-in replacement

.. code-block:: bash

   # 启动 vLLM 服务
   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --tensor-parallel-size 2 \
       --max-model-len 8192

   # Agent 通过标准 OpenAI 客户端连接
   client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="not-needed"
   )
