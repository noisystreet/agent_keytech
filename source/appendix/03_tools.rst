.. _appendix-tools:

===============================
工具与框架
===============================

Agent 框架
==============

.. list-table::
   :header-rows: 1

   * - 框架
     - 语言
     - 特点
   * - LangChain
     - Python
     - 最流行的 Agent 框架，功能全面
   * - CrewAI
     - Python
     - 轻量级多 Agent 编排
   * - AutoGen
     - Python
     - 微软多 Agent 对话框架
   * - Semantic Kernel
     - C# / Python
     - 微软 AI 编排 SDK

推理部署
============

.. list-table::
   :header-rows: 1

   * - 工具
     - 用途
     - 特点
   * - vLLM
     - LLM 推理加速
     - PagedAttention，高吞吐
   * - Ollama
     - 本地模型管理
     - 一键运行，适合开发
   * - TGI
     - LLM 推理服务
     - HuggingFace 官方方案
   * - llama.cpp
     - CPU 推理
     - 量化 + 纯 CPU 运行

向量数据库
==============

.. list-table::
   :header-rows: 1

   * - 数据库
     - 特点
     - 适用场景
   * - Chroma
     - 轻量嵌入式
     - 原型开发
   * - Milvus
     - 分布式高可用
     - 生产级大规模
   * - Qdrant
     - Rust 实现
     - 高性能检索
   * - Pinecone
     - SaaS 托管
     - 无需运维
