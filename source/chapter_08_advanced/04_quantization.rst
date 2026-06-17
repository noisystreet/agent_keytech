.. _chapter-08-quantization:

===============================
量化技术
===============================

量化（Quantization）是将模型权重从高精度（FP16）压缩到低精度（INT4/INT8）
的技术。它的核心洞察是：

**模型权重的精度不是越多越好**。16 位浮点数能表示 6 万多个不同的值，
但大多数模型中 90% 的权重只集中在很小的范围内。你用 4 位（16 个值）
也能近似出相同的推理结果。

这个发现对 Agent 部署有直接影响。一个 70B 的模型在 FP16 下需要 140GB 显存
（一张 A100 放不下），量化到 INT4 只需要 35GB——一张 A100 就能跑。
这意味着你可以在更便宜的硬件上运行更强的模型。

量化的基本概念
================

.. list-table::
   :header-rows: 1

   * - 精度
     - 存储大小
     - 相对 FP16 体积
     - 质量损失
     - 典型场景
   * - FP32
     - 4 bytes/参数
     - 200%
     - 无
     - 训练
   * - FP16
     - 2 bytes/参数
     - 100%（基准）
     - 无
     - 训练
   * - INT8
     - 1 byte/参数
     - 50%
     - 几乎无
     - 高精度推理
   * - INT4
     - 0.5 byte/参数
     - 25%
     - 轻微（可接受）
     - 高效率推理
   * - NF4
     - 0.5 byte/参数
     - 25%
     - 几乎无（优于 INT4）
     - QLoRA 微调

.. code-block:: python

   # 量化的数学本质
   def quantize_to_int8(fp16_weights):
       """
       将 FP16 权重量化为 INT8。
       scale 是映射到 int8 范围的比例因子。

       看似简单，但实际工程中有几个关键问题：
       1. 按块量化（group-wise）vs 按层量化（per-tensor）
       2. 对称量化 vs 非对称量化
       3. 校准数据集的选择
       """
       max_val = fp16_weights.abs().max()
       scale = 127.0 / max_val
       quantized = (fp16_weights * scale).round().to(torch.int8)
       return quantized, scale

量化对推理速度的影响
=====================

量化主要通过两个维度加速推理：

1. **存储带宽减少**：INT4 的数据量是 FP16 的 1/4，从显存读取权重的时间
   大幅缩短。这对 Agent 场景特别重要——每次推理都需要读取大量权重
2. **计算加速**：INT4 的矩阵乘法比 FP16 快 2-4 倍（取决于硬件）

.. code-block:: python

   # 同一个 8B 模型在不同精度下的推理性能
   performance = {
       "FP16": {"size": "16 GB", "speed": "30 tokens/s", "quality": "基准"},
       "INT8": {"size": "8 GB",  "speed": "45 tokens/s", "quality": "几乎无损失"},
       "INT4": {"size": "4 GB",  "speed": "60 tokens/s", "quality": "轻微下降"},
       "NF4":  {"size": "4 GB",  "speed": "58 tokens/s", "quality": "接近 INT8"},
   }

两种主流量化方法
====================

1. GPTQ（Post-Training Quantization）
----------------------------------------

GPTQ 的核心思路：量化后，用一小部分校准数据微调量化参数，恢复精度损失。

.. code-block:: python

   from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

   quantize_config = BaseQuantizeConfig(
       bits=4,                   # 量化位数
       group_size=128,           # 分组大小（越小精度越高，但量化后模型越大）
       desc_act=False,           # 是否按列激活排序
   )

   model = AutoGPTQForCausalLM.from_pretrained(
       "meta-llama/Llama-3.1-8B-Instruct",
       quantize_config=quantize_config
   )
   model.quantize(calibration_data)

GPTQ 的关键参数是 group_size：

.. list-table::
   :header-rows: 1

   * - group_size
     - 精度
     - 模型大小（8B 模型）
     - 适合场景
   * - 32
     - 最好
     - 5.5 GB
     - 精度优先
   * - 128
     - 良好
     - 4.5 GB
     - 默认推荐
   * - 256
     - 可接受
     - 4.2 GB
     - 存储优先

2. AWQ（Activation-Aware Weight Quantization）
----------------------------------------------

AWQ 比 GPTQ 更新，思路是：**不是所有权重的重要性都一样**.

有些权重对应更重要的激活值（即对模型的输出影响更大），应该保留更高精度。
AWQ 通过分析激活值的分布来识别哪些权重更重要，然后对它们做更精细的量化。

.. code-block:: python

   # AWQ 量化（使用 bitsandbytes）
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig

   quant_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype="float16",
       bnb_4bit_quant_type="nf4",  # NF4 比普通 INT4 更好
       bnb_4bit_use_double_quant=True,  # 双重量化，进一步压缩
   )
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-3.1-8B-Instruct",
       quantization_config=quant_config
   )

量化对 Agent 的具体影响
============================

量化对 Agent 的影响和纯文本生成不同。Agent 需要**精确的工具调用**——
输出格式必须严格符合 JSON 规范，推理步骤必须逻辑一致。

.. list-table::
   :header-rows: 1

   * - Agent 能力
     - FP16
     - INT8
     - INT4/NF4
   * - 简单问答
     - 正常
     - 正常
     - 正常
   * - 工具调用格式
     - 正常
     - 正常
     - 偶见格式错误
   * - 多步推理（3-5 步）
     - 正常
     - 正常
     - 正常
   * - 复杂推理（>5 步）
     - 正常
     - 正常
     - 偶见逻辑断裂
   * - 长上下文（>32K）
     - 正常
     - 正常
     - 质量下降明显

.. admonition:: Agent 量化的最佳实践
   :class: tip

   - **简单搜索+总结 Agent**：INT4 足够了，省一半显存
   - **多步推理 Agent**：INT8 或 NF4，INT4 在多步逻辑上偶有异常
   - **代码生成 Agent**：FP16，工具调用格式必须 100% 准确
   - **上生产前务必测试**：在评估集上对比量化前后的工具调用准确率

量化与 Agent 部署的协同
===========================

量化的真正价值在于它改变了 Agent 的部署架构选择。

.. code-block:: python

   # 量化前后部署方案对比
   before_quant = {
       "70B模型": "需要 2 张 A100-80G (约 $50/小时)",
       "部署成本": "高",
       "响应延迟": "低（推理速度快）",
       "Agent 能力": "强",
   }

   after_quant = {
       "70B模型（INT4）": "1 张 A100-80G 就够了",
       "8B模型（INT8）": "1 张 RTX 4090-24G 就能跑",
       "部署成本": "降低 50-80%",
       "响应延迟": "略微增加但可接受",
       "Agent 能力": "轻微下降（需要验证）",
   }

性价比曲线
=============

量化的收益是**递减**的：

.. code-block:: text

   从 FP16 到 INT8：
   → 存储减少 50% + 速度提升 50%
   → 质量损失：几乎为零
   → 收益最大，无脑推荐

   从 INT8 到 INT4：
   → 存储再减少 25% + 速度再提升 30%
   → 质量损失：轻微（但 Agent 场景可能影响工具调用）
   → 边际收益递减，需要评估

   INT4 到 NF4：
   → 存储相同，速度相同
   → 质量比 INT4 好一点
   → 如果必须用 4bit，选 NF4

.. code-block:: python

   # 量化选择决策流程
   def select_quantization(task_complexity: str, budget: str) -> str:
       """根据任务复杂度和预算选择量化策略"""
       if task_complexity == "simple" and budget == "low":
           return "INT4 / NF4"
       elif task_complexity == "simple" and budget == "high":
           return "INT8"
       elif task_complexity == "complex" and budget == "low":
           return "NF4"
       elif task_complexity == "complex" and budget == "high":
           return "FP16"
       return "INT8"  # 默认推荐
