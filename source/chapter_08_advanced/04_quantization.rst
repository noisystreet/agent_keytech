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
   # 把一个 FP16 的数值范围 [-max, max] 映射到 INT8 的 [-127, 127]
   # 每一个损失了一些精度，但存储减半

   def quantize_to_int8(fp16_weights):
       """
       将 FP16 权重量化为 INT8。
       实际实现要复杂得多（按块量化、对称/非对称等）。
       """
       max_val = fp16_weights.abs().max()
       scale = 127.0 / max_val  # 缩放因子
       quantized = (fp16_weights * scale).round().to(torch.int8)
       return quantized, scale  # scale 需要存储，用于反量化

   def dequantize_to_fp16(quantized, scale):
       """将 INT8 反量化为 FP16"""
       return quantized.to(torch.float16) / scale

两种主流量化方法
====================

1. GPTQ（Post-Training Quantization）
----------------------------------------

GPTQ 在量化后需要用一小部分校准数据微调量化参数，以恢复精度损失。

.. code-block:: bash

   # GPTQ 量化流程
   pip install auto-gptq

   from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

   quantize_config = BaseQuantizeConfig(
       bits=4,                   # 量化位数
       group_size=128,           # 分组大小（越小精度越高）
       desc_act=False,           # 是否按列激活排序
   )

   model = AutoGPTQForCausalLM.from_pretrained(
       "meta-llama/Llama-3.1-8B-Instruct",
       quantize_config=quantize_config
   )

   # 用校准数据微调量化参数
   model.quantize(calibration_data)
   model.save_pretrained("quantized-model/")

2. AWQ（Activation-Aware Weight Quantization）
----------------------------------------------

AWQ 比 GPTQ 更新，思路是：**不是所有权重的重要性都一样**。有些权重
对应更重要的激活值（即对模型的输出影响更大），应该保留更高精度。

.. code-block:: python

   # AWQ 量化（使用 bitsandbytes）
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig

   quant_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype="float16",
       bnb_4bit_quant_type="nf4",  # NF4 比 INT4 更好
       bnb_4bit_use_double_quant=True,  # 双重量化，进一步压缩
   )

   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-3.1-8B-Instruct",
       quantization_config=quant_config
   )

量化对 Agent 的影响
======================

Agent 场景中，量化带来的推理速度提升非常显著：

.. code-block:: python

   # 同一个 8B 模型在不同精度下的推理速度
   performance = {
       "FP16": {
           "size": "16 GB",
           "speed": "30 tokens/s",
           "quality": "基准"
       },
       "INT8": {
           "size": "8 GB",
           "speed": "45 tokens/s",
           "quality": "约等于 FP16"
       },
       "INT4": {
           "size": "4 GB",
           "speed": "60 tokens/s",
           "quality": "轻微下降（常规任务不感知）"
       },
   }

但量化也有代价。对于 Agent 来说，最关键的是**量化后模型工具调用和
推理链的准确性**。经验是：

- INT8：几乎没有可感知的质量损失，Agent 行为正常
- INT4：常规任务正常，复杂推理链偶见异常（如 JSON 格式错误增加）
- NF4：比 INT4 好，接近 INT8 水平

.. admonition:: 量化的最佳实践
   :class: tip

   如果 Agent 只需要做简单的搜索+总结，INT4 足够。
   如果 Agent 涉及复杂多步推理（5 步以上），建议用 INT8 或 NF4。
   如果 Agent 需要精确的工具调用（如代码生成），优先用 FP16。
   不要在生产环境中盲打量化——先在评估集上测试量化后的工具调用准确率。

量化与 Agent 部署的协同
===========================

量化的真正价值在于它改变了 Agent 的部署架构选择：

.. code-block:: python

   # 量化前：需要 2 张 A100 才能跑 70B 模型
   # 量化后（INT4）：1 张 A100 就能跑
   # 成本降低 50%，部署复杂度大幅下降

   # 对于 8B 模型：
   # FP16：需要 24G 显存一张卡
   # INT4：可以在 8G 显存的消费级显卡上运行
   # 成本从"服务器级"降到了"个人电脑级"

这意味着你可以把更强的模型（70B→8B 量级）部署到更多场景。
但不要贪心——量化的性价比曲线是指数递减的。从 FP16 到 INT8
收益很大（50% 存储节省 + 几乎无质量损失），从 INT8 到 INT4
收益较小（25% 存储节省 + 轻微质量损失）。
