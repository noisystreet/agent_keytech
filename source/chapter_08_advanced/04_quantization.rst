.. _chapter-08-quantization:

===============================
量化技术
===============================

量化（Quantization）将模型权重从高精度（FP16）压缩到低精度（INT4/INT8），
在几乎不损失质量的前提下大幅降低模型大小和推理延迟。

.. list-table::
   :header-rows: 1

   * - 量化精度
     - 权重大小
     - 质量损失
     - 典型场景
   * - FP16
     - 100%（基准）
     - 无
     - 训练
   * - INT8
     - 50%
     - 几乎无
     - 推理（高精度）
   * - INT4
     - 25%
     - 轻微
     - 推理（高效率）
   * - NF4
     - 25%
     - 几乎无
     - QLoRA 微调

.. code-block:: bash

   # 使用 bitsandbytes 加载量化模型
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig

   quant_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype="float16",
       bnb_4bit_quant_type="nf4"
   )

   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-3.1-8B-Instruct",
       quantization_config=quant_config
   )
