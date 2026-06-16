.. _chapter-08-fine-tuning:

===============================
微调策略
===============================

微调（Fine-tuning）是在预训练模型基础上，用特定领域数据进一步训练，提升
模型在该领域的效果。

.. code-block:: python

   # QLoRA：高效的参数微调方法
   from peft import LoraConfig, get_peft_model

   lora_config = LoraConfig(
       r=16,           # LoRA 秩
       lora_alpha=32,  # 缩放系数
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )

   model = AutoModelForCausalLM.from_pretrained("llama-3.1-8b")
   model = get_peft_model(model, lora_config)

   # QLoRA：4bit 基础 + LoRA 适配器
   model = AutoModelForCausalLM.from_pretrained(
       "llama-3.1-8b",
       quantization_config=load_in_4bit(),
   )
   model = get_peft_model(model, lora_config)
   # 可训练参数：16M（原始 8B 的 0.2%）
