.. _chapter-08-fine-tuning:

===============================
微调策略
===============================

微调（Fine-tuning）是在预训练模型基础上，用特定领域数据进一步训练。
很多人听到"微调"第一反应是"我是不是需要微调一个模型来做 Agent？"

我的回答通常是：**先别微调，先做提示词工程和 RAG。**

为什么？因为微调的成本和风险远高于前两者。微调一次模型可能花费几千到几万
元，而改几句提示词是零成本。只有当你确认"提示词和 RAG 已经到极限了"，
才考虑微调。这个"极限"的判断标准很简单：你的 Agent 在特定任务上的表现
系统性低于期望，而且你有一批高质量的训练数据。

什么时候应该微调？
====================

微调不是万能药，它在以下场景中最有效：

.. list-table::
   :header-rows: 1

   * - 场景
     - 是否推荐微调
     - 原因
   * - 模型不熟悉你的工具格式
     - 推荐
     - 微调让模型学会特定的 JSON 输出格式
   * - 模型总在特定领域犯错
     - 推荐
     - 用领域数据纠正系统性错误
   * - 需要更低的推理成本
     - 不推荐微调
     - 应该蒸馏，不是微调
   * - 模型表现随机波动
     - 不推荐
     - 先检查提示词和示例是否稳定
   * - 需要新增知识
     - 不推荐微调
     - 用 RAG 更便宜、更灵活

Full Fine-tuning vs LoRA
==========================

微调有两种主要方式：全量微调和参数高效微调（PEFT）。

全量微调更新模型的所有参数。效果最好，但显存消耗最大——一个 7B 模型的全量
微调需要约 56GB 显存（AdamW 优化器的额外状态占用）。对大多数人来说，
这不是一个好选择。

LoRA（Low-Rank Adaptation）是目前最实用的方案。它的核心洞察是：
**模型参数的更新量可以分解为低秩矩阵**。你不需要更新所有参数，
而是在原始权重旁边挂载两个小矩阵来模拟参数更新。

.. code-block:: python

   # QLoRA：4bit 量化 + LoRA，一张 24G 显卡就能微调 7B 模型
   from peft import LoraConfig, get_peft_model

   lora_config = LoraConfig(
       r=16,              # LoRA 秩——影响"表达能力"
       lora_alpha=32,     # 缩放系数"影响"更新幅度"
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05, # 防止过拟合
       bias="none",
       task_type="CAUSAL_LM"
   )

   # 4bit 量化加载（QLoRA）
   model = AutoModelForCausalLM.from_pretrained(
       "llama-3.1-8b",
       quantization_config=load_in_4bit(),
   )
   model = get_peft_model(model, lora_config)
   # 可训练参数：约 16M（原始 8B 参数的 0.2%）

关于 LoRA 的 `r` 值，我见过很多人纠结这个参数。经验是：r=8 到 r=32
之间，对于 Agent 工具调用和格式对齐，r=16 通常就够了。更大的 r 不会
带来明显提升，只会增加计算量。

Agent 微调的数据准备
========================

微调 Agent 和微调普通 LLM 的数据格式不同。Agent 的数据包含**推理链**
和**工具调用轨迹**：

.. code-block:: python

   # Agent 微调数据的格式
   agent_training_sample = {
       "messages": [
           {"role": "system", "content": "你是一个搜索助手..."},
           {"role": "user", "content": "查一下 2024 年诺贝尔物理学奖"},
           {"role": "assistant", "content": json.dumps({
               "thought": "需要搜索诺贝尔奖信息",
               "tool": "search",
               "params": {"query": "2024 诺贝尔物理学奖"}
           })},
           {"role": "tool", "content": "John Hopfield, Geoffrey Hinton..."},
           {"role": "assistant", "content": json.dumps({
               "thought": "找到了，现在组织回答",
               "answer": "2024 年诺贝尔物理学奖授予..."
           })},
       ]
   }

一条好的 Agent 微调数据包含完整的思考-行动-观察-回答循环。数据质量比
数据量更重要——1000 条高质量的多步 Agent 轨迹，比 10000 条单轮问答
更能提升 Agent 能力。
