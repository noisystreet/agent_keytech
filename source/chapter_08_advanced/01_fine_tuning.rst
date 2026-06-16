.. _chapter-08-fine-tuning:

===============================
微调策略
===============================

微调（Fine-tuning）是在预训练模型基础上，用特定领域数据进一步训练。
很多人听到"微调"第一反应是"我是不是需要微调一个模型来做 Agent？"

我的回答通常是：**先别微调，先做提示词工程和 RAG。**

为什么？因为微调的成本和风险远高于前两者。微调一次模型可能花费几千到几万
元，而改几句提示词是零成本。只有当你确认"提示词和 RAG 已经到极限了"，
才考虑微调。

这个"极限"的判断标准很简单：你的 Agent 在特定任务上的表现系统性低于期望，
而且你有一批高质量的训练数据。如果你的 Agent 只是在某些边界情况下表现不好，
微调可能是杀鸡用牛刀——先试着加几个 Few-shot 示例或调整工具描述。

什么时候应该微调？
====================

微调不是万能药。以下场景最有效，而有些场景微调完全是错误的方向。

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
这不是一个好选择。而且全量微调后的模型体积和原始模型一样大，
存储和分发成本都很高。

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

关于 LoRA 参数的几个经验：

.. code-block:: python

   # r 值选择
   r_recommendations = {
       "r=8":  "工具格式对齐、输出格式微调",
       "r=16": "大多数任务的首选——平衡表达能力和计算量",
       "r=32": "复杂工具链、多步推理模式微调（需要更多数据）",
       "r=64": "极少数需要大幅度调整的场景（不推荐，容易过拟合）",
   }

   # target_modules 选择
   module_recommendations = {
       "q_proj, v_proj": "标准选择，大多数场景够用",
       "q_proj, k_proj, v_proj, o_proj": "更强表达力，但计算量更大",
       "all": "慎用，计算量暴涨且容易过拟合",
   }

Agent 微调的数据准备
========================

微调 Agent 和微调普通 LLM 的数据格式不同。Agent 的数据包含**推理链**
和**工具调用轨迹**：

.. code-block:: python

   # Agent 微调数据的格式（ChatML 格式）
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

数据质量的两个关键点：

1. **轨迹完整性**：每条数据应该包含完整的感知-思考-行动-观察循环。
   缺少任何一环，模型学到的行为模式就是不完整的。

2. **错误轨迹也很重要**：除了成功的数据，也应该包含"工具调用失败后
   重试"和"最终承认失败"的轨迹。否则模型在面对错误时不知道该怎么做。

.. code-block:: python

   # 也包含错误恢复的数据
   error_trajectory = {
       "messages": [
           {"role": "system", "content": "..."},
           {"role": "user", "content": "查一下..."},
           # 第一次失败
           {"role": "assistant", "content": json.dumps({
               "tool": "search", "params": {"query": "..."}
           })},
           {"role": "tool", "content": "[error] 搜索服务超时"},
           # Agent 重试
           {"role": "assistant", "content": json.dumps({
               "thought": "搜索超时，换个搜索词重试",
               "tool": "search", "params": {"query": "..."}
           })},
           # 成功
           {"role": "tool", "content": "结果..."},
           {"role": "assistant", "content": json.dumps({
               "answer": "找到了..."
           })},
       ]
   }

数据量需要多少？
==================

一个常见的问题是："我需要多少条微调数据？"

.. code-block:: python

   # 数据量与效果的关系（经验值）
   data_estimates = {
       "100-500条":  "格式对齐、工具调用格式稳定",
       "500-2000条": "行为模式改善、特定任务效果提升",
       "2000-10000条": "显著的行为变化、新能力习得",
       "10000+条":   "大规模行为迁移（通常不必要）",
   }

   # 注意：数据质量 >> 数据量
   # 1000 条高质量的多步 Agent 轨迹 > 10000 条单轮问答
   # 每条数据都经过人工审核，比批量自动生成的效果好得多

微调后的评估
================

微调完成后，最关键的一步是评估。不要只测试"模型能不能回答训练集中的问题"。
要在**未见过的任务**上测试 Agent 的行为。

.. code-block:: python

   def evaluate_finetuned_agent(agent, test_suite):
       """微调后的 Agent 评估"""
       results = {"format_accuracy": 0, "task_completion": 0, "error_handling": 0}

       for test in test_suite:
           response = agent.run(test["task"])

           # 检查格式正确性
           if is_valid_json(response):
               results["format_accuracy"] += 1

           # 检查任务完成度
           if test["expected_keyword"] in response:
               results["task_completion"] += 1

           # 检查错误处理（如果测试包含错误场景）
           if test.get("expects_error_handling"):
               if "重试" in response or "抱歉" in response:
                   results["error_handling"] += 1

       n = len(test_suite)
       return {k: v/n for k, v in results.items()}
