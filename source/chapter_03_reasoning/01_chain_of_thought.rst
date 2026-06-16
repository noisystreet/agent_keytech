.. _chapter-03-cot:

===============================
思维链（Chain-of-Thought）
===============================

思维链（Chain-of-Thought, CoT）是提示 LLM 逐步推理而非直接给出答案的技术。
Wei et al. (2022) 发现，在提示词中加入"Let's think step by step"可以
显著提升复杂推理任务的准确率，在数学推理（GSM8K）上从 18% 提升到 58%。

这个提升幅度之大，让很多人第一次意识到：**LLM 不是不会推理，而是需要
被引导去推理。** 直接问"结果是多少"，模型倾向于"猜"答案。但如果引导它
"一步步计算"，模型就能把复杂问题分解成可控的小步骤。

为什么 CoT 有效？
========================

CoT 的核心价值在于将隐式推理过程**显式化**。

LLM 本质上是一个"next token predictor"。当你直接问"24 乘以 37 等于多少？"
时，模型需要在单次前向传播中完成从问题到答案的映射——这相当于让一个人
不经过任何中间计算直接报出 888 这个结果。即使对 LLM 来说，这个"一步到位"
的映射也是困难的。

CoT 的做法是把推理过程展开：先算 24×30=720，再算 24×7=168，
然后 720+168=888。每一步计算都很简单，模型在每一步的正确率都很高，
最后累积出正确结果。

.. list-table::
   :header-rows: 1

   * - 方面
     - 直接回答
     - CoT 逐步推理
   * - 推理过程
     - 隐含在单次生成中
     - 每一步显式输出
   * - 错误定位
     - 只看到错误答案
     - 可以定位到哪一步出错
   * - 人类可解释
     - 差
     - 好
   * - 复杂推理
     - 困难
     - 显著提升
   * - Token 消耗
     - 少
     - 多（但值得）

三种 CoT 变体
===================

1. Zero-shot CoT
------------------------------

最简单的实现方式——在问题后追加"逐步推理"指令。不需要任何示例。

.. code-block:: python

   def zero_shot_cot(llm, question: str) -> str:
       prompt = f"""
       问题：{question}

       逐步推理（Let's think step by step）：
       """
       reasoning = llm.generate(prompt, temperature=0.0)
       return reasoning

这个方法的优势是**零成本**（不需要准备 Few-shot 示例），对大多数推理任务
都有不错的提升。缺点是不够稳定——模型可能在某些步骤上"走偏"。

2. Few-shot CoT
------------------------------

提供推理示例，引导模型模仿正确的推理模式。示例的质量直接影响效果。

.. code-block:: python

   def few_shot_cot(llm, question: str) -> str:
       prompt = """
       问题：教室有 5 排椅子，每排 6 把，坐了 12 个人，还有多少个空位？
       推理：5 排 × 6 把 = 30 把椅子。30 - 12 = 18 个空位。答案是 18。

       问题：商店有 23 个苹果，卖出 7 个后又进货 15 个，现在有多少个？
       推理：
       """
       reasoning = llm.generate(prompt, temperature=0.0)
       return reasoning

Few-shot 示例的选择有几个技巧：

.. code-block:: python

   # 示例选择技巧
   tips = {
       "相关性": "示例和用户问题类型越接近越好，数学题配数学题示例",
       "多样性": "示例覆盖不同的推理模式（正向推理、反向推理、对比推理）",
       "清晰度": "每个示例的推理步骤要清晰分开，不要一步跨太大",
       "数量": "2-4 个示例效果最好，太多了反而分散注意力",
   }

3. Auto-CoT
------------------------------

为解决手动设计 Few-shot 示例的人力成本，Auto-CoT（Zhang et al., 2022）
自动从数据集中选取多样化示例并生成推理链。

.. code-block:: python

   class AutoCoT:
       def __init__(self, llm, example_pool: list):
           self.llm = llm
           self.example_pool = example_pool  # 预置示例池

       def generate(self, question: str, k: int = 3) -> str:
           # 选择与当前问题最相关的 k 个示例
           selected = self._select_diverse_examples(question, k)
           example_text = "\n\n".join(
               f"问题：{ex['question']}\n推理：{ex['reasoning']}\n答案：{ex['answer']}"
               for ex in selected
           )
           prompt = f"{example_text}\n\n问题：{question}\n推理："
           return self.llm.generate(prompt, temperature=0.0)

CoT 在 Agent 中的实际应用
==============================

Agent 场景中，CoT 有三个典型用途。理解这些用途可以帮助你在合适的
地方使用 CoT，避免不必要的 token 开销。

1. 工具调用决策
---------------------------------

让 LLM 逐步分析应该调用哪个工具，而不是直接猜测。

.. code-block:: python

   def tool_selection_with_cot(task: str, tools: list) -> str:
       """用 CoT 分析应该调用哪个工具"""
       prompt = f"""
       任务：{task}
       可用工具：{', '.join(t.name for t in tools)}

       逐步分析：
       1. 这个任务需要什么类型的信息？
       2. 哪个工具能提供这类信息？
       3. 这个工具有什么限制？

       最终决策：
       """
       return llm.generate(prompt, temperature=0.0)

2. 多步推理分解
---------------------------------

复杂任务拆解为可执行的子步骤。

.. code-block:: python

   def decompose_with_cot(task: str) -> list:
       """用 CoT 分解复杂任务"""
       prompt = f"""
       任务：{task}

       逐步分解：
       1. 这个任务的最终目标是什么？
       2. 需要先获取哪些信息？
       3. 这些信息之间有什么依赖关系？
       4. 按什么顺序获取这些信息？

       执行计划：
       """
       plan = llm.generate(prompt, temperature=0.0)
       return parse_steps(plan)

3. 错误回溯
---------------------------------

当 Agent 执行失败时，用 CoT 分析失败原因。

.. code-block:: python

   def error_analysis_with_cot(task: str, steps: list, error: str) -> str:
       """用 CoT 回溯错误原因"""
       prompt = f"""
       任务：{task}
       执行步骤：{steps}
       错误：{error}

       逐步分析：
       1. 哪一步出了问题？
       2. 问题是出在工具调用还是推理？
       3. 如果重试，应该调整什么？
       """
       return llm.generate(prompt, temperature=0.0)

CoT 与 ReAct 的区别
=======================

一个常见的混淆点：CoT 和 ReAct 都是在"逐步思考"，但它们有本质区别。

.. list-table::
   :header-rows: 1

   * - 维度
     - CoT
     - ReAct
   * - 行动能力
     - 纯推理，不调用外部工具
     - 推理 + 工具调用交替
   * - 信息源
     - 仅依赖模型内部知识
     - 依赖外部工具获取实时信息
   * - 幻觉风险
     - 高（无法验证事实）
     - 低（可交叉验证）
   * - Token 消耗
     - 低（只推理）
     - 高（推理 + 工具调用结果）
   * - 核心用途
     - 提升推理准确性
     - 让 Agent 能行动

CoT 是 ReAct 的"推理组件"——在实际的 Agent 系统中，CoT 嵌入在 ReAct
循环中，在每次工具调用前做推理分析。

Practical Tips
================

.. admonition:: 什么时候用 CoT，什么时候不用？
   :class: tip

   用 CoT 的时机：
   - 数学计算、逻辑推理、多步骤分析
   - 需要链式推理的任务

   不用 CoT 的时机：
   - 常识问答（"首都是哪里？"）
   - 直接的事实查询
   - 创意写作（CoT 会限制创造力）

.. admonition:: 控制 CoT 的输出长度
   :class: caution

   CoT 会生成大量中间 token。一个 5 步的推理可能需要 500+ token。
   对于 Agent 的多步循环，每步都做完整 CoT 会迅速消耗上下文预算。

   建议：只在关键决策步骤使用 CoT，常规步骤用直接生成。

Limitations
============

.. list-table::
   :header-rows: 1

   * - 局限
     - 说明
     - 缓解方案
   * - 成本增加
     - CoT 生成大量中间 token，延迟和费用上升
     - 简单问题走直接回答，复杂问题走 CoT
   * - 虚假推理
     - LLM 可能生成"看起来合理"但实际错误的推理链
     - 结合工具验证每一步
   * - 不适用于所有任务
     - 对常识推理帮助不大，主要用于数学、逻辑、规划
     - 根据任务类型选择策略
   * - 推理链过长
     - 超过一定长度后，中间步骤质量下降
     - 分段推理 + 摘要

参考文献
============

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", 2022
- Kojima et al., "Large Language Models are Zero-Shot Reasoners", 2022
- Zhang et al., "Automatic Chain of Thought Prompting in Large Language Models", 2022
