.. _chapter-08-distillation:

===============================
模型蒸馏
===============================

知识蒸馏（Knowledge Distillation）的核心思想很简单：**让小模型学大模型的本事**。
大模型（比如 70B）太贵、太慢，不适合大规模部署；小模型（比如 7B）快、便宜，
但能力不够。蒸馏就是在两者之间架一座桥。

但这里有一个很多人忽略的点：蒸馏不是让小模型"记住"大模型的答案，而是让它
**学会大模型的"思维方式"**。这个区别决定了蒸馏的效果。

如果你只是让小模型去模仿大模型的输出，它学到的是"在这个输入下应该输出什么"。
但如果你让小模型学习大模型在不同输入下的输出概率分布，它学到的就是
"为什么这么输出"——这是两种完全不同的学习效果。

蒸馏的工作流程
================

一个完整的蒸馏项目通常包含以下步骤：

.. code-block:: text

   1. 准备教师模型（大模型，如 GPT-4、Llama-3-70B）
   2. 准备训练数据（无标签或弱标签数据）
   3. 教师模型在数据上生成软标签（输出概率分布）
   4. 学生模型（小模型，如 Llama-3-8B）同时学习软标签和硬标签
   5. 评估学生模型的效果
   6. 如果效果不满意，调整温度 T 或 alpha 权重

.. code-block:: python

   def distillation_pipeline(
       teacher_model, student_model,
       unlabeled_data, T=2.0, alpha=0.5
   ):
       """完整的蒸馏流水线"""
       for batch in unlabeled_data:
           # 教师模型生成软标签
           with torch.no_grad():
               teacher_logits = teacher_model(batch)

           # 学生模型前向传播
           student_logits = student_model(batch)

           # 蒸馏损失
           loss = distillation_loss(
               student_logits, teacher_logits,
               batch["labels"], alpha=alpha, T=T
           )

           # 反向传播
           loss.backward()
           optimizer.step()

硬标签 vs 软标签
====================

标准的监督学习用的是**硬标签**——"这张图是猫，不是狗"。Loss 是预测和
硬标签之间的交叉熵。

但蒸馏用的是**软标签** （Soft Labels）。软标签不是"猫=1, 狗=0", 而是
"猫=0.9, 狗=0.08, 兔子=0.02"。软标签包含了**类间关系**——猫和狗比较像，
猫和兔子不太像。这种关系信息在大模型的知识结构中天然存在。

.. code-block:: python

   # 硬标签：
   #   "苹果" → [1, 0, 0]  # 只告诉你是苹果，没告诉你苹果和梨更接近
   #
   # 软标签（教师模型的输出）：
   #   "苹果" → [0.85, 0.10, 0.03, 0.02]
   #            苹果   梨   香蕉  橘子
   #   它隐含了：苹果比香蕉更接近梨

蒸馏的核心公式
====================

.. code-block:: python

   def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
       """
       蒸馏损失 = 硬标签损失 + 软标签损失

       T（温度）控制软标签的"平滑程度"。
       T 越大，概率分布越平滑，小模型学到的是 "类别之间的相对关系"。
       T 越小，越接近硬标签，学到的就是"标准的答案"。
       """
       # 硬标签损失：小模型必须学会"正确的答案"
       hard_loss = cross_entropy(student_logits, labels)

       # 软标签损失：小模型还要学会"大模型的思考方式"
       soft_labels = softmax(teacher_logits / T)
       soft_pred = softmax(student_logits / T)
       soft_loss = KL_divergence(soft_labels, soft_pred) * (T ** 2)

       # 加权组合
       return alpha * hard_loss + (1 - alpha) * soft_loss

温度 T 的作用经常被误解：

- T=1：基本就是老师输出的原始分布，差异被完整保留
- T=2 到 4：分布更平滑，类别间的细微差异被放大，小模型能学到更多"相对关系"
- T>5：分布过于平坦，所有类别的概率接近，信息丢失

经验值是 T=2 到 T=4 之间。对于 Agent 场景，建议从 T=2 开始尝试。

什么时候蒸馏有用？
====================

.. list-table::
   :header-rows: 1

   * - 场景
     - 蒸馏效果
     - 说明
   * - 分类/标注任务
     - 优秀
     - 软标签包含丰富的类间关系
   * - 生成任务
     - 良好
     - 蒸馏后的模型输出更流畅、更"像人"
   * - Agent 推理链
     - 中等
     - 可以蒸馏推理模式，但工具调用需要额外训练
   * - 工具调用
     - 有限
     - 工具调用的关键是格式准确性，蒸馏帮助不大

蒸馏 vs 量化
================

一个常见的混淆点：蒸馏和量化都在做"缩小模型"这件事，但思路完全不同。

.. table:: 蒸馏 vs 量化
   :widths: 20 40 40

   ==================  ========================================  ========================================
   维度                蒸馏                                      量化
   ==================  ========================================  ========================================
   核心理念            让小模型学习大模型的行为                    用更低精度表示权重
   是否需要训练        需要（重新训练小模型）                      不需要（后处理）
   参数量              减少（7B vs 70B）                          不变（只是精度降低）
   推理速度提升        5-10 倍                                    1.5-2 倍
   质量损失            取决于数据量和训练方式                      INT8 几乎无损失，INT4 轻微
   两者关系            可以组合使用（先蒸馏再量化）                |
   ==================  ========================================  ========================================

Agent 场景中蒸馏的特殊价值
============================

Agent 部署的最大成本不是 GPU 时长，而是**延迟和吞吐量**。一个 70B 模型
每步推理需要几百毫秒，对于需要多步推理的 Agent 来说，这个延迟会层层叠加。
蒸馏到一个 7B 或 8B 模型，推理速度可以提升 5-10 倍，成本降低到 1/10。

但代价是什么？一个被蒸馏的 Agent 在复杂推理、处理长上下文、应对未见过的
工具调用时，能力会下降。所以实践中常用的策略是**两阶段决策**：

.. code-block:: python

   class TieredAgent:
       """分层 Agent：简单任务用小模型，复杂任务用大模型"""
       def __init__(self, small_model, large_model, complexity_threshold=0.7):
           self.small = small_model  # 蒸馏后的模型
           self.large = large_model  # 原始大模型
           self.threshold = complexity_threshold

       def run(self, task: str) -> str:
           # 先让小模型评估任务复杂度
           complexity = self._estimate_complexity(task)

           if complexity < self.threshold:
               return self.small.run(task)  # 简单任务，小模型
           else:
               return self.large.run(task)  # 复杂任务，大模型

       def _estimate_complexity(self, task) -> float:
           prompt = f"评估任务复杂度（0-1）：{task}"
           return float(self.small.generate(prompt, temperature=0.0))

这种策略在实际项目中可以节省 60-80% 的推理成本，而用户体验几乎没有下降——
因为 80% 的用户请求是简单的。

.. admonition:: 蒸馏的企业级实践
   :class: tip

   如果要在生产环境中做蒸馏，建议：
   1. 先用 100-500 条数据做小规模实验，确认蒸馏方向是否正确
   2. 用教师模型生成大量（1万+）训练数据的软标签
   3. 在 Agent 的评估集上测试蒸馏后的模型，重点关注工具调用准确率
   4. 如果蒸馏后的模型在某些场景下表现不佳，针对这些场景补充训练数据
