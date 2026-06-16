.. _chapter-08-distillation:

===============================
模型蒸馏
===============================

知识蒸馏（Knowledge Distillation）的核心思想很简单：**让小模型学大模型的本事**。
大模型太贵、太慢，不适合大规模部署；小模型快、便宜，但能力不够。蒸馏就是在
两者之间架一座桥。

但这里有一个很多人忽略的点：蒸馏不是让小模型"记住"大模型的答案，而是让它
**学会大模型的"思维方式"**。这个区别决定了蒸馏的效果。

硬标签 vs 软标签
====================

标准的监督学习用的是**硬标签**——"这张图是猫，不是狗"。Loss 是预测和
硬标签之间的交叉熵。

但蒸馏用的是**软标签**（Soft Labels）。软标签不是"猫=1, 狗=0", 而是
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

温度 T 的作用经常被误解。T=1 时基本就是老师输出的原始分布。T>1 时分布
变得更加"平坦"，类别之间的细微差异被放大——这有助于小模型学到老师在做
"接近判断"时的细微差别。经验值是 T=2 到 T=4 之间。

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
               # 简单任务，小模型直接处理
               return self.small.run(task)
           else:
               # 复杂任务，交给大模型
               return self.large.run(task)

       def _estimate_complexity(self, task) -> float:
           prompt = f"评估任务复杂度（0-1）：{task}"
           return float(self.small.generate(prompt, temperature=0.0))

这种策略在实际项目中可以节省 60-80% 的推理成本，而用户体验几乎没有下降——
因为 80% 的用户请求是简单的。
