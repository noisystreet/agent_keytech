.. _chapter-08-distillation:

===============================
模型蒸馏
===============================

知识蒸馏（Knowledge Distillation）将大模型（教师）的知识迁移到小模型（学生），
在不显著降低性能的前提下减小模型体积和推理成本。

.. code-block:: python

   # 蒸馏的核心：学生在教师输出的分布上学习
   def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
       # 硬标签损失（标准交叉熵）
       hard_loss = cross_entropy(student_logits, labels)

       # 软标签损失（KL 散度）
       soft_labels = softmax(teacher_logits / T)
       soft_pred = softmax(student_logits / T)
       soft_loss = KL_divergence(soft_labels, soft_pred) * (T ** 2)

       # 加权组合
       return alpha * hard_loss + (1 - alpha) * soft_loss
