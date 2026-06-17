.. _chapter-08-rlhf:

===============================
RLHF 与偏好对齐
===============================

RLHF（Reinforcement Learning from Human Feedback）是大模型对齐的核心技术。
ChatGPT 为什么比其他模型"听话"——比如不会帮你写钓鱼邮件、不会辱骂用户、
不会编造明显的事实？最重要的原因就是 RLHF。

但 RLHF 经常被误解。很多人以为它只是"用强化学习再训练一下"，其实 RLHF
是一个**三阶段的系统** ，每个阶段解决不同的问题。只理解其中一两个阶段，
就无法真正理解它为什么有效、为什么难做。

.. mermaid::

   flowchart LR
       Data[人类偏好数据] --> Train[训练奖励模型]
       Train --> RM[奖励模型]
       RM --> PPO[PPO 优化<br>最大化奖励]
       PPO --> Policy[策略模型]
       Policy --> RM

三阶段详解
================

阶段 1：SFT（监督微调）
------------------------------

先让模型学会"好的回答长什么样"。用人工标注的高质量对话数据做标准有监督训练。
这一步做完后，模型已经能生成不错的回答——但它的回答风格是"模仿"出来的，
不一定是"用户想要的"。

阶段 2：训练奖励模型（Reward Model）
----------------------------------------

这一步很关键。你让模型对一个 prompt 生成多个回答，然后让人工标注"哪个更好"。
奖励模型就是一个打分器——给定 prompt + 回答，输出一个分数。

.. code-block:: python

   # 奖励模型的训练数据（人工标注的偏好对）
   preference_data = [
       {
           "prompt": "帮我写一封求职邮件",
           "chosen": "尊敬的HR您好...",          # 人类更喜欢的回答
           "rejected": "嗨，听说你们在招人？"      # 人类不太喜欢的回答
       },
   ]

   def train_reward_model(model, data):
       """
       奖励模型用 Bradley-Terry 模型来建模偏好。
       核心思路：让 chosen 的分数尽可能高于 rejected。
       """
       for item in data:
           chosen_score = model.score(item["prompt"], item["chosen"])
           rejected_score = model.score(item["prompt"], item["rejected"])
           # loss = -log(sigmoid(chosen - rejected))
           # 当 chosen_score >> rejected_score 时，loss 接近 0
           # 当 chosen_score < rejected_score 时，loss 很大
           loss = -log(sigmoid(chosen_score - rejected_score))
           loss.backward()

奖励模型训练的难点在于**一致性**。两个标注员对"哪个回答更好"可能有分歧，
这会造成奖励模型的训练信号互相矛盾。实践中通常需要：
1. 详细的标注指南（什么算"好"、什么算"安全"）
2. 每个标注样本由多人标注，取共识
3. 定期检查标注一致性

阶段 3：PPO 优化
------------------------------

用强化学习让策略模型（Policy Model）对齐奖励模型的打分。

.. code-block:: python

   def ppo_loss(old_logprobs, new_logprobs, rewards, kl_penalty):
       """
       PPO 的核心思想：更新策略时不要偏离太多。

       - rewards: 奖励模型给出的分数
       - kl_penalty: 防止模型"跑偏"的约束项
       - ratio: 新策略/旧策略的概率比
       - clipped: 限制更新幅度，防止一步迈太大
       """
       ratio = exp(new_logprobs - old_logprobs)
       clipped = clamp(ratio, 1-epsilon, 1+epsilon)
       loss = -min(ratio * rewards, clipped * rewards)
       loss += beta * kl_penalty
       return loss.mean()

PPO 中有一个容易忽视的关键设计：**KL 散度惩罚**。

为什么需要它？想象一个模型发现"只要说某些固定句式就能拿到高分"，于是
不管用户问什么，它都输出"这是一个很好的问题，让我为您解答……"来骗取
奖励模型的高分。KL 惩罚约束模型的输出分布不能偏离原始版本太远，
防止它"投机取巧"。

Agent 场景的 RLHF 设计
============================

Agent 的 RLHF 和纯语言模型的 RLHF 有一个重大区别：Agent 不仅要优化"说什么"，
还要优化"做什么"。一个 Agent 的"好行为"包含多个维度：

.. code-block:: python

   class AgentRLHF:
       """
       Agent 的 RLHF：不仅要看回答质量，还要看行为质量。
       """
       def __init__(self, policy_agent, reward_model):
           self.policy = policy_agent
           self.reward = reward_model

       def compute_agent_reward(self, task, result):
           """
           Agent 的综合奖励 = 结果质量 + 效率 + 安全 + 工具使用

           多目标奖励设计的核心挑战：如何平衡不同目标？
           - 质量最重要但不好量化
           - 效率容易量化但可能牺牲质量
           - 安全是硬约束但不能过于保守
           """
           # 结果质量（LLM as Judge 打分）
           quality = self.reward.score(task, result["answer"])

           # 执行效率（鼓励用更少的步骤）
           efficiency = 1.0 / (1.0 + result["steps"])

           # 安全合规
           safety = 1.0 if not result["safety_violations"] else 0.0

           # 工具使用质量（是否在正确时机调用了正确工具）
           tool_quality = self._evaluate_tool_usage(result["tool_calls"])

           return (0.4 * quality + 0.15 * efficiency +
                   0.25 * safety + 0.2 * tool_quality)

       def _evaluate_tool_usage(self, tool_calls):
           """评估工具调用的正确性"""
           if not tool_calls:
               return 0.5  # 没有工具调用也算中立
           correct = sum(1 for call in tool_calls if call["success"])
           return correct / len(tool_calls)

这个综合奖励函数的设计体现了 Agent 场景的特殊性：

.. list-table::
   :header-rows: 1

   * - 维度
     - 权重
     - 为什么这么设
     - 如果权重过高会怎样
   * - 结果质量
     - 0.4
     - 最核心的目标
     - Agent 为了"好看"而编造答案
   * - 安全
     - 0.25
     - 安全违规是不可接受的
     - Agent 过于保守，拒绝合理请求
   * - 工具使用
     - 0.2
     - Agent 的核心能力
     - 过度使用工具，为用而用
   * - 效率
     - 0.15
     - 不要浪费 token
     - 过于简略，跳过必要步骤

DPO：不需要奖励模型的替代方案
================================

2024 年以来，DPO（Direct Preference Optimization）逐渐流行。它的核心创新是：
**不需要单独训练一个奖励模型** ，直接在偏好数据上优化策略。

.. code-block:: python

   def dpo_loss(policy_logprobs, ref_logprobs, chosen, rejected, beta=0.1):
       """
       DPO 的核心公式。
       它直接在偏好数据上优化，不需要经过奖励模型。

       直观理解：
       - 让模型在"chosen"回答上的概率增加
       - 让模型在"rejected"回答上的概率减少
       - beta 控制变化的幅度
       """
       # 计算 chosen 和 rejected 的对数概率比
       chosen_ratio = policy_logprobs[chosen] - ref_logprobs[chosen]
       rejected_ratio = policy_logprobs[rejected] - ref_logprobs[rejected]

       # DPO 损失
       loss = -log(sigmoid(beta * (chosen_ratio - rejected_ratio)))
       return loss.mean()

.. list-table::
   :header-rows: 1

   * - 对比
     - RLHF
     - DPO
   * - 需要训练奖励模型
     - 是（多一个阶段，增加训练复杂度）
     - 否（端到端，简化流程）
   * - 实现复杂度
     - 高（需要管理多个模型和训练循环）
     - 低（只需策略模型和参考模型）
   * - 训练稳定性
     - 需要精细调参，PPO 容易不稳定
     - 相对稳定，收敛更容易
   * - 计算资源
     - 需要额外存储和训练奖励模型
     - 节省奖励模型的开销
   * - 效果上限
     - 理论上更高（奖励模型可以持续优化）
     - 受限于偏好数据质量
   * - 适合场景
     - 有标注团队、追求极致效果
     - 快速实验、资源有限

RLHF 的工程陷阱
=================

.. admonition:: 陷阱：奖励模型过拟合
   :class: caution

   奖励模型训练到后期会在"简单样本"上过拟合，即它对大多数 prompt 给出
   接近的分数，丧失了区分能力。解决方案：定期用验证集检查奖励模型的
   **准确率**——如果准确率接近 100%，说明过拟合了，需要更难的样本。

.. admonition:: 陷阱：PPO 的 KL 灾难
   :class: caution

   PPO 中 KL 惩罚项的系数 beta 很敏感。beta 太大，模型几乎不变；
   beta 太小，模型只顾追求高分而偏离原始能力。实用建议：
   先固定 beta=0.1，观察 KL 散度的变化趋势，如果 KL 增长过快则增大 beta。

.. admonition:: 陷阱：Agent 的稀疏奖励
   :class: caution

   Agent 任务往往是"成功/失败"的二元结果，中间步骤没有明确奖励。
   这导致奖励信号非常稀疏，PPO 难以有效学习。缓解方案：
   1. 对中间步骤也设置子目标奖励
   2. 用行为克隆（BC）初始化，再用 RLHF 微调
