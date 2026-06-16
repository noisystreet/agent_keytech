.. _chapter-08-rlhf:

===============================
RLHF 与偏好对齐
===============================

RLHF（Reinforcement Learning from Human Feedback）是大模型对齐的核心技术。
ChatGPT 为什么比其他模型"听话"？最重要的原因就是 RLHF。

但 RLHF 经常被误解。很多人以为它只是"用强化学习再训练一下"，其实 RLHF
是一个**三阶段的系统**，每个阶段解决不同的问题。

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
这一步做完后，模型已经能生成不错的回答。

阶段 2：训练奖励模型（Reward Model）
------------------------------

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
       # ...
   ]

   # 奖励模型学习区分"好回答"和"差回答"
   def train_reward_model(model, data):
       for item in data:
           chosen_score = model.score(item["prompt"], item["chosen"])
           rejected_score = model.score(item["prompt"], item["rejected"])
           # loss 让 chosen 的分数高于 rejected
           loss = -log(sigmoid(chosen_score - rejected_score))
           # 反向传播

这里的关键洞察：奖励模型并不直接告诉你"什么是最好的回答"，
它只是告诉你"这个回答比那个好"。只要比较关系足够多，奖励模型就能
学会人类偏好的边界。

阶段 3：PPO 优化
------------------------------

用强化学习让策略模型（Policy Model）**对齐**奖励模型的打分。

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
       loss += beta * kl_penalty  # KL 散度惩罚
       return loss.mean()

PPO 中有一个容易被忽视的设计：**KL 散度惩罚**。如果没有它，模型为了
最大化奖励可能会"作弊"——生成语法上正确但语义上有问题的内容来骗取高分。
KL 惩罚让模型不能离初始版本太远，就像"你不能为了讨好老师而胡说八道"。

RLHF 的局限
================

RLHF 不是万能的，而且有几个明显的问题：

.. admonition:: RLHF 的三大问题
   :class: caution

   1. **奖励黑客（Reward Hacking）**：模型找到奖励模型的漏洞，用看似合规
      但实际无意义的内容骗取高分
   2. **多样性丧失**：对齐后的模型倾向于输出"安全但平庸"的内容，
      创造性下降
   3. **标注偏差**：奖励模型反映的是标注者的偏好，如果标注者群体单一，
      模型就会有偏见

Agent 场景中的 RLHF 变体
============================

Agent 的 RLHF 和纯语言模型的 RLHF 有一个重大区别：Agent 不仅要优化"说什么"，
还要优化"做什么"。

.. code-block:: python

   # Agent 的 RLHF：不仅要看回答质量，还要看行为质量
   class AgentRLHF:
       def __init__(self, policy_agent, reward_model):
           self.policy = policy_agent  # 正在训练的 Agent
           self.reward = reward_model  # 奖励模型

       def compute_agent_reward(self, task, result):
           """Agent 的综合奖励 = 结果质量 + 效率 + 安全"""
           # 结果质量
           quality = self.reward.score(task, result["answer"])

           # 执行效率（鼓励用更少的步骤完成任务）
           efficiency = 1.0 / (1.0 + result["steps"])
           # 如果最优是 2 步，用了 5 步，效率 = 1/(1+5) = 0.17

           # 安全合规（是否触发了安全规则）
           safety = 1.0 if not result["safety_violations"] else 0.0

           return 0.6 * quality + 0.2 * efficiency + 0.2 * safety

这个综合奖励函数的设计值得注意的三个权重：
- **0.6 结果质量**：最终答案正确最重要
- **0.2 效率**：但也不能绕圈子。用了不必要步骤要扣分
- **0.2 安全**：错了可以接受，但违规不可接受

RLHF vs DPO
============

2024 年以来，DPO（Direct Preference Optimization）逐渐流行，因为它
**不需要训练单独的奖励模型**，直接在偏好数据上优化策略。

.. list-table::
   :header-rows: 1

   * - 对比
     - RLHF
     - DPO
   * - 需要训练奖励模型
     - 是（多一个阶段）
     - 否（端到端）
   * - 实现复杂度
     - 高
     - 低
   * - 训练稳定性
     - 需要精细调参
     - 相对稳定
   * - 效果上限
     - 理论上更高
     - 受限于偏好数据质量
   * - 适合场景
     - 有标注团队、追求极致效果
     - 快速实验、资源有限

选择建议：如果你有标注团队和计算资源，RLHF 效果上限更高；如果你是个人开发者
或小团队，DPO 更容易上手，效果也足够好。
