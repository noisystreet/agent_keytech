.. _chapter-08-rlhf:

===============================
RLHF 与偏好对齐
===============================

RLHF（Reinforcement Learning from Human Feedback）是使 LLM 与人类偏好对齐的核心技术。
对于 Agent 场景，RLHF 可以优化 Agent 的"行为"而不仅是"语言"。

.. mermaid::

   flowchart LR
       Data[人类偏好数据] --> Train[训练奖励模型]
       Train --> RM[奖励模型]
       RM --> PPO[PPO 优化]
       PPO --> Policy[策略模型]
       Policy --> RM

.. code-block:: python

   # RLHF 的核心损失函数（简化）
   def ppo_loss(old_logprobs, new_logprobs, rewards, kl_penalty):
       ratio = exp(new_logprobs - old_logprobs)
       clipped = clamp(ratio, 1-epsilon, 1+epsilon)
       loss = -min(ratio * rewards, clipped * rewards)
       loss += beta * kl_penalty  # KL 散度惩罚
       return loss.mean()
