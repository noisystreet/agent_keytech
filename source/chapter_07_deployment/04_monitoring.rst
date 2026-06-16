.. _chapter-07-monitoring:

===============================
监控与可观测性
===============================

生产环境中，Agent 的可观测性比传统 API 更重要——Agent 的推理过程是黑箱，
必须通过日志和追踪来理解它的行为。

.. code-block:: python

   class Observability:
       """Agent 运行时的监控与追踪"""
       def trace(self, agent_run):
           return {
               "timestamp": agent_run.timestamp,
               "latency_ms": agent_run.latency_ms,
               "llm_calls": len(agent_run.llm_calls),
               "tool_calls": len(agent_run.tool_calls),
               "tokens_used": agent_run.total_tokens,
               "cost_usd": agent_run.total_tokens / 1_000_000 * model_price,
               "errors": agent_run.errors,
           }

       def log_thought_process(self, agent_run):
           """记录 Agent 的完整思考过程用于调试"""
           for step in agent_run.steps:
               logger.info(f"Step {step.n}: {step.thought}")
               if step.action:
                   logger.info(f"  Action: {step.action}")
                   logger.info(f"  Result: {step.observation[:200]}")
