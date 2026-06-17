.. _chapter-07-monitoring:

===============================
监控与可观测性
===============================

部署 Agent 之后，最让人头疼的问题不是"模型推理速度够不够快"，
而是**"我的 Agent 刚才到底干了什么？"**

Agent 的决策过程是一个黑箱——你给了一个输入，它经过多轮 Reasoning + Acting
循环，输出一个结果。但如果结果不对，你很难直接看出是哪一步出了问题。
是模型理解错了？工具选错了？API 返回了异常数据？还是推理链在某一步断掉了？

这就是 Agent 需要**可观测性**而非简单监控的原因。监控告诉你"出问题了"，
可观测性告诉你"哪里出了问题"。

Agent 的可观测性金字塔
=========================

.. mermaid::

   flowchart TD
       L1[Level 1: 指标<br>延迟 / Token 消耗 / 错误率] --> L2
       L2[Level 2: 日志<br>完整推理链 + 工具调用记录] --> L3
       L3[Level 3: 追踪<br>每步耗时 + 输入输出全链路] --> L4
       L4[Level 4: 回放<br>完全重现 Agent 的执行过程]

大多数团队只做到了 Level 1（看个延迟和错误率），但调试 Agent 问题
至少需要 Level 2，最好到 Level 3。

Level 1：指标监控
====================

.. code-block:: python

   class MetricsCollector:
       """Agent 运行时的基础指标"""
       def collect(self, agent_run):
           return {
               "latency_ms": agent_run.latency_ms,
               "llm_calls": len(agent_run.llm_calls),
               "tool_calls": len(agent_run.tool_calls),
               "tokens_used": agent_run.total_tokens,
               "cost_usd": agent_run.total_tokens / 1_000_000 * model_price,
               "errors": agent_run.errors,
               "success": agent_run.success,
           }

这些指标可以接入 Prometheus + Grafana。注意 **success 的定义**——
不是 Agent 没有抛异常就算成功，而是它完成了用户的任务。判断任务是否
完成需要额外的逻辑，比如检查最终回答是否包含了关键信息。

.. code-block:: python

   # Prometheus 指标定义示例
   from prometheus_client import Histogram, Counter, Gauge

   agent_latency = Histogram(
       "agent_run_duration_seconds",
       "Agent 执行耗时（秒）",
       buckets=[1, 2, 5, 10, 30, 60, 120]
   )
   agent_tokens = Counter(
       "agent_tokens_total",
       "Agent 消耗的 token 总数",
       ["model"]
   )
   agent_errors = Counter(
       "agent_errors_total",
       "Agent 执行错误数",
       ["error_type"]  # tool_timeout / llm_error / safety_blocked
   )
   active_agents = Gauge(
       "agent_active_runs",
       "当前正在执行的 Agent 数量"
   )

Level 2：日志记录
====================

.. code-block:: python

   class ThoughtLogger:
       """记录 Agent 的完整思考过程"""
       def __init__(self):
           self.logger = logging.getLogger("agent")

       def log_step(self, step):
           self.logger.info("=" * 40)
           self.logger.info(f"Step {step.n}:")
           self.logger.info(f"  Thought: {step.thought}")
           if step.action:
               self.logger.info(f"  Action: {step.action}")
               self.logger.info(f"  Params: {step.params}")
               self.logger.info(f"  Result: {step.observation}")
           self.logger.info(f"  Token cost: {step.tokens}")

**结构化日志** 是关键。不要用 ``print()`` 或简单的文本日志，
而是用 JSON 格式记录每步的结构化数据：

.. code-block:: python

   # 结构化日志比文本日志好用得多
   structured_log = {
       "step": 2,
       "thought": "需要搜索李飞飞的论文",
       "tool": "search",
       "tool_input": {"query": "李飞飞 2024 论文"},
       "tool_result_summary": "找到 15000 条结果...",
       "tool_latency_ms": 1200,
       "tokens_used": 456,
   }
   logger.info(json.dumps(structured_log))

这样你就可以用任何日志分析工具（ELK、Datadog）搜索和聚合。比如"查询所有
tool_latency_ms > 5000 的步骤"——这在文本日志里几乎不可能。

Level 3：全链路追踪
====================

.. code-block:: python

   class TraceCollector:
       """生成 Agent 执行的可视化追踪"""
       def generate_trace(self, agent_run):
           trace = []
           for step in agent_run.steps:
               trace.append({
                   "span_id": f"step-{step.n}",
                   "parent_id": f"step-{step.n-1}" if step.n > 0 else None,
                   "name": step.action or "思考",
                   "start_time": step.start_time,
                   "end_time": step.end_time,
                   "status": "ok" if not step.error else "error",
                   "error": step.error,
               })
           return trace

全链路追踪的价值在于你能看到**时间都花在哪里了**。我见过一个 Agent 项目，
追踪后发现 80% 的时间花在了一个特定的工具调用上（某个外部 API 响应慢），
而模型推理只占了 15%。如果没有追踪，团队可能一直在优化模型推理速度——
完全抓错了方向。

.. code-block:: python

   # 集成 OpenTelemetry 进行分布式追踪
   from opentelemetry import trace
   from opentelemetry.trace import StatusCode

   tracer = trace.get_tracer("agent")

   class TracedAgent:
       """带 OpenTelemetry 追踪的 Agent"""
       def run(self, task: str) -> str:
           with tracer.start_as_current_span("agent.run") as span:
               span.set_attribute("task", task)
               result = self._execute(task)

               if result["success"]:
                   span.set_status(StatusCode.OK)
               else:
                   span.set_status(StatusCode.ERROR, result["error"])

               for step in result["steps"]:
                   self._trace_step(step)

               return result["output"]

       def _trace_step(self, step):
           with tracer.start_as_current_span(f"step.{step.tool or 'think'}") as span:
               span.set_attribute("step", step.n)
               span.set_attribute("tool", step.tool or "none")
               span.set_attribute("latency_ms", step.latency_ms)
               if step.error:
                   span.set_status(StatusCode.ERROR, step.error)

Level 4：执行回放
====================

这是最高级别的可观测性——记录 Agent 的完整输入输出和状态变更，
在需要时可以**完全重现**它的执行过程。

对于 Agent 来说，一个实用的回放格式是"可视化的步骤流"：

.. code-block:: text

   用户输入：帮我查一下李飞飞最近的研究方向
   ┌─────────────────────────────────────────────┐
   │ Step 1: 搜索 "李飞飞 2024 研究"             │
   │ 结果：找到 Stanford 个人页面                 │
   │ 耗时：1.2s                                  │
   ├─────────────────────────────────────────────┤
   │ Step 2: 打开页面，提取研究方向               │
   │ 结果：空间智能、具身 AI                      │
   │ 耗时：2.5s                                  │
   ├─────────────────────────────────────────────┤
   │ Step 3: 综合回答                             │
   │ 耗时：3.1s                                  │
   ├─────────────────────────────────────────────┤
   │ 总计：6.8s  | 3 步 | 2 次工具调用 | 1520 tokens │
   └─────────────────────────────────────────────┘

告警规则设计
================

Agent 的告警规则和传统 API 不同，需要关注 Agent 特有的信号。

.. code-block:: python

   def alert_rules():
       """Agent 场景的关键告警规则"""
       return [
           {
               "name": "tool_timeout_high",
               "condition": "avg(tool_latency_ms) > 10000",  # 10 秒
               "severity": "warning",
               "message": "工具调用平均耗时超过 10 秒",
           },
           {
               "name": "agent_loop_exceeded",
               "condition": "avg(agent_steps) > 8",
               "severity": "warning",
               "message": "Agent 平均执行步数超过 8 步，可能存在死循环",
           },
           {
               "name": "cost_spike",
               "condition": "sum(agent_cost_usd) > 100",
               "severity": "critical",
               "message": "Agent 日消耗超过 $100",
           },
           {
               "name": "success_rate_drop",
               "condition": "avg(success_rate) < 0.7",
               "severity": "critical",
               "message": "Agent 任务成功率低于 70%",
           },
       ]

.. admonition:: 监控的迭代节奏
   :class: tip

   - **上线第一天**：Level 1 指标监控（延迟、错误率、Token 消耗）
   - **上线第一周**：加到 Level 2（完整日志），排查初始问题
   - **稳定运行后**：加到 Level 3（可观测追踪），做性能优化
   - **需要调试复杂问题**：用到 Level 4（回放），重现极端场景

   不要一开始就上 Level 4。可观测性是"够用就好"——加多了反而增加
   存储成本和系统复杂度。先从 Level 1 开始，遇到问题时再升级。
