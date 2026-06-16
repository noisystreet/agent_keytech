.. _chapter-02-evaluation:

===============================
Agent 评估体系
===============================

评估是 Agent 开发的"指南针"。没有评估，优化就无从谈起。Agent 系统的评估
远比传统 NLP 任务复杂——不仅要衡量回答质量，还要评估工具调用、多步规划、
成本效率等多个维度。

评估金字塔
==============

.. mermaid::

   flowchart TD
       subgraph Pyramid [Agent 评估金字塔]
           T1[任务成功率] --- T2[执行效率]
           T2 --- T3[鲁棒性]
           T3 --- T4[安全性]
           T4 --- T5[成本控制]
       end

.. list-table::
   :header-rows: 1

   * - 维度
     - 指标
     - 衡量什么
   * - 任务成功率
     - 完成率、准确率
     - Agent 能否正确完成任务
   * - 执行效率
     - 步数、延迟、Token 消耗
     - Agent 完成任务是否高效
   * - 鲁棒性
     - 异常处理率、重试率
     - 面对错误和异常能否恢复
   * - 安全性
     - 越狱成功率、敏感信息泄露率
     - Agent 是否遵循安全约束
   * - 成本控制
     - 每次调用的 Token 数、API 费用
     - Agent 运行的经济性

核心评估指标
==================

1. 任务成功率
------------------

.. code-block:: python

   def evaluate_success_rate(agent, test_cases: list) -> dict:
       """评估 Agent 的任务完成情况"""
       results = {"success": 0, "partial": 0, "failed": 0, "total": len(test_cases)}

       for case in test_cases:
           # 执行 Agent
           output = agent.run(case["input"])

           # 检查是否满足预期
           if output == case["expected_output"]:
               results["success"] += 1
           elif case["validator"](output, case["expected_output"]):
               results["partial"] += 1
           else:
               results["failed"] += 1

       results["success_rate"] = results["success"] / results["total"]
       return results

2. 执行效率
------------------

.. code-block:: python

   def evaluate_efficiency(agent, test_cases: list) -> dict:
       """评估 Agent 执行效率"""
       total_steps = 0
       total_tokens = 0
       total_latency = 0.0

       for case in test_cases:
           result = agent.run_with_trace(case["input"])
           total_steps += result.steps
           total_tokens += result.total_tokens
           total_latency += result.latency

       n = len(test_cases)
       return {
           "avg_steps": total_steps / n,
           "avg_tokens": total_tokens / n,
           "avg_latency": total_latency / n,
       }

3. 鲁棒性
------------------

.. code-block:: python

   def evaluate_robustness(agent, test_cases: list) -> dict:
       """评估 Agent 的鲁棒性"""
       recovery_count = 0
       hallucination_count = 0

       for case in test_cases:
           # 注入干扰：工具返回异常、超时等
           result = agent.run_with_errors(case["input"])
           if result.recovered:
               recovery_count += 1
           if result.hallucination:
               hallucination_count += 1

       return {
           "recovery_rate": recovery_count / len(test_cases),
           "hallucination_rate": hallucination_count / len(test_cases),
       }

测试数据集构建
==================

高质量的评估需要覆盖全面的测试场景：

.. list-table::
   :header-rows: 1

   * - 测试类型
     - 说明
     - 示例
   * - 标准任务
     - 典型用户请求
     - "帮我查一下天气"
   * - 边界情况
     - 输入异常或极端
     - 空输入、超长输入
   * - 错误恢复
     - 工具调用失败
     - API 超时、返回空结果
   * - 安全测试
     - 越狱或注入攻击
     - "忽略之前的指令..."
   * - 多轮对话
     - 连续交互
     - 需要记忆上下文的任务

评估流程自动化
==================

.. code-block:: python

   class AgentBenchmark:
       def __init__(self, agent, test_suite: dict):
           self.agent = agent
           self.test_suite = test_suite  # {category: [test_cases]}

       def run_all(self) -> dict:
           """运行全量评估"""
           report = {}

           for category, cases in self.test_suite.items():
               report[category] = {
                   "success": evaluate_success_rate(self.agent, cases),
                   "efficiency": evaluate_efficiency(self.agent, cases),
                   "robustness": evaluate_robustness(self.agent, cases),
               }

           return report

       def regression_check(self, baseline: dict, threshold=0.05) -> bool:
           """回归检查：新版本是否比基线差"""
           current = self.run_all()

           for category in current:
               for metric in current[category]:
                   if current[category][metric]["success_rate"] < \
                      baseline[category][metric]["success_rate"] - threshold:
                       print(f"⚠️ {category}/{metric} 显著下降！")
                       return False
           return True

.. admonition:: 评估的迭代节奏
   :class: tip

   建议：
   - **每次修改后**：运行回归测试（5-10 个核心场景，30 秒内完成）
   - **每周**：运行全量评估（50-100 个场景，覆盖所有维度）
   - **每迭代**：根据评估结果分析瓶颈，确定下一轮优化方向

   记住：**"不评估就优化，等于蒙眼开车。"**

参考文献
============

- Liang et al., "Holistic Evaluation of Language Models", 2022
- Liu et al., "AgentBench: Evaluating LLMs as Agents", 2023
