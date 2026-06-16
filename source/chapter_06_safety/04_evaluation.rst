.. _chapter-06-evaluation:

===============================
评估体系
===============================

Agent 的评估比传统模型的评估更复杂——不仅评估最终回答，还要评估工具调用、
推理过程和整体效率。

评估维度
============

.. list-table::
   :header-rows: 1

   * - 维度
     - 指标
     - 方法
   * - 任务完成率
     - Success@K
     - 判断最终结果是否正确
   * - 工具使用效率
     - 调用次数、正确率
     - 统计工具调用的必要性和正确性
   * - 推理质量
     - 推理链一致性
     - 检查推理步骤是否合乎逻辑
   * - 鲁棒性
     - 面对干扰的稳定性
     - 修改问题措辞观察效果

.. code-block:: python

   def evaluate_agent(agent, test_suite):
       results = {}
       for task, expected, tools in test_suite:
           response = agent.run(task)
           results[task] = {
               "correct": response.answer == expected,
               "tool_calls": len(response.tool_calls),
               "steps": response.steps,
               "latency": response.latency,
           }
       return results
