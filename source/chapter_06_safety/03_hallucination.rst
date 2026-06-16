.. _chapter-06-hallucination:

===============================
幻觉缓解
===============================

幻觉是 LLM 生成不真实或不合逻辑内容的倾向。在 Agent 场景中，幻觉的危害
被工具执行能力放大——一个"幻觉"出来的 API 调用可能造成真实的影响。

缓解策略
============

- **RAG 强约束**：要求 Agent 必须引用检索到的事实，不推理、不推测
- **自我校验**：Agent 在提交前重新验证自己的回答
- **工具约束**：所有信息必须通过工具获得，不允许"凭空生成"

.. code-block:: python

   def run_with_fact_checking(agent, task):
       # 首轮：正常回答
       answer = agent.run(task)

       # 校验轮：验证每个声明的来源
       check_prompt = f"""
       对于以下回答，指出每个声明的可信度（Verified/Unverified）：
       回答：{answer}
       如果存在 Unverified 声明，请修正。
       """
       verified = agent.run(check_prompt)
       return verified
