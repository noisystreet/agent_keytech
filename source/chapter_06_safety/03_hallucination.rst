.. _chapter-06-hallucination:

===============================
幻觉缓解
===============================

幻觉是 LLM 生成不真实或不合逻辑内容的倾向。在 Agent 场景中，幻觉的危害
被工具执行能力放大——一个"幻觉"出来的 API 调用可能造成真实的影响，
比如错误的数据操作或误导性的决策。

Agent 场景中的三类幻觉
==========================

.. list-table::
   :header-rows: 1

   * - 幻觉类型
     - 表现
     - Agent 场景中的后果
   * - 事实性幻觉
     - 编造不存在的事实
     - 用户得到错误信息，基于错误信息做决策
   * - 工具幻觉
     - 编造不存在的工具或参数
     - 工具调用失败，或调用了错误的 API
   * - 推理幻觉
     - 推理链条中有不成立的中间结论
     - 多步任务路径偏离，浪费 Token 和时间

缓解策略全景
================

1. RAG 强约束
------------------------------

要求 Agent 必须引用检索到的事实，不推理、不推测。

.. code-block:: python

   class RAGFactGuard:
       def __init__(self, retriever, threshold=0.8):
           self.retriever = retriever
           self.threshold = threshold

       def verify(self, statement: str, context_docs: list) -> dict:
           """验证陈述是否被检索到的文档支持"""
           # 将陈述拆分为可验证的声明
           claims = self._extract_claims(statement)

           verdicts = []
           for claim in claims:
               # 在上下文中搜索支持证据
               evidence = self._search_evidence(claim, context_docs)
               is_supported = evidence["score"] > self.threshold
               verdicts.append({
                   "claim": claim,
                   "supported": is_supported,
                   "evidence": evidence["text"],
                   "confidence": evidence["score"]
               })

           return {
               "all_supported": all(v["supported"] for v in verdicts),
               "details": verdicts
           }

       def _extract_claims(self, text) -> list:
           return [s.strip() for s in text.split("。") if len(s) > 5]

       def _search_evidence(self, claim, docs) -> dict:
           # 简化实现：检索最相关的文档片段
           return {"score": 0.9, "text": docs[0] if docs else ""}

2. 自我校验
------------------------------

Agent 在提交最终结果前，重新验证自己的回答。

.. code-block:: python

   class SelfValidator:
       def __init__(self, llm, max_retries=2):
           self.llm = llm
           self.max_retries = max_retries

       def validate(self, agent_output: str) -> str:
           """自我校验并修正"""
           for attempt in range(self.max_retries):
               check_prompt = f"""
               对以下回答进行事实核查：

               回答：{agent_output}

               请指出：
               1. 任何可能不准确的陈述
               2. 缺乏依据的推断
               3. 需要修正的地方

               如果发现错误，请给出修正版本。
               """
               review = self.llm.generate(check_prompt, temperature=0.0)
               if "未发现错误" in review or "没有错误" in review:
                   return agent_output
               # 根据审查意见修正
               agent_output = self.llm.generate(
                   f"原始回答：{agent_output}\n审查意见：{review}\n请修正："
               )
           return agent_output

3. 工具约束：所有信息必须通过工具获得
----------------------------------------

强制 Agent 在回答中标注信息来源。

.. code-block:: python

   class ToolSourcedOnly:
       """强制所有回答必须来自工具调用结果"""
       def run(self, agent, task: str) -> dict:
           response = agent.run(task)
           # 解析回答中的声明及其来源
           claims = self._parse_claims_with_sources(response)
           unsourced = [c for c, s in claims if not s]
           if unsourced:
               return {
                   "answer": response,
                   "warning": f"以下声明缺乏工具来源：{unsourced}",
                   "unsourced_count": len(unsourced)
               }
           return {"answer": response, "warning": None}

       def _parse_claims_with_sources(self, text) -> list:
           # 检查是否包含 [来源: tool_name] 标记
           return [(line, "[来源" in line) for line in text.split("\n") if line]

4. 置信度校准
------------------------------

让 LLM 对自己回答的不确定性给出量化估计。

.. code-block:: python

   class ConfidenceCalibrator:
       def __init__(self, llm):
           self.llm = llm

       def answer_with_confidence(self, question: str) -> dict:
           response = self.llm.generate(f"""
               回答问题，并在回答后给出你的置信度评分（0-100）：

               问题：{question}

               格式：
               回答：...
               置信度：XX/100
               理由：...
           """)
           confidence = self._extract_confidence(response)

           if confidence < 60:
               # 置信度过低，触发额外验证
               verification = self.llm.generate(
                   f"请再次确认以下回答的准确性：\n{response}"
               )
               return {
                   "answer": response,
                   "confidence": confidence,
                   "verified": True,
                   "verification": verification
               }
           return {"answer": response, "confidence": confidence, "verified": False}

5. 多路径校验
------------------------------

通过对比多路推理结果的一致性来发现幻觉。

.. code-block:: python

   class MultiPathValidator:
       def __init__(self, llm, n_paths=3):
           self.llm = llm
           self.n_paths = n_paths

       def validate(self, question: str, candidate: str) -> dict:
           """用多条推理路径验证候选答案"""
           paths = []
           for i in range(self.n_paths):
               # 用不同的温度采样生成不同推理路径
               response = self.llm.generate(
                   f"问题：{question}\n请用不同的方法验证以下答案：\n{candidate}",
                   temperature=0.5 + i * 0.2
               )
               paths.append(response)

           # 检查各路径是否一致
           consistency = self._check_consistency(paths)
           return {
               "consistent": consistency > 0.7,
               "consistency_score": consistency,
               "paths": paths
           }

       def _check_consistency(self, paths) -> float:
           # 简化实现：计算语义相似度
           return 0.85

缓解策略选择矩阵
================

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐策略
     - 额外成本
   * - 事实问答
     - RAG 强约束
     - 检索延迟
   * - 多步推理
     - 自我校验
     - 额外 1-2 次 LLM 调用
   * - 工具调用
     - 工具约束
     - 几乎无成本
   * - 高风险决策
     - 置信度校准 + 多路径校验
     - 2-5 倍 LLM 调用
   * - 日常 Agent 任务
     - 组合使用（校验 + 约束）
     - 约 30% 额外成本

.. admonition:: 幻觉不可能完全消除
   :class: warning

   当前的技术水平下，幻觉只能缓解，无法根除。建议：
   1. **分层防护**：低成本策略（工具约束）+ 高成本策略（多路径校验）组合
   2. **风险适配**：风险越高的操作，启用更严格的校验
   3. **用户透明**：标记不确定的回答，让用户自行判断
   4. **持续监控**：上线后持续跟踪 Agent 的回答准确性

参考文献
============

- Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection", 2023
- Ji et al., "Survey of Hallucination in Natural Language Generation", 2023
