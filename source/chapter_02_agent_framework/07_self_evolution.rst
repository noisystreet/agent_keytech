.. _chapter-02-self-evolution:

===============================
自进化能力
===============================

传统的 Agent 是无状态的——每次交互从零开始推理，不会从错误中学习。
**自进化能力** 是 Agent 突破这一局限的关键跃迁：让 Agent 能随着使用
积累经验、沉淀技能、持续改进。

为什么需要自进化？
====================

.. list-table::
   :header-rows: 1

   * - 维度
     - 传统 Agent
     - 自进化 Agent
   * - 错误处理
     - 每次犯同样的错
     - 犯错一次，记录规则，永不再犯
   * - 技能积累
     - 依赖开发者编写工具
     - 自动从经验中沉淀技能
   * - 个性化
     - 无状态，每次重新了解用户
     - 跨会话记忆用户偏好
   * - 优化效率
     - 需要手动调参和改 prompt
     - 自动评估并调整策略
   * - 团队复用
     - 知识只存在于个人经验
     - 技能可分享、可复用、可改进

三种进化范式
================

.. mermaid::

   flowchart LR
       subgraph Paradigm [三种进化范式]
           direction LR
           C[纠正型<br>反馈→规则] --> O[优化型<br>评估→调整]
           O --> A[自主型<br>经验→技能→复用]
       end

1. 纠正型进化：从反馈中学习
------------------------------

核心思路：当 Agent 犯错时，自动生成纠正规则，防止相同错误再次发生。

.. code-block:: python

   class CorrectiveLearner:
       def __init__(self):
           self.rules = []  # 积累的行为规则

       def on_error(self, task: str, action: str, error: str):
           """任务出错时，自动生成纠正规则"""
           rule = self._generate_rule(task, action, error)
           self.rules.append(rule)
           return rule

       def _generate_rule(self, task, action, error) -> str:
           """用 LLM 从错误中提炼规则"""
           prompt = f"""
           任务：{task}
           执行的操作：{action}
           错误：{error}

           请提炼一条规则，避免下次犯相同错误。
           规则格式：当 [条件] 时，必须 [怎么做]。
           """
           return llm.generate(prompt)

       def get_system_prompt(self) -> str:
           """将积累的规则注入 System Prompt"""
           rules_text = "\n".join(f"- {r}" for r in self.rules)
           return f"以下是你从经验中积累的规则：\n{rules_text}"

2. 优化型进化：评估-调整循环
------------------------------

通过持续评估自身表现，自动调整策略和参数。

.. code-block:: python

   class OptimizingLearner:
       def __init__(self, llm):
           self.llm = llm
           self.strategies = {}  # {task_type: best_strategy}

       def after_action(self, task: str, result: dict):
           """完成任务后，评估并优化"""
           task_type = self._classify(task)

           # 评估成功度
           score = self._evaluate(result)

           # 对比历史最佳策略
           best = self.strategies.get(task_type, {"score": 0})

           if score > best["score"]:
               # 新策略更优，生成策略描述并保存
               strategy = self._extract_strategy(task, result)
               self.strategies[task_type] = {
                   "strategy": strategy,
                   "score": score
               }

       def _evaluate(self, result: dict) -> float:
           """评估结果质量：成功与否 + 效率 + 用户满意度"""
           success = 1.0 if result["success"] else 0.0
           efficiency = min(1.0, result["expected_steps"] / result["actual_steps"])
           return 0.6 * success + 0.2 * efficiency + 0.2 * result.get("satisfaction", 0.5)

3. 自主型进化：经验→技能→复用
------------------------------

这是最前沿的进化范式——Agent 在完成一个复杂任务后，自动将解决方案
提炼为可复用的技能文件，后续同类任务可直接调用。

.. code-block:: python

   class AutonomousLearner:
       def __init__(self):
           self.skills = {}  # {skill_name: skill_file}

       def after_completion(self, task: str, trace: list):
           """完成任务后，判断是否需要沉淀为技能"""
           if self._is_skill_worthy(trace):
               skill = self._create_skill(task, trace)
               name = skill["name"]
               self.skills[name] = skill
               return f"已创建技能：{name}"
           return None

       def _is_skill_worthy(self, trace: list) -> bool:
           """判断是否值得沉淀为技能：
           - 多步任务（>= 3 步）
           - 涉及多个工具调用
           - 可复用的通用性
           """
           return len(trace) >= 3

       def _create_skill(self, task: str, trace: list) -> dict:
           """将执行轨迹提炼为可复用的技能"""
           prompt = f"""
           任务：{task}
           执行步骤：
           {self._format_trace(trace)}

           请将以上执行过程提炼为可复用的技能，包含：
           1. 技能名称和触发条件
           2. 执行步骤
           3. 需要哪些工具
           4. 注意事项和陷阱
           """
           skill_content = llm.generate(prompt)
           return {
               "name": self._extract_name(task),
               "content": skill_content,
               "created": datetime.now(),
               "usage_count": 0,
           }

       def find_skill(self, task: str) -> str:
           """为新任务匹配合适的技能"""
           prompt = f"新任务：{task}\n已有技能：{list(self.skills.keys())}\n最匹配哪个技能？"
           match = llm.generate(prompt)
           if match in self.skills:
               self.skills[match]["usage_count"] += 1
               return self.skills[match]["content"]
           return None

.. admonition:: 从 Harness Engineering 到自进化
   :class: note

   2026 年兴起的 Harness Engineering 理念为自进化提供了方法论基础。其核心
   思想是：**"AI 每次犯错，都应该留下一条不会重犯的规则。"** 自进化能力
   将这一手动过程自动化——从"开发者给 Agent 造缰绳"变为"Agent 自己给自己
   造缰绳"。

   典型代表：Hermes Agent 的 GEPA 自我进化引擎，使用类反向传播方式优化
   prompt，仅需 100-500 次评估即可完成策略迭代。

自进化的工程实践
==================

.. list-table::
   :header-rows: 1

   * - 阶段
     - 目标
     - 方法
     - 频率
   * - 运行时
     - 即时纠正
     - 错误捕获 → 规则生成 → System Prompt 注入
     - 每次任务
   * - 会话级
     - 策略优化
     - 任务完成 → 评估 → 策略更新
     - 每次对话
   * - 跨会话
     - 技能沉淀
     - 执行跟踪 → 技能提炼 → 技能库
     - 每日/每周
   * - 团队级
     - 知识共享
     - 技能导出 → 审核 → 团队技能市场
     - 按需

.. code-block:: python

   # 完整的自进化 Agent 框架示例
   class SelfEvolvingAgent:
       def __init__(self, llm):
           self.llm = llm
           self.corrective = CorrectiveLearner()
           self.optimizing = OptimizingLearner(llm)
           self.autonomous = AutonomousLearner()

       def run(self, task: str) -> str:
           # 尝试匹配已有技能
           skill = self.autonomous.find_skill(task)
           system_prompt = self.corrective.get_system_prompt()

           try:
               result = self._execute(task, system_prompt, skill)
               self.optimizing.after_action(task, result)
               skill_msg = self.autonomous.after_completion(task, result["trace"])
               return result["output"]
           except Exception as e:
               rule = self.corrective.on_error(task, result.get("action"), str(e))
               return f"执行出错，已记录规则：{rule}"

       def _execute(self, task, system_prompt, skill):
           # 实际执行逻辑
           pass

.. admonition:: 自进化的边界
   :class: warning

   自进化不是万能的：
   - **规则膨胀**：累积过多规则可能降低模型性能，需要定期清理和合并
   - **过拟合**：过度优化过往任务可能损害在未知任务上的表现
   - **技能质量**：自动生成的技能需要人工审核门槛，防止错误知识固化
   - **安全边界**：自进化不应覆盖安全约束和原则性规则

   建议对进化内容设置**分层审核机制**——低风险规则自动采纳，高风险变更需人工确认。
