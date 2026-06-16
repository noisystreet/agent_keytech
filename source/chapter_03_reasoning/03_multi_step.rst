.. _chapter-03-multi-step:

===============================
多步推理
===============================

多步推理是 Agent 在面对需要多次工具调用才能解决的复杂任务时的核心能力。
这与单步 ReAct 不同——每一步的行动结果可能改变下一步的推理路径。

.. code-block:: python

   # 多步推理示例：查询某人的最新论文
   task = "查一下李飞飞教授 2024 年发表的论文"

   # Step 1: 搜索学者基本信息
   thought1 = "我需要先找到李飞飞教授的 Google Scholar 页面"
   action1 = search("李飞飞 Stanford professor Google Scholar")

   # Step 2: 根据搜索结果定位最新论文
   thought2 = "找到了她的页面，现在筛选 2024 年的论文"
   action2 = get_page("https://scholar.google.com/...")

   # Step 3: 整理结果
   thought3 = "找到了 5 篇 2024 年的论文，按引用排序"
   answer = "李飞飞教授 2024 年发表的论文包括：..."
