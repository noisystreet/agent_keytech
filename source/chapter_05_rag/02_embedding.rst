.. _chapter-05-embedding:

===============================
文本嵌入
===============================

文本嵌入（Text Embedding）将文本转换为向量表示，是向量检索的基础。
不同的嵌入模型对下游 Agent 的表现有显著影响。

.. admonition:: 嵌入模型的"语文"能力
   :class: funfact

   嵌入模型的质量决定了 Agent 的"语文"水平——它决定了文本之间的相似度计算是否准确。
   一个中文嵌入模型可能认为"苹果"和"香蕉"相似（都是水果），但不认为"苹果"和"MacBook"相关。
   而一个代码嵌入模型可能认为 "sort()" 和 "sorted()" 高度相似。选择合适的嵌入模型
   比搭建检索系统本身更重要。

.. code-block:: python

   # 常见的嵌入模型选择
   embeddings = {
       "text-embedding-3-large": {"dim": 3072, "cost": "$0.13/M"},
       "text-embedding-3-small": {"dim": 1536, "cost": "$0.02/M"},
       "bge-large-zh":          {"dim": 1024, "cost": "免费（本地）"},
   }

   def embed(text: str, model="text-embedding-3-small"):
       response = openai.Embedding.create(model=model, input=text)
       return response.data[0].embedding
