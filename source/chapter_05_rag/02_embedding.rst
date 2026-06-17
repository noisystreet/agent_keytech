.. _chapter-05-embedding:

===============================
文本嵌入
===============================

文本嵌入（Text Embedding）是将文本转换为向量表示的技术。它是 RAG 检索的
第一道门槛——嵌入质量直接决定了"Agent 能看到什么信息"。

很多人把嵌入当成一个"装上去就能用"的组件：调个 API 把文档转成向量，
存进向量数据库，然后查一下。但如果你真的在生产环境中调试过 Agent 的
检索质量，就会知道事情没这么简单。嵌入模型的选择、维度的权衡、相似度
计算方式——每个选择都在影响 Agent 最终的回答质量。

嵌入的本质：语义压缩
======================

嵌入模型做的事情可以理解为**语义压缩**：把一段不定长的文本"压"成一个
固定维度的向量（通常是 256 到 3072 维）。这个向量要保留原文的核心语义，
以至于向量空间中距离相近的两个点，对应的文本在语义上也相近。

.. code-block:: python

   # 嵌入向量的核心思想
   text_a = "苹果很好吃"
   text_b = "香蕉也很好吃"
   text_c = "MacBook 性能很好"

   vec_a = embed(text_a)  # → [0.12, 0.45, -0.33, ...] 768维
   vec_b = embed(text_b)  # → [0.11, 0.42, -0.30, ...] 接近 vec_a（都是水果）
   vec_c = embed(text_c)  # → [-0.21, 0.78, 0.15, ...]  远离 vec_a（不同主题）

   # 相似度：cosine(vec_a, vec_b) ≈ 0.92
   # 相似度：cosine(vec_a, vec_c) ≈ 0.15

.. admonition:: 冷知识：Word2Vec 的"国王 - 男人 + 女人 = 女王"
   :class: note

   2013 年 Google 的 Word2Vec 震撼了 NLP 界——训练出的词向量居然有
   **类比推理** 能力：vec("国王") - vec("男人") + vec("女人") ≈ vec("女王")。
   这背后的原理是嵌入向量在空间中编码了语义关系。今天的 Agent 嵌入模型
   继承了同样的基因，只不过向量维度和数据规模都大了几个数量级。
   BERT 的 768 维向量可以编码比 Word2Vec 的 300 维丰富得多的语义关系。

但这里有一个微妙的问题：**"相似"的定义高度依赖于嵌入模型的训练数据。**
一个在新闻语料上训练的模型，认为"苹果"和"香蕉"相似。一个在代码语料上
训练的模型，认为 "sort()" 和 "sorted()" 相似。一个在电商数据上训练的模型，
认为"苹果"和"iPhone"相似。

你选择什么嵌入模型，就是在选择什么"相似"定义——这个选择比大多数人
意识到的要重要得多。

.. admonition:: 为什么"苹果"有时候不是水果？
   :class: story

   如果你在网上搜"苹果"，搜索引擎返回的第一条结果通常是苹果公司（Apple Inc.）
   的主页，而不是水果苹果的百科。这不是因为搜索引擎"笨"，而是因为在互联网
   上，"苹果"这个词作为公司名出现的频率远高于作为水果。嵌入模型也会学到
   同样的偏见。这就是为什么**领域特定的嵌入模型**比通用模型在特定场景下
   表现好得多——一个在法律文书上训练的嵌入模型，知道"苹果"指的是一个
   商标案例，而不是水果或手机公司。

嵌入模型是如何训练的？
=========================

大多数现代嵌入模型使用**对比学习** （Contrastive Learning）训练。
训练数据的格式是三元组：(query, positive_passage, negative_passage)。
模型的训练目标是：让 query 和 positive 的向量距离更近，和 negative 的距离更远。

.. code-block:: python

   # 对比学习的核心 loss
   def contrastive_loss(query_vec, pos_vec, neg_vec, margin=0.5):
       """
       Triplet Loss：让正例距离 < 反例距离 - margin
       """
       pos_dist = cosine_distance(query_vec, pos_vec)
       neg_dist = cosine_distance(query_vec, neg_vec)
       loss = max(0, pos_dist - neg_dist + margin)
       return loss

这就解释了为什么嵌入模型在"见过"的数据上表现好，在"没见过"的数据上
表现差。如果你的 Agent 需要检索中文法律文档，但嵌入模型是在英文维基百科
上训练的，那检索质量大概率不会好——因为模型没有"学过"法律文本的相似性。

主流嵌入模型对比
====================

.. list-table::
   :header-rows: 1

   * - 模型
     - 维度
     - 最大长度
     - 语言
     - 推荐场景
     - MTEB 得分
   * - text-embedding-3-small
     - 1536
     - 8191
     - 多语言
     - 通用、成本敏感
     - 62.3
   * - text-embedding-3-large
     - 3072
     - 8191
     - 多语言
     - 高质量优先
     - 64.6
   * - BAAI/bge-large-zh-v1.5
     - 1024
     - 512
     - 中文
     - 中文场景首选
     - 63.0 (中文)
   * - BAAI/bge-m3
     - 1024
     - 8192
     - 多语言
     - 长文档+多语言
     - 64.2
   * - intfloat/multilingual-e5-large
     - 1024
     - 512
     - 多语言
     - 学术研究
     - 64.5
   * - jina-embeddings-v3
     - 1024
     - 8192
     - 多语言
     - 长文档检索
     - 64.5

**MTEB** （Massive Text Embedding Benchmark）是当前最权威的嵌入模型评测基准。
但注意：MTEB 分数高的模型不一定是你的场景的最优选择。MTEB 覆盖的任务很广，
如果你的 Agent 只需要做**检索** （Retrieval），你应该看 MTEB 的 Retrieval 子分数，
而非总体分数。

维度的权衡
==============

嵌入向量的维度是一个常见的"纠结"点。

.. code-block:: python

   # 维度对存储和计算的影响
   # 假设有 100 万条文本
   dim_256 = 256 * 4 * 1_000_000    # ≈ 1 GB   (float32)
   dim_1024 = 1024 * 4 * 1_000_000  # ≈ 4 GB
   dim_3072 = 3072 * 4 * 1_000_000  # ≈ 12 GB

   # 对于 100 万条数据用 exact search：
   # 每次搜索需要计算 100 万次余弦相似度
   # dim_256:  ~20ms
   # dim_1024: ~80ms
   # dim_3072: ~240ms

这里有一个很多人不知道的细节：**OpenAI 的 text-embedding-3 系列支持
降维**，而且降维后的质量下降比你想的要少。官方文档显示，text-embedding-3-large
降维到 256 维后，在大多数基准上的性能只下降了不到 5%。

.. code-block:: python

   import openai

   # 即使模型支持 3072 维，你也可以要求更小的维度
   response = openai.Embedding.create(
       model="text-embedding-3-large",
       input="文本内容",
       dimensions=256  # 明确指定输出维度
   )
   # 好处：省存储、省带宽、搜索更快
   # 代价：检索精度略微下降

为什么不是维度越高越好？因为高维度带来的收益是**递减**的。1024 维到 3072 维
的提升远小于 256 维到 1024 维。而存储和计算成本是线性增长的。

嵌入的预处理
==============

嵌入前的文本预处理对检索质量的影响，超过大多数人的预期。

.. code-block:: python

   def prepare_for_embedding(text: str, strategy="prefix"):
       """
       嵌入前的文本预处理策略。
       很多嵌入模型在训练时使用了特定的前缀模板。
       如果不匹配，检索质量会下降 10-30%。
       """
       strategies = {
           # BGE 系列：添加查询/文档前缀
           "bge": lambda t: (
               f"为文本生成向量表示：{t}"
               if is_query
               else f"将文本转化为向量：{t}"
           ),
           # E5 系列
           "e5": lambda t: (
               f"query: {t}" if is_query else f"passage: {t}"
           ),
           # Jina 系列
           "jina": lambda t: t,  # 不需要前缀
       }
       return strategies[strategy](text)

   # 不匹配前缀的后果
   query = "苹果的公司背景"
   doc = "苹果公司由 Steve Jobs 创立"

   # 错误做法：直接嵌入
   q_vec = bge_model.encode(query)          # 效果差
   d_vec = bge_model.encode(doc)            # 效果差

   # 正确做法：加上前缀
   q_vec = bge_model.encode(f"为文本生成向量表示：{query}")  # 效果好
   d_vec = bge_model.encode(f"将文本转化为向量：{doc}")      # 效果好

我见过不止一个团队因为忽略了这个前缀，导致检索质量差到无法上线。
每个嵌入模型的 HuggingFace 页面都会说明前缀要求——先看再用的三分钟时间，
可以省下几天调试时间。

密集嵌入 vs 稀疏嵌入
=======================

密集嵌入（Dense Embedding）是当前的主流，但它不是唯一的选择。
稀疏嵌入（Sparse Embedding）在某些场景下仍有不可替代的优势。

.. list-table::
   :header-rows: 1

   * - 对比维度
     - 密集嵌入（如 BGE、OpenAI）
     - 稀疏嵌入（如 BM25、SPLADE）
   * - 表现形式
     - 固定维度的稠密向量
     - 基于词袋的高维稀疏向量
   * - 语义理解
     - 好（能识别"汽车"和"车"是同一概念）
     - 差（只匹配精确关键词）
   * - 精确匹配
     - 差（"iPhone 15"和"iPhone 15 Pro"可能混淆）
     - 好（精确匹配型号、ID、代码）
   * - 领域迁移
     - 需要微调或换模型
     - 无参数，直接可用
   * - 可解释性
     - 差（不知道为什么认为相似）
     - 好（可以查看匹配了哪些词）
   * - Agent 场景
     - 语义搜索、开放域问答
     - 代码搜索、精确匹配、ID 查询

.. admonition:: 为什么 Agent 场景需要两者兼顾
   :class: tip

   Agent 经常需要同时处理两类检索需求：

   1. **语义检索**："帮我找一下关于 Transformer 的论文" → 密集嵌入擅长此道
   2. **精确匹配**："查询订单 #ORD-2024-001 的状态" → 稀疏嵌入精准命中

   这就是为什么生产 RAG 系统几乎都使用**混合搜索**——两者结合，取长补短。

相似度计算方法
================

选择相似度计算方法也会影响检索结果。

.. code-block:: python

   import numpy as np

   def cosine_similarity(a, b):
       """余弦相似度：关注方向而非长度。最常用。"""
       return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
       # 范围: [-1, 1]  实际常用: [0, 1]（向量经 L2 归一化后）

   def dot_product(a, b):
       """点积：同时考虑方向和长度。"""
       return np.dot(a, b)

   def euclidean_distance(a, b):
       """欧氏距离：绝对距离。"""
       return np.linalg.norm(a - b)

   # 大多数嵌入模型预设了 cosine（内部做了 L2 归一化）
   # 如果换了相似度方法，结果会不一样

实践中，绝大多数嵌入模型都默认使用余弦相似度。如果你用点积替代，
结果排序可能会有微妙差异——这不是"谁对谁错"的问题，而是你的嵌入模型
在训练时就是基于余弦相似度优化的。

嵌入缓存
============

Agent 运行时，嵌入计算是一个容易被忽视的性能瓶颈。

.. code-block:: python

   class EmbeddingCache:
       """
       嵌入缓存：避免重复计算相同文本的嵌入。
       在 Agent 场景中，大量文本（如系统提示词、常见问题）是重复的。
       """
       def __init__(self, backend, cache_size=10000):
           self.backend = backend  # 实际的嵌入模型
           self.cache = {}
           self.cache_size = cache_size

       def embed(self, text: str) -> list:
           # 用文本 hash 作为缓存键
           key = hashlib.md5(text.encode()).hexdigest()
           if key in self.cache:
               return self.cache[key]

           vec = self.backend.embed(text)
           if len(self.cache) < self.cache_size:
               self.cache[key] = vec
           return vec

   # 使用示例
   cached_embedder = EmbeddingCache(openai_embedder)
   # 对同一文本多次调用时，只有第一次会调用 API
   v1 = cached_embedder.embed("什么是 Agent？")  # 调用 API
   v2 = cached_embedder.embed("什么是 Agent？")  # 命中缓存

在生产环境中，嵌入缓存可以将检索延迟降低 50-80%，尤其是对于热门的
用户查询。缓存策略也很简单：LRU（最近最少使用），容量限制在几万条。

维度的降维技巧
================

对于大规模部署，降维是一个有效的手段。

.. code-block:: python

   from sklearn.decomposition import PCA

   class DimensionalityReducer:
       """
       嵌入降维：在检索质量损失可控的前提下，大幅降低存储和搜索成本。
       """
       def __init__(self, target_dim=256):
           self.pca = PCA(n_components=target_dim)
           self.fitted = False

       def fit(self, embeddings: np.ndarray):
           """在代表性数据上训练 PCA"""
           self.pca.fit(embeddings)
           self.fitted = True

       def transform(self, embedding: np.ndarray) -> np.ndarray:
           return self.pca.transform(embedding.reshape(1, -1))[0]

   # 使用示例
   reducer = DimensionalityReducer(target_dim=256)
   all_vectors = embedder.embed_all(documents)
   reducer.fit(all_vectors)  # 在全部数据上训练 PCA

   # 后续的新向量都经过降维
   reduced = [reducer.transform(v) for v in all_vectors]
   # 存储：12GB → 1GB（3072→256 维）

但降维有一个注意事项：**先降维再建索引，还是建索引后降维？**
答案是先降维再建索引。因为向量数据库的索引结构（如 HNSW）依赖于
向量之间的距离关系，降维后再建索引才能保证索引的准确性。

嵌入质量的评估
================

最后，如何判断你选的嵌入模型在你的数据上表现如何？不要只看 MTEB 分数，
应该在自己的数据上跑检索测试。

.. code-block:: python

   def evaluate_embedding(embedder, test_queries, test_corpus, k=10):
       """
       嵌入模型检索质量评估。
       test_queries: [{"query": "...", "relevant_docs": ["doc1", "doc2"]}]
       """
       # 1. 嵌入所有文档
       doc_vectors = {doc_id: embedder.embed(text)
                     for doc_id, text in test_corpus.items()}

       # 2. 对每个 query 检索
       total_recall = 0
       for item in test_queries:
           q_vec = embedder.embed(item["query"])

           # 计算所有文档相似度
           scores = {
               doc_id: cosine_similarity(q_vec, vec)
               for doc_id, vec in doc_vectors.items()
           }

           # 取 top-k
           top_k = sorted(scores, key=scores.get, reverse=True)[:k]

           # 计算召回率
           relevant = set(item["relevant_docs"])
           retrieved = set(top_k)
           recall = len(relevant & retrieved) / len(relevant)
           total_recall += recall

       avg_recall = total_recall / len(test_queries)
       return {"recall@k": avg_recall, "k": k}

   # 实测建议：准备 50-100 条 query + 对应答案
   # 跑 recall@10，如果低于 80%，说明嵌入模型不适合你的场景

在 Agent 场景中，一个好的经验值是：recall@10 应该达到 85% 以上，
否则 Agent 的最终回答质量会受到检索噪声的显著影响。
如果你的 recall 不达标，优先尝试换嵌入模型，而不是微调 Agent 的提示词。
