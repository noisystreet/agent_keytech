.. _chapter-05-chunking:

===============================
文档分块
===============================

文档分块（Chunking）是将长文档切分为可检索的小块。分块策略直接影响检索效果。

常见分块策略
================

- **固定大小分块**：按 token 数均分，简单但可能切断语义
- **语义分块**：按段落、标题或语义边界切分，保留完整语义
- **递归分块**：从语义边界（如标题→段落→句子→字符）逐级尝试

.. code-block:: python

   class RecursiveChunker:
       def __init__(self, chunk_size=500, overlap=50):
           self.chunk_size = chunk_size
           self.overlap = overlap

       def chunk(self, text: str) -> List[str]:
           separators = ["\n\n", "\n", "。", "；", "，"]
           chunks = []
           current = ""

           for separator in separators:
               segments = text.split(separator)
               for seg in segments:
                   if len(tokenize(current + seg)) > self.chunk_size:
                       chunks.append(current)
                       current = seg
                   else:
                       current += seg
           if current:
               chunks.append(current)
           return chunks
