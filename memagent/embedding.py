"""无第三方依赖的文本嵌入：字符 n-gram 哈希 + 余弦相似度。

为什么用字符 n-gram：
- 中文没有空格分词，字符 bigram/trigram 天然捕获中文语义片段；
- 英文同样适用，对大小写与标点不敏感；
- 用哈希把 n-gram 映射到固定维度向量，无需训练、无依赖。
"""

from __future__ import annotations

import math
import re
from typing import Iterable

DIM = 256  # 哈希向量维度


def _hash_ngram(gram: str, salt: int) -> int:
    """把 n-gram 字符串哈希到 [0, DIM) 空间（带盐，避免碰撞偏置）。"""
    h = 2166136261
    for ch in gram + chr(salt):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % DIM


def ngrams(text: str, ns: Iterable[int] = (2, 3)) -> list[str]:
    """提取字符 n-gram。归一化：小写、保留中日韩字符与字母数字。"""
    norm = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff]", "", text.lower())
    out: list[str] = []
    for n in ns:
        if len(norm) < n:
            continue
        out.extend(norm[i : i + n] for i in range(len(norm) - n + 1))
    return out


def normalize(vec: list[float]) -> list[float]:
    """L2 归一化（零向量原样返回）。"""
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_text(text: str, dim: int = DIM) -> list[float]:
    """把文本嵌入为固定维度向量（各 n-gram 等权叠加 + L2 归一化）。"""
    vec = [0.0] * dim
    grams = ngrams(text)
    if not grams:
        return vec
    for salt in range(2):  # bigram 与 trigram 用不同盐
        for gram in grams:
            idx = _hash_ngram(gram, salt)
            vec[idx] += 1.0
    return normalize(vec)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度，输入应为 L2 归一化向量。"""
    if len(a) != len(b):
        raise ValueError("向量维度不一致")
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s
