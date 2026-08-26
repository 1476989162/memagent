"""无第三方依赖的文本嵌入：字符 n-gram 哈希 + 余弦相似度。

为什么默认用字符 n-gram：
- 中文没有空格分词，字符 bigram/trigram 天然捕获中文语义片段；
- 英文同样适用，对大小写与标点不敏感；
- 用哈希把 n-gram 映射到固定维度向量，无需训练、无依赖。

碰撞治理（v0.3.1，检索天花板修复）：
- 维度 256 → 1024；FNV-1a 哈希加 `_mix` 低位混洗——旧版直接 `% 256` 取
  低位、末字符主导，造成跨文本系统性假相关（「火锅撞首都」类误命中）。
  现在无关文本相似度回到 0.00~0.04，短查询的「查不到」能被诚实识别。
- 符号哈希的病态（伪碰撞反向抵消真信号）由 agent.retrieve() 的词汇重叠
  保底兜底（见 agent.py REL_CANCEL_*）。
- 旧版（256 维，无混洗）持久化向量在 Memory.from_dict 里按维度失配检测
  并用当前嵌入器重建，存量记忆自动迁移。

**可插拔语义嵌入（v0.3.2）**：`embed_text()` 是单点——通过 `set_embedder()`
可换成真语义嵌入（OpenAI 兼容远程 / 本地 sentence-transformers，见
embedders.py），跨措辞等价（「技术栈」↔「Python 程序员」）才能命中。
默认 HashEmbedder 保持零依赖；换后端时向量的维度迁移仍由 from_dict 的
维度失配检测完成。词汇重叠保底与子串重排是**文本级**逻辑，不依赖后端。
"""

from __future__ import annotations

import math
import re
from typing import Iterable

DIM = 1024  # 默认哈希嵌入维度


class HashEmbedder:
    """默认零依赖嵌入器：字符 n-gram 带盐哈希 + 符号叠加。"""

    dim = DIM

    def embed(self, text: str) -> list[float]:
        return _embed_hash(text, DIM)


# 当前生效的嵌入器（单例，进程内所有 Agent 共用；默认零依赖哈希）
_EMBEDDER = HashEmbedder()


def set_embedder(embedder) -> None:
    """替换全局嵌入器（语义后端接入点）。

    只要求 embedder 提供 `.dim`（int）与 `.embed(text) -> list[float]`
    （L2 归一化）。存量记忆在下次加载时按维度失配自动重建；进程内已加载
    的记忆需自行重建或重启进程。
    """
    global _EMBEDDER
    _EMBEDDER = embedder


def get_embedder():
    return _EMBEDDER


def embedding_dim() -> int:
    """当前嵌入器的向量维度（存量迁移判断用）。"""
    return _EMBEDDER.dim


def _hash_ngram(gram: str, salt: int) -> int:
    """把 n-gram 字符串哈希到 [0, DIM) 空间（带盐，避免碰撞偏置）。"""
    h = 2166136261
    for ch in gram + chr(salt):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return _mix(h) % DIM


def _hash_sign(gram: str, salt: int) -> int:
    """第二个独立哈希 → +1/−1（符号哈希：碰撞从加性偏置变成对称噪声）。

    符号取自混洗值的**高位段**（第 16 位），与索引的低 10 位位段不相交——
    保证「同索引」与「同符号」彼此独立，碰撞对齐的期望概率是 1/2 而非被
    位段相关抬高。初始状态与索引哈希不同，进一步解耦。"""
    h = 40389  # 不同的初始状态，保证与索引哈希独立
    for ch in gram + chr(salt):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return 1 if ((_mix(h) >> 16) & 1) else -1


def _mix(h: int) -> int:
    """FNV-1a 的最终混洗：fold-xor + 奇数乘，打散低位。

    直接 `h % 1024` 取的是 FNV-1a 低位——低位缺乏雪崩（末字符主导），
    会造成跨文本的系统性假相关。混洗后再取模，桶分布回到均匀。
    """
    h ^= h >> 16
    h = (h * 0x45D9F3B) & 0xFFFFFFFF
    h ^= h >> 16
    return h & 0xFFFFFFFF


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


def _embed_hash(text: str, dim: int) -> list[float]:
    """哈希嵌入本体：各 n-gram 带符号等权叠加 + L2 归一化。"""
    vec = [0.0] * dim
    grams = ngrams(text)
    if not grams:
        return vec
    for salt in range(2):  # bigram 与 trigram 用不同盐
        for gram in grams:
            vec[_hash_ngram(gram, salt)] += _hash_sign(gram, salt)
    return normalize(vec)


def embed_text(text: str, dim: int | None = None) -> list[float]:
    """把文本嵌入为固定维度向量（转发给当前全局嵌入器）。

    dim 参数仅为旧调用方兼容：显式传 dim 时强制走哈希嵌入；缺省走
    当前嵌入器（语义后端替换后，此处即单点）。
    """
    if dim is not None:
        return _embed_hash(text, dim)
    return _EMBEDDER.embed(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度，输入应为 L2 归一化向量。"""
    if len(a) != len(b):
        raise ValueError("向量维度不一致")
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s
