"""自动压缩：提取式摘要 + 相似记忆合并（模拟人脑"语义化"旧经历）。

没有 LLM 依赖：用词频 + 位置先验做抽取式摘要，把一簇相似记忆
融合成一条概括，保留可检索的语义要点。
"""

from __future__ import annotations

import re
from collections import Counter

from .embedding import cosine_similarity, embed_text

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句。"""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _tokenize(text: str) -> list[str]:
    """轻量分词：中文按字，英文按词，用于频率统计。"""
    tokens: list[str] = []
    for ch in re.findall(r"[\u4e00-\u9fff]", text):
        tokens.append(ch)
    tokens.extend(re.findall(r"[a-zA-Z0-9']+", text.lower()))
    return tokens


def extractive_summary(text: str, max_sentences: int = 2, max_chars: int = 120) -> str:
    """抽取式摘要：按词频加权 + 首句先验选出最重要的句子。"""
    sents = split_sentences(text)
    if not sents:
        return text[:max_chars]
    freq = Counter(_tokenize(text))
    scored: list[tuple[float, str]] = []
    for i, s in enumerate(sents):
        f = sum(freq[t] for t in _tokenize(s))
        position_bonus = 1.5 if i == 0 else 1.0  # 开头通常最核心
        length_penalty = 1.0 if len(s) <= 60 else 0.6
        scored.append((f * position_bonus * length_penalty, s))
    scored.sort(key=lambda x: -x[0])
    picked: list[str] = []
    total = 0
    for _, s in scored:
        if total + len(s) > max_chars and picked:
            break
        picked.append(s)
        total += len(s)
        if len(picked) >= max_sentences:
            break
    result = " ".join(picked)
    return result[:max_chars] or text[:max_chars]


def merge_similar(memories: list, threshold: float = 0.62) -> list[list]:
    """把内容高度相似的记忆聚成簇（贪心：相似度高于阈值的归并）。

    用内容嵌入比较而非存储向量——存储向量可能已被再巩固漂移。
    """
    clusters: list[list] = []
    for mem in memories:
        placed = False
        for cl in clusters:
            if cosine_similarity(embed_text(mem.content), embed_text(cl[0].content)) >= threshold:
                cl.append(mem)
                placed = True
                break
        if not placed:
            clusters.append([mem])
    return clusters
