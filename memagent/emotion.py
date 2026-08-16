"""情绪系统：三条轴（价/唤醒/自我相关）+ 出厂基本情绪 + 调制公式。

零依赖纯 Python，不接 LLM。情绪作为记忆的调制信号，不是独立系统。

设计原则：
- 出厂基本情绪来自 Ekman 跨文化研究（恐惧/快乐/悲伤/厌恶/愤怒/惊讶/好奇/中性）
- 调制公式模拟杏仁核-海马-前额叶回路
- 恐惧检测 = 杏仁核短路（高负价+高唤醒 → τ 放大 100 倍）
- 好奇心检测 = 前额叶过滤（低自我相关+中等唤醒+正价 → τ 缩小 0.3 倍）
"""
from __future__ import annotations

from dataclasses import dataclass

# ════════════════════════════════════════════════════════════
# 三条轴
# ════════════════════════════════════════════════════════════
# valence（价）：[-1.0, 1.0]   -1=极度负面(恐惧/厌恶)  1=极度正面(快乐)
# arousal（唤醒）：[0.0, 1.0]   0=平静/麻木  1=极度激动
# self_relevance（自我相关）：[0.0, 1.0]  0=与"我"无关  1=高度涉及自我

@dataclass(frozen=True)
class Emotion:
    valence: float
    arousal: float
    self_relevance: float
    label: str = "neutral"

    def clamp(self) -> "Emotion":
        return Emotion(
            valence=max(-1.0, min(1.0, self.valence)),
            arousal=max(0.0, min(1.0, self.arousal)),
            self_relevance=max(0.0, min(1.0, self.self_relevance)),
            label=self.label,
        )


# ════════════════════════════════════════════════════════════
# 出厂基本情绪（Ekman + 好奇/中性）
# ════════════════════════════════════════════════════════════

BASIC_EMOTIONS: dict[str, Emotion] = {
    "fear":      Emotion(valence=-0.9, arousal=0.9, self_relevance=0.8, label="fear"),
    "joy":       Emotion(valence=0.8,  arousal=0.7, self_relevance=0.6, label="joy"),
    "sadness":   Emotion(valence=-0.7, arousal=0.2, self_relevance=0.7, label="sadness"),
    "disgust":   Emotion(valence=-0.6, arousal=0.5, self_relevance=0.5, label="disgust"),
    "anger":     Emotion(valence=-0.5, arousal=0.8, self_relevance=0.6, label="anger"),
    "surprise":  Emotion(valence=0.0,  arousal=0.8, self_relevance=0.2, label="surprise"),
    "curiosity": Emotion(valence=0.3,  arousal=0.5, self_relevance=0.2, label="curiosity"),
    "neutral":   Emotion(valence=0.0,  arousal=0.2, self_relevance=0.1, label="neutral"),
}


# ════════════════════════════════════════════════════════════
# 出厂调制参数
# ════════════════════════════════════════════════════════════

# τ 调制
K_VALENCE_TAU: float = 0.6       # 负价→τ延长, 正价→τ缩短
K_AROUSAL_TAU: float = 0.3       # 高唤醒→τ延长（威胁/兴奋）
K_SELF_TAU: float = 0.4          # 高自我相关→τ延长

# 编码调制
K_ENCODING: float = 0.8          # 高唤醒→编码加深

# 再巩固调制
K_SELF_DRIFT: float = 1.5        # 高自我相关→drift升高（易改写）

# 检索调制
K_CONGRUENCE: float = 0.5        # 情绪一致性对检索的提升系数

# 恐惧检测
FEAR_VALENCE: float = -0.7
FEAR_AROUSAL: float = 0.7
FEAR_BOOST: float = 100.0

# 好奇心检测
NOVELTY_SELF: float = 0.3
NOVELTY_BOOST: float = 0.3


# ════════════════════════════════════════════════════════════
# 调制函数
# ════════════════════════════════════════════════════════════

def tau_factor(emotion: Emotion | None) -> float:
    """情绪对遗忘速度的调制因子。

    τ_effective = τ_base × τ_factor(emotion)

    - 恐惧 → factor≈31 (τ 极大，几乎不遗忘)
    - 快乐 → factor≈0.57 (τ 缩短，快乐会淡化)
    - 好奇 → factor≈0.29 (τ 极短，看完即忘)
    - 高自我相关 → factor > 1 (与自我相关更牢)
    - None → factor=1.0 (无情绪调制)
    """
    if emotion is None:
        return 1.0
    e = emotion.clamp()

    valence_contrib = 1.0 - e.valence * K_VALENCE_TAU
    arousal_contrib = 1.0 + e.arousal * K_AROUSAL_TAU
    self_contrib = 1.0 + e.self_relevance * K_SELF_TAU
    factor = valence_contrib * arousal_contrib * self_contrib

    # 恐惧（杏仁核短路）
    if e.valence < FEAR_VALENCE and e.arousal > FEAR_AROUSAL:
        factor *= FEAR_BOOST

    # 好奇心/新奇（低相关 + 中等唤醒 + 正价 → 快速丢弃）
    if (e.self_relevance < NOVELTY_SELF
            and 0.2 < e.arousal < 0.7
            and e.valence > 0):
        factor *= NOVELTY_BOOST

    return max(0.01, factor)


def encoding_factor(emotion: Emotion | None) -> float:
    """情绪对编码强度的调制因子。

    importance_effective = importance × encoding_factor(emotion)
    高唤醒 → 编码加深（注意力集中，海马体深度加工）
    """
    if emotion is None:
        return 1.0
    return 1.0 + emotion.arousal * K_ENCODING


def drift_factor(emotion: Emotion | None) -> float:
    """情绪对再巩固可塑性的调制因子。

    drift_effective = drift_base × drift_factor(emotion)
    高自我相关 → 可塑性强（身份相关记忆不断进化）
    恐惧类 → 可塑性极低（冻结，不可改写）
    """
    if emotion is None:
        return 1.0
    e = emotion.clamp()
    factor = 1.0 + e.self_relevance * K_SELF_DRIFT

    if e.valence < FEAR_VALENCE and e.arousal > FEAR_AROUSAL:
        factor *= 0.01

    return max(0.0, factor)


def congruence_factor(current: Emotion | None, memory: Emotion | None) -> float:
    """情绪一致性调制因子。

    当前情绪与记忆情绪相似 → 提升检索分数
    当前情绪与记忆情绪相反 → 降低检索分数

    模拟"情绪一致性效应"：快乐时更容易想起快乐记忆。
    """
    if current is None or memory is None:
        return 1.0

    valence_match = current.valence * memory.valence  # 同号为正
    arousal_match = 1.0 - abs(current.arousal - memory.arousal)

    factor = (1.0
              + valence_match * K_CONGRUENCE
              + arousal_match * K_CONGRUENCE * 0.3)

    return max(0.1, min(2.0, factor))


# ════════════════════════════════════════════════════════════
# 关键词启发式情绪推断（零依赖中文词典）
# ════════════════════════════════════════════════════════════

_EMOTION_KEYWORDS: dict[str, str] = {
    "害怕": "fear", "恐惧": "fear", "恐怖": "fear", "危险": "fear",
    "威胁": "fear", "可怕": "fear", "惊慌": "fear", "吓得": "fear",
    "惊吓": "fear", "恐惧感": "fear", "害怕": "fear", "惊恐": "fear",
    "快乐": "joy", "开心": "joy", "高兴": "joy", "幸福": "joy",
    "愉快": "joy", "享受": "joy", "欢乐": "joy", "喜欢": "joy",
    "伤心": "sadness", "难过": "sadness", "悲伤": "sadness", "痛苦": "sadness",
    "失去": "sadness", "孤独": "sadness", "哭泣": "sadness", "悲痛": "sadness",
    "愤怒": "anger", "生气": "anger", "恼火": "anger", "恨": "anger",
    "发怒": "anger", "怒火": "anger",
    "恶心": "disgust", "讨厌": "disgust", "厌恶": "disgust", "反感": "disgust",
    "恶臭": "disgust",
    "惊讶": "surprise", "意外": "surprise", "没想到": "surprise",
    "震惊": "surprise", "吃惊": "surprise",
    "好奇": "curiosity", "想知道": "curiosity", "探索": "curiosity",
    "为什么": "curiosity", "怎么": "curiosity", "如何": "curiosity",
}


def infer_emotion(text: str) -> Emotion:
    """从文本关键词推断情绪（零依赖启发式）。

    命中多个情绪关键词时取最高频的标签。
    无命中时返回 neutral。
    """
    t = text.lower()
    hits: dict[str, int] = {}
    for kw, label in _EMOTION_KEYWORDS.items():
        if kw in t:
            hits[label] = hits.get(label, 0) + 1

    if not hits:
        return BASIC_EMOTIONS["neutral"]

    best = max(hits, key=lambda k: hits[k])
    return BASIC_EMOTIONS[best]