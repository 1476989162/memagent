"""兴趣向量系统（InterestVector）：三条路径共同塑造成长方向。

显式指定 · 情绪驱动 · 知识密度涌现，三条路径同时生效：
- 显式指定：用户主动 set_growth_direction("音乐", 0.8) → 兴趣值直接写入
- 情绪驱动：强情绪记忆写入时，所属主题兴趣值随 arousal 递增
- 密度涌现：高频检索/成功回忆的主题，兴趣值自动累积

兴趣值 [0~1] 调制三处：
  编码 importance = base × (1 + intensity × 0.5)
  检索 relevance = base × (1 + intensity × 0.3)
  遗忘 τ_effective = base × (1 + intensity × 1.0)

慢衰减防止兴趣固化：每次 respond 周期 decay_rate=0.01，不活跃的领域自然降级。

零依赖：主题匹配纯关键词，无 LLM。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class InterestVector:
    """成长方向（兴趣向量）：主题→兴趣强度。"""
    # 兴趣强度（用户 set + 情绪累积 + 检索累积 + 时间衰减）
    intensities: dict[str, float] = field(default_factory=dict)
    # 用户注册的主题→关键词集合（用于文本自动主题检测）
    topic_keywords: dict[str, set[str]] = field(default_factory=dict)
    # 兴趣衰减率（每 respond 周期，防止兴趣永久固化）
    decay_rate: float = 0.01
    # 兴趣值边界
    max_intensity: float = 1.0
    min_intensity: float = 0.0
    # 自动演化参数
    emotion_delta_per_unit: float = 0.03   # 情绪驱动增量（× arousal）
    access_delta: float = 0.005            # 每次检索命中增量
    recall_success_delta: float = 0.015    # 成功回忆（高相关检索）增量

    def register_topic(self, topic: str, keywords: Iterable[str]) -> None:
        """注册一个主题及其关键词集合，供 remember() 自动匹配。"""
        self.topic_keywords[topic] = set(kw.lower() for kw in keywords)
        self.intensities.setdefault(topic, 0.0)

    def detect_topics(self, text: str) -> list[str]:
        """关键词匹配：返回与文本匹配的所有已注册主题。"""
        if not self.topic_keywords:
            return []
        t = text.lower()
        matched = []
        for topic, kws in self.topic_keywords.items():
            if any(kw in t for kw in kws):
                matched.append(topic)
        return matched

    def update(self, topic: str, delta: float) -> float:
        """更新兴趣值并返回新值。"""
        cur = self.intensities.get(topic, self.min_intensity)
        new = max(self.min_intensity, min(self.max_intensity, cur + delta))
        self.intensities[topic] = new
        return new

    def apply_decay(self) -> None:
        """兴趣衰减：不活跃的领域自然降级，模拟真实兴趣漂移。"""
        for topic in list(self.intensities.keys()):
            self.intensities[topic] = max(
                self.min_intensity,
                self.intensities[topic] * (1 - self.decay_rate),
            )

    def get(self, topic: str) -> float:
        """获取主题兴趣值（未注册主题返回 0）。"""
        return self.intensities.get(topic, 0.0)

    def set(self, topic: str, intensity: float) -> None:
        """手动设置兴趣值（显式指定路径）。"""
        self.intensities[topic] = max(self.min_intensity, min(self.max_intensity, intensity))
        self.topic_keywords.setdefault(topic, set())  # 保证主题存在

    def top(self, n: int = 5) -> list[tuple[str, float]]:
        """兴趣最高的前n个主题。"""
        return sorted(self.intensities.items(), key=lambda x: -x[1])[:n]

    def to_dict(self) -> dict:
        return {
            "intensities": dict(self.intensities),
            "topic_keywords": {t: sorted(kw) for t, kw in self.topic_keywords.items()},
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "InterestVector":
        data = data or {}
        obj = cls()
        obj.intensities = {
            str(topic): max(obj.min_intensity, min(obj.max_intensity, float(value)))
            for topic, value in (data.get("intensities") or {}).items()
        }
        obj.topic_keywords = {
            str(topic): {str(keyword).lower() for keyword in keywords}
            for topic, keywords in (data.get("topic_keywords") or {}).items()
        }
        for topic in obj.intensities:
            obj.topic_keywords.setdefault(topic, set())
        return obj

    @property
    def active_topics(self) -> list[str]:
        """当前有非零兴趣的主题。"""
        return [t for t, v in self.intensities.items() if v > self.min_intensity]
