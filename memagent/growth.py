"""生长与自主学习引擎（GrowthEngine）：预测-验证循环 + 模式提取 + 概念形成 + 自主提问。

零依赖纯 Python。LLM 可选增强（无 key 时用规则启发式）。

三个增长循环：
1. 预测-验证：每次观察对照历史预期，产生预测误差
2. 模式提取：同类记忆积累超过阈值 → 提炼"如果X则Y"的模式
3. 概念形成：多个实例共享特征 → 抽象为概念（皮层层级化）

自主提问：基于兴趣向量 + 预测缺口生成待探索问题。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable
import math
import time


# ---- 预测 ----

@dataclass
class Prediction:
    topic: str
    trigger: str          # 触发条件关键词
    expected: str         # 预期结果
    confidence: float = 0.0  # 当前置信度 (0~1)
    successes: int = 0
    total_tests: int = 0
    updated_at: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.successes / self.total_tests

    def to_dict(self) -> dict:
        return asdict(self)


# ---- 模式 ----

@dataclass
class Pattern:
    topic: str
    antecedent: str    # 如果 X
    consequent: str    # 则 Y
    support: int = 0   # 出现次数
    confidence: float = 0.0  # support/antecedent_count
    abstracted: bool = False  # 是否已抽象成概念


# ---- 观察 ----

@dataclass
class Observation:
    content: str
    topic: str | None = None
    matched_predictions: list = field(default_factory=list)


# ---- 概念 ----

@dataclass
class Concept:
    name: str
    topic: str
    abstract_properties: list = field(default_factory=list)
    instance_count: int = 0
    confidence: float = 0.0  # 抽象置信度


def _word_overlap(a: str, b: str) -> float:
    """简单词级重叠率。"""
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class GrowthEngine:
    def __init__(self):
        self.predictions: list[Prediction] = []
        self.patterns: list[Pattern] = []
        self.concepts: list[Concept] = []
        self.growth_history: list = []
        self.growth_step_count: int = 0
        self.autonomous_questions: list[str] = []
        # 兴趣向量代理（外部注入）
        self._interest_getter: Callable[[str], float] | None = None
        # 主题模式计数（用于触发模式提取）
        self._topic_pattern_count: dict[str, list[dict]] = {}

    def set_interest_getter(self, getter: Callable[[str], float]):
        self._interest_getter = getter

    def interest(self, topic: str) -> float:
        if self._interest_getter:
            return self._interest_getter(topic)
        return 0.0

    # ---- 预测管理 ----

    def predict(self, topic: str, trigger: str, expected: str, confidence: float = 0.3) -> Prediction:
        p = Prediction(topic=topic, trigger=trigger, expected=expected, confidence=confidence, updated_at=time.time())
        self.predictions.append(p)
        return p

    def find_predictions(self, topic: str | None = None, trigger_contains: str | None = None) -> list[Prediction]:
        r = self.predictions
        if topic:
            r = [p for p in r if p.topic == topic]
        if trigger_contains:
            r = [p for p in r if trigger_contains in p.trigger]
        return r

    # ---- 观察与匹配 ----

    def observe(self, content: str, topic: str | None = None) -> Observation:
        obs = Observation(content=content, topic=topic)
        matched = []
        for p in self.predictions:
            if topic and p.topic != topic:
                continue
            score = _word_overlap(p.trigger, content)
            if score < 0.2:
                continue
            obs.matched_predictions.append({"prediction": p.to_dict(), "match_score": score})
            matched.append((p, score))

        # 逐条对比：预期 vs 现实
        for p, score in matched:
            p.total_tests += 1
            # 预期内容是否在实际内容中出现
            expected_overlap = _word_overlap(p.expected, content)
            if expected_overlap >= 0.3:
                p.successes += 1
                p.confidence = min(1.0, p.confidence + 0.1)
            else:
                p.confidence = max(0.0, p.confidence - 0.1)

            self.growth_history.append({
                "step": self.growth_step_count,
                "type": "predict_verify",
                "prediction": p.trigger + "→" + p.expected,
                "confidence_change": expected_overlap >= 0.3,
                "new_confidence": round(p.confidence, 3),
                "observation": content,
            })

        return obs

    # ---- 模式提取 ----

    def record_pattern(self, topic: str, antecedent: str, consequent: str):
        """记录一条观察到的前件-后件对。累积到阈值后提炼为模式。"""
        self._topic_pattern_count.setdefault(topic, []).append({
            "antecedent": antecedent,
            "consequent": consequent,
        })
        self._try_extract_pattern(topic)

    def _try_extract_pattern(self, topic: str):
        records = self._topic_pattern_count.get(topic, [])
        if len(records) < 3:
            return
        # 找前件相同、后件相似的记录组
        seen = set()
        for rec in records:
            key = (rec["antecedent"], rec["consequent"])
            if key in seen:
                continue
            seen.add(key)
            similar = [r for r in records if r["antecedent"] == rec["antecedent"] and _jaccard(r["consequent"], rec["consequent"]) >= 0.4]
            if len(similar) >= 2:
                self.patterns.append(Pattern(
                    topic=topic,
                    antecedent=rec["antecedent"],
                    consequent=rec["consequent"],
                    support=len(similar),
                    confidence=len(similar) / max(len(records), 1),
                ))
                self.growth_history.append({
                    "step": self.growth_step_count,
                    "type": "pattern_extracted",
                    "topic": topic,
                    "pattern": f"{rec['antecedent']}→{rec['consequent']}",
                    "support": len(similar),
                    "confidence": round(len(similar) / max(len(records), 1), 3),
                })

    # ---- 概念形成 ----

    def form_concept(self, name: str, topic: str, abstract_properties: list[str] | None = None,
                     instance_count: int = 0) -> Concept:
        c = Concept(name=name, topic=topic, abstract_properties=abstract_properties or [],
                    instance_count=instance_count, confidence=min(1.0, instance_count * 0.2))
        self.concepts.append(c)
        self.growth_history.append({
            "step": self.growth_step_count,
            "type": "concept_formed",
            "topic": topic,
            "concept": name,
            "properties": abstract_properties or [],
            "instance_count": instance_count,
            "confidence": c.confidence,
        })
        return c

    def strengthen_concept(self, name: str, delta: float = 0.1):
        for c in self.concepts:
            if c.name == name:
                c.confidence = min(1.0, c.confidence + delta)
                c.instance_count += 1
                return c
        return None

    # ---- 预测生成（从模式中提炼预测） ----

    def generate_predictions_from_patterns(self) -> list[Prediction]:
        new = []
        for p in self.patterns:
            already = any(pred.trigger == p.antecedent and pred.expected == p.consequent for pred in self.predictions)
            if already:
                continue
            conf = p.confidence * 0.5 + 0.3
            pred = self.predict(topic=p.topic, trigger=p.antecedent, expected=p.consequent, confidence=conf)
            new.append(pred)
        return new

    # ---- 自主提问 ----

    def autonomous_query(self, max_questions: int = 3) -> list[str]:
        questions = []
        # 低置信度预测 → 待验证问题
        uncertain = sorted(self.predictions, key=lambda p: p.confidence)[:5]
        for p in uncertain:
            if p.confidence < 0.6 and len(questions) < max_questions:
                questions.append(f"验证：当{p.trigger}时，{p.expected}吗？")

        # 兴趣高但概念少的领域 → 探索问题
        topics = set(p.topic for p in self.patterns)
        for t in topics:
            if len(questions) >= max_questions:
                break
            interest = self.interest(t)
            concepts_count = sum(1 for c in self.concepts if c.topic == t)
            if interest > 0.5 and concepts_count == 0:
                questions.append(f"探索：关于{t}的规律是什么？")

        # 相邻兴趣主题 → 跨界问题
        if len(questions) < max_questions:
            topics_list = list(topics)
            for i, t1 in enumerate(topics_list):
                for t2 in topics_list[i+1:]:
                    if self.interest(t1) > 0.3 and self.interest(t2) > 0.3:
                        questions.append(f"关联：{t1}和{t2}之间有什么联系？")
                        if len(questions) >= max_questions:
                            break
                if len(questions) >= max_questions:
                    break

        self.autonomous_questions = questions[:max_questions]
        return self.autonomous_questions

    # ---- 生长步进 ----

    def grow_step(self, content: str, topic: str | None = None) -> dict:
        self.growth_step_count += 1
        obs = self.observe(content, topic=topic)
        # 从观察自动推断前件-后件（简单规则：包含"因为/所以/如果/那么"）
        self._infer_pattern_from_observation(content, topic)
        return {
            "step": self.growth_step_count,
            "matched_predictions": len(obs.matched_predictions),
            "new_patterns": len(self.patterns),
            "new_concepts": len(self.concepts),
            "new_questions": self.autonomous_query(),
        }

    def _infer_pattern_from_observation(self, content: str, topic: str | None):
        """简单规则：检测因果关键词 → 记录模式"""
        for sep, a_key, c_key in [("所以", "前件", "后件"), ("因为", "前件", "后件"), ("如果", "前件", "后件"),
                                   ("那么", "前件", "后件"), ("→", "前件", "后件")]:
            if sep in content and topic:
                parts = content.split(sep)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    self.record_pattern(topic, parts[0].strip(), parts[1].strip())

    def growth_summary(self) -> dict:
        return {
            "total_steps": self.growth_step_count,
            "predictions": len(self.predictions),
            "patterns": len(self.patterns),
            "concept_count": len(self.concepts),
            "top_predictions": [p.to_dict() for p in sorted(self.predictions, key=lambda x: -x.confidence)[:5]],
            "top_patterns": [{"antecedent": p.antecedent, "consequent": p.consequent, "support": p.support, "confidence": p.confidence} for p in sorted(self.patterns, key=lambda x: -x.support)[:5]],
            "concepts": [c.__dict__ for c in self.concepts],
        }

    def to_dict(self) -> dict:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "patterns": [asdict(p) for p in self.patterns],
            "concepts": [asdict(c) for c in self.concepts],
            "growth_history": list(self.growth_history),
            "growth_step_count": self.growth_step_count,
            "autonomous_questions": list(self.autonomous_questions),
            "topic_pattern_count": dict(self._topic_pattern_count),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "GrowthEngine":
        obj = cls()
        data = data or {}
        obj.predictions = [Prediction(**row) for row in (data.get("predictions") or [])]
        obj.patterns = [Pattern(**row) for row in (data.get("patterns") or [])]
        obj.concepts = [Concept(**row) for row in (data.get("concepts") or [])]
        obj.growth_history = list(data.get("growth_history") or [])
        obj.growth_step_count = int(data.get("growth_step_count", 0))
        obj.autonomous_questions = list(data.get("autonomous_questions") or [])
        obj._topic_pattern_count = {
            str(topic): list(rows)
            for topic, rows in (data.get("topic_pattern_count") or {}).items()
        }
        return obj
