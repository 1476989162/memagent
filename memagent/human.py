"""人类优势增强与短处去除模块（HumanStrengths）：模拟人脑的高级认知功能。

新增机制：
1. 情境编码（Episodic Context Encoding）：检索时编码情境线索（时间/地点/情绪/在场者），
   情境匹配时提升检索——对应人脑的"编码特异性原则"（Tulving）。
2. 元认知监控（Metacognitive Monitor）：追踪预测准确率与检索命中率，
   给出"认知自信度"——知道自己知道什么、不知道什么。
3. 偏差检测（Bias Detector）：检测确认偏误、可得性启发、近因效应等认知偏差，
   给出偏差警告——对应人脑的"元认知校准"。
4. 前瞻记忆（Prospective Memory）：记住"将来要做的事"，在合适时机主动提醒——
   对应人脑的"基于事件的前瞻记忆"。
5. 精细复述（Elaborative Rehearsal）：回忆时把新信息与已有知识网络关联，
   而非简单重复——对应人脑的"深度加工优势"。
6. 情绪调节（Emotion Regulation）：恐惧100x降为更合理范围，
   加入认知重评（reappraisal）机制——对应人脑的前额叶-杏仁核调控回路。
7. 睡眠记忆分级（Memory Triage）：睡眠时按重要性分级处理，
   优先巩固高价值记忆——对应人脑的"选择性巩固"。
8. 间隔重复优化（Spaced Repetition）：基于遗忘曲线动态调整复习间隔，
   在即将遗忘时精准触发——对应人脑的"间隔效应"。

设计原则：
- 保留人类优势（情绪一致、情境编码、元认知、精细加工）
- 去除人类短处（极端恐惧、认知偏差、无法校准、近因主导）
- 零依赖纯 Python，LLM 可选增强
- 与现有模块无缝集成（通过 Agent 属性访问）
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any

from .decay import DecayScorer
from .emotion import Emotion, tau_factor, encoding_factor, drift_factor


# ════════════════════════════════════════════════════════════
# 1. 情境编码（Episodic Context Encoding）
# ════════════════════════════════════════════════════════════

@dataclass
class Context:
    """记忆编码时的情境线索——对应人脑的编码特异性原则。"""
    timestamp: float = 0.0          # 时间戳
    emotion_label: str = "neutral"  # 情绪标签
    location: str = ""              # 地点（可选）
    co_present: list = field(default_factory=list)  # 在场者
    activity: str = ""              # 正在做什么

    def match_score(self, other: "Context") -> float:
        """两个情境的匹配度（0~1）——情境越相似，检索越容易。"""
        if not isinstance(other, Context):
            return 0.0
        score = 0.0
        # 情绪匹配（强权重）
        if self.emotion_label == other.emotion_label:
            score += 0.4
        # 地点匹配
        if self.location and self.location == other.location:
            score += 0.3
        # 在场者重叠
        if self.co_present and other.co_present:
            overlap = len(set(self.co_present) & set(other.co_present))
            total = len(set(self.co_present) | set(other.co_present))
            score += 0.2 * (overlap / total if total else 0)
        # 活动匹配
        if self.activity and self.activity == other.activity:
            score += 0.1
        return min(1.0, score)


@dataclass
class ContextualMemory:
    """带情境编码的记忆条目——每次检索都附带当前情境。"""
    memory_id: str
    context: Context
    encoding_strength: float = 1.0  # 编码强度（深度加工越深越强）

    def retrieval_cue_match(self, current: Context) -> float:
        """当前情境与编码情境的匹配度——情境依赖性检索。"""
        return self.context.match_score(current)


class ContextualEncoding:
    """情境编码管理器：为记忆附加情境线索，检索时利用情境匹配。"""

    def __init__(self):
        self.contexts: dict[str, ContextualMemory] = {}
        self.current_context: Context = Context()

    def encode(self, memory_id: str, context: Context | None = None,
               strength: float = 1.0) -> ContextualMemory:
        """为记忆编码情境线索。"""
        ctx = context or self.current_context
        cm = ContextualMemory(
            memory_id=memory_id,
            context=ctx,
            encoding_strength=strength
        )
        self.contexts[memory_id] = cm
        return cm

    def set_current(self, context: Context) -> None:
        """更新当前情境（每次交互时调用）。"""
        self.current_context = context

    def get_context_bonus(self, memory_id: str) -> float:
        """获取情境匹配加成（1.0 = 无加成，>1.0 = 情境匹配提升）。"""
        if memory_id not in self.contexts:
            return 1.0
        cm = self.contexts[memory_id]
        match = cm.retrieval_cue_match(self.current_context)
        # 编码越深、情境越匹配，加成越高（最高 1.5x）
        return 1.0 + match * 0.5 * cm.encoding_strength

    def to_dict(self) -> dict:
        return {
            "contexts": {
                mid: {
                    "memory_id": cm.memory_id,
                    "timestamp": cm.context.timestamp,
                    "emotion_label": cm.context.emotion_label,
                    "location": cm.context.location,
                    "co_present": cm.context.co_present,
                    "activity": cm.context.activity,
                    "encoding_strength": cm.encoding_strength,
                }
                for mid, cm in self.contexts.items()
            },
            "current": {
                "timestamp": self.current_context.timestamp,
                "emotion_label": self.current_context.emotion_label,
                "location": self.current_context.location,
                "co_present": self.current_context.co_present,
                "activity": self.current_context.activity,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextualEncoding":
        obj = cls()
        data = data or {}
        for mid, cm_data in (data.get("contexts") or {}).items():
            ctx = Context(
                timestamp=cm_data.get("timestamp", 0.0),
                emotion_label=cm_data.get("emotion_label", "neutral"),
                location=cm_data.get("location", ""),
                co_present=cm_data.get("co_present", []),
                activity=cm_data.get("activity", ""),
            )
            obj.contexts[mid] = ContextualMemory(
                memory_id=cm_data.get("memory_id", mid),
                context=ctx,
                encoding_strength=cm_data.get("encoding_strength", 1.0),
            )
        cur = data.get("current") or {}
        obj.current_context = Context(
            timestamp=cur.get("timestamp", 0.0),
            emotion_label=cur.get("emotion_label", "neutral"),
            location=cur.get("location", ""),
            co_present=cur.get("co_present", []),
            activity=cur.get("activity", ""),
        )
        return obj


# ════════════════════════════════════════════════════════════
# 2. 元认知监控（Metacognitive Monitor）
# ════════════════════════════════════════════════════════════

class JudgmentType(str, Enum):
    """元认知判断类型。"""
    PREDICTION = "prediction"      # 对未来事件的预测
    RECALL_CONFIDENCE = "recall_confidence"  # 回忆时的自信度
    ESTIMATION = "estimation"      # 数值估计


@dataclass
class MetacognitiveRecord:
    """一条元认知校准记录。"""
    timestamp: float
    judgment_type: JudgmentType
    predicted: float              # 预测值/自信度 (0~1)
    actual: float | None          # 实际结果 (0~1, None = 未知)
    description: str = ""

    @property
    def error(self) -> float | None:
        if self.actual is None:
            return None
        return self.predicted - self.actual

    @property
    def is_overconfident(self) -> bool:
        """是否过度自信（预测远高于实际）。"""
        e = self.error
        return e is not None and e > 0.2


class MetacognitiveMonitor:
    """元认知校准器：追踪预测准确率，给出认知自信度报告。

    解决人类短处：
    - 人类常过度自信（预测80%确信，实际只有50%）
    - 人类校准差（无法准确评估自己知道什么）

    去除方法：
    - 追踪所有预测与回忆的校准曲线
    - 给出"校准后的自信度"（而非原始自信度）
    - 发现系统性偏差时自动警告
    """

    def __init__(self):
        self.records: list[MetacognitiveRecord] = []
        self._calibration_cache: dict[str, float] = {}

    def record_prediction(self, judgment_type: JudgmentType,
                         predicted: float, actual: float | None = None,
                         description: str = "") -> MetacognitiveRecord:
        """记录一个预测/判断，后续可验证。"""
        rec = MetacognitiveRecord(
            timestamp=time.time(),
            judgment_type=judgment_type,
            predicted=max(0.0, min(1.0, predicted)),
            actual=max(0.0, min(1.0, actual)) if actual is not None else None,
            description=description,
        )
        self.records.append(rec)
        return rec

    def verify(self, record: MetacognitiveRecord, actual: float) -> None:
        """验证一个先前的预测。"""
        record.actual = max(0.0, min(1.0, actual))
        self._calibration_cache.clear()  # 验证变化，清除缓存

    def calibration_report(self) -> dict:
        """校准报告：预测准确率的系统性偏差分析。"""
        verified = [r for r in self.records if r.actual is not None]
        if not verified:
            return {"status": "insufficient_data", "records": 0}

        errors = [r.error for r in verified if r.error is not None]
        if not errors:
            return {"status": "no_errors", "records": len(verified)}

        mean_error = sum(errors) / len(errors)  # >0 = 过度自信
        abs_errors = [abs(e) for e in errors]
        mean_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else 0

        # 分类型统计
        by_type: dict[str, list[float]] = defaultdict(list)
        for r in verified:
            by_type[r.judgment_type.value].append(r.error)

        type_calibration = {}
        for jtype, errs in by_type.items():
            type_calibration[jtype] = {
                "mean_error": round(sum(errs) / len(errs), 3),
                "count": len(errs),
                "overconfident": sum(1 for e in errs if e > 0.2),
                "underconfident": sum(1 for e in errs if e < -0.2),
            }

        return {
            "status": "calibrated",
            "records": len(verified),
            "mean_bias": round(mean_error, 3),  # >0 = 过度自信系统偏差
            "mean_abs_error": round(mean_abs_error, 3),
            "by_type": type_calibration,
            "recommendation": self._calibration_recommendation(mean_error),
        }

    def _calibration_recommendation(self, mean_error: float) -> str:
        """根据校准结果给出建议。"""
        if mean_error > 0.3:
            return "严重过度自信：预测确信度应平均下调 {:.0%}".format(mean_error)
        elif mean_error > 0.15:
            return "中度过度自信：预测确信度应下调 {:.0%}".format(mean_error)
        elif mean_error < -0.3:
            return "严重不自信：预测确信度可上调 {:.0%}".format(-mean_error)
        elif mean_error < -0.15:
            return "中度不自信：预测确信度可上调 {:.0%}".format(-mean_error)
        else:
            return "校准良好：预测准确率可靠"

    def adjusted_confidence(self, raw_confidence: float,
                           judgment_type: JudgmentType = JudgmentType.PREDICTION) -> float:
        """校准后的自信度——自动修正系统性偏差。"""
        report = self.calibration_report()
        if report["status"] != "calibrated":
            return raw_confidence

        bias = report.get("mean_bias", 0.0)
        # 过度自信时下调，不自信时上调
        adjusted = raw_confidence - bias
        return max(0.05, min(0.95, adjusted))

    def to_dict(self) -> dict:
        return {
            "records": [
                {
                    "timestamp": r.timestamp,
                    "judgment_type": r.judgment_type.value,
                    "predicted": r.predicted,
                    "actual": r.actual,
                    "description": r.description,
                }
                for r in self.records[-100:]  # 只保留最近100条
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetacognitiveMonitor":
        obj = cls()
        data = data or {}
        for r in (data.get("records") or []):
            obj.records.append(MetacognitiveRecord(
                timestamp=r.get("timestamp", 0.0),
                judgment_type=JudgmentType(r.get("judgment_type", "prediction")),
                predicted=r.get("predicted", 0.5),
                actual=r.get("actual"),
                description=r.get("description", ""),
            ))
        return obj


# ════════════════════════════════════════════════════════════
# 3. 偏差检测（Bias Detector）
# ════════════════════════════════════════════════════════════

class BiasType(str, Enum):
    """认知偏差类型——人类短处清单。"""
    CONFIRMATION = "confirmation"      # 确认偏误：只记支持已有信念的
    RECENCY = "recency"                # 近因效应：过度重视最近信息
    AVAILABILITY = "availability"      # 可得性启发：容易想起的认为更常见
    ANCHORING = "anchoring"            # 锚定效应：被初始值过度影响
    SURVIVORSHIP = "survivorship"      # 幸存者偏差：只看到成功案例
    OVERCONFIDENCE = "overconfidence"  # 过度自信：高估自己准确性


@dataclass
class BiasWarning:
    """偏差警告。"""
    bias_type: BiasType
    severity: float  # 0~1
    description: str
    suggestion: str
    timestamp: float = field(default_factory=time.time)


class BiasDetector:
    """认知偏差检测器：监控记忆检索和决策中的系统性偏差。

    去除人类短处的方法：
    - 检测确认偏误 → 主动搜索反面证据
    - 检测近因效应 → 提升远期高价值记忆的权重
    - 检测可得性启发 → 引入基率信息
    - 检测过度自信 → 下调确信度
    """

    def __init__(self):
        self.warnings: list[BiasWarning] = []
        self._bias_history: dict[str, list[float]] = defaultdict(list)

    def check_confirmation_bias(self, retrieved_memories: list,
                                 query: str,
                                 prior_beliefs: list[str] | None = None) -> BiasWarning | None:
        """检测确认偏误：检索结果是否全部支持已有信念？"""
        if not retrieved_memories or not prior_beliefs:
            return None

        # 简化：检查检索结果是否高度同质化
        if len(retrieved_memories) >= 3:
            # 如果全部记忆的情绪标签相同，可能存在确认偏义
            emotions = [getattr(m, 'emotion', None) for m in retrieved_memories]
            emotions = [e for e in emotions if e is not None]
            if emotions and len(set(e.label for e in emotions)) == 1:
                return self._add_warning(BiasType.CONFIRMATION, 0.4,
                    "检索结果高度同质化：可能只召回了支持现有信念的证据",
                    "建议主动搜索反面证据或不同视角的记忆")
        return None

    def check_recency_bias(self, retrieved_memories: list,
                          now: float,
                          threshold_ratio: float = 0.7) -> BiasWarning | None:
        """检测近因效应：检索结果是否过度集中在近期？"""
        if not retrieved_memories:
            return None

        # 检查时间分布
        recent_count = sum(
            1 for m in retrieved_memories
            if hasattr(m, 'created_at') and now - m.created_at < 3600  # 最近1小时（agent时间）
        )
        ratio = recent_count / len(retrieved_memories)

        if ratio > threshold_ratio:
            return BiasWarning(
                bias_type=BiasType.RECENCY,
                severity=ratio - 0.5,
                description=f"检索结果 {ratio:.0%} 集中在近期记忆，可能忽略远期高价值信息",
                suggestion="提升远期高重要性记忆的检索权重，或主动搜索跨时间段的证据",
            )
        return None

    def check_availability_bias(self, retrieved_memories: list,
                                 topic: str) -> BiasWarning | None:
        """检测可得性启发：容易回忆的是否被高估？"""
        if not retrieved_memories:
            return None

        # 高可得性 = 高频检索 + 高强度
        high_availability = [
            m for m in retrieved_memories
            if hasattr(m, 'access_count') and m.access_count > 5
            and hasattr(m, 'importance') and m.importance > 0.7
        ]
        if len(high_availability) == len(retrieved_memories) and len(retrieved_memories) >= 2:
            return BiasWarning(
                bias_type=BiasType.AVAILABILITY,
                severity=0.5,
                description=f"关于'{topic}'的检索全部来自高频记忆，可能被可得性启发扭曲",
                suggestion="主动搜索低频但可能更相关的记忆，引入基率信息",
            )
        return None

    def check_overconfidence(self, raw_confidence: float,
                            calibration_bias: float = 0.0) -> BiasWarning | None:
        """检测过度自信。"""
        if raw_confidence > 0.85 and calibration_bias > 0.15:
            return BiasWarning(
                bias_type=BiasType.OVERCONFIDENCE,
                severity=calibration_bias,
                description=f"高确信度({raw_confidence:.0%}) + 系统过度自信偏差({calibration_bias:.0%})",
                suggestion=f"校准后确信度应为 {max(0.05, raw_confidence - calibration_bias):.0%}",
            )
        return None

    def _add_warning(self, bias_type: BiasType, severity: float,
                    description: str, suggestion: str) -> BiasWarning:
        """添加警告并返回。"""
        w = BiasWarning(bias_type=bias_type, severity=severity,
                       description=description, suggestion=suggestion)
        self.warnings.append(w)
        self._bias_history[bias_type.value].append(severity)
        return w

    def get_active_warnings(self, max_age_seconds: float = 3600) -> list[BiasWarning]:
        """获取活跃的偏差警告。"""
        now = time.time()
        return [w for w in self.warnings if now - w.timestamp < max_age_seconds]

    def bias_report(self) -> dict:
        """偏差检测报告。"""
        if not self.warnings:
            return {"status": "clean", "message": "未检测到显著认知偏差"}

        # 按类型聚合
        by_type: dict[str, list[float]] = defaultdict(list)
        for w in self.warnings:
            by_type[w.bias_type.value].append(w.severity)

        summary = {}
        for btype, severities in by_type.items():
            summary[btype] = {
                "count": len(severities),
                "mean_severity": round(sum(severities) / len(severities), 3),
                "max_severity": round(max(severities), 3),
            }

        return {
            "status": "detected",
            "total_warnings": len(self.warnings),
            "active_warnings": len(self.get_active_warnings()),
            "by_type": summary,
        }

    def to_dict(self) -> dict:
        return {
            "warnings": [
                {
                    "bias_type": w.bias_type.value,
                    "severity": w.severity,
                    "description": w.description,
                    "suggestion": w.suggestion,
                    "timestamp": w.timestamp,
                }
                for w in self.warnings[-50:]  # 只保留最近50条
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BiasDetector":
        obj = cls()
        data = data or {}
        for w in (data.get("warnings") or []):
            obj.warnings.append(BiasWarning(
                bias_type=BiasType(w.get("bias_type", "overconfidence")),
                severity=w.get("severity", 0.5),
                description=w.get("description", ""),
                suggestion=w.get("suggestion", ""),
                timestamp=w.get("timestamp", time.time()),
            ))
        return obj


# ════════════════════════════════════════════════════════════
# 4. 前瞻记忆（Prospective Memory）
# ════════════════════════════════════════════════════════════

@dataclass
class ProspectiveTask:
    """前瞻记忆任务——记住将来要做的事。"""
    id: str
    description: str
    trigger_type: str  # "time" | "event" | "activity"
    trigger_value: str  # 触发条件（时间点/事件/活动）
    created_at: float = field(default_factory=time.time)
    deadline: float | None = None  # 截止时间
    priority: float = 0.5  # 优先级
    completed: bool = False
    completed_at: float | None = None

    @property
    def is_overdue(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline and not self.completed

    @property
    def urgency(self) -> float:
        """紧急度（0~1）：越接近截止时间越高。"""
        if self.deadline is None:
            return self.priority
        remaining = self.deadline - time.time()
        if remaining <= 0:
            return 1.0
        # 24小时内（agent时间）逐渐升高
        return min(1.0, 1.0 - remaining / 86400)


class ProspectiveMemory:
    """前瞻记忆管理器：记住将来要做的事，在合适时机主动提醒。

    对应人脑的"基于事件的前瞻记忆"（event-based PM）和
    "基于时间的前瞻记忆"（time-based PM）。
    """

    def __init__(self):
        self.tasks: dict[str, ProspectiveTask] = {}
        self.completed_count: int = 0

    def add_task(self, description: str, trigger_type: str = "event",
                trigger_value: str = "", deadline: float | None = None,
                priority: float = 0.5) -> ProspectiveTask:
        """添加一个前瞻记忆任务。"""
        import uuid
        task = ProspectiveTask(
            id=uuid.uuid4().hex[:8],
            description=description,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            deadline=deadline,
            priority=priority,
        )
        self.tasks[task.id] = task
        return task

    def complete_task(self, task_id: str) -> bool:
        """完成任务。"""
        if task_id in self.tasks:
            self.tasks[task_id].completed = True
            self.tasks[task_id].completed_at = time.time()
            self.completed_count += 1
            return True
        return False

    def get_due_tasks(self, current_activity: str = "",
                     current_time: float | None = None) -> list[ProspectiveTask]:
        """获取当前应触发的任务。"""
        now = current_time or time.time()
        due = []
        for task in self.tasks.values():
            if task.completed:
                continue
            # 基于时间的触发
            if task.trigger_type == "time" and task.deadline and now >= task.deadline:
                due.append(task)
            # 基于事件的触发
            elif task.trigger_type == "event" and task.trigger_value in current_activity:
                due.append(task)
            # 基于活动的触发
            elif task.trigger_type == "activity" and task.trigger_value == current_activity:
                due.append(task)
            # 逾期任务
            elif task.is_overdue:
                due.append(task)

        # 按紧急度排序
        due.sort(key=lambda t: -t.urgency)
        return due

    def get_pending_count(self) -> int:
        """获取待办任务数。"""
        return sum(1 for t in self.tasks.values() if not t.completed)

    def to_dict(self) -> dict:
        return {
            "tasks": {
                tid: {
                    "id": t.id,
                    "description": t.description,
                    "trigger_type": t.trigger_type,
                    "trigger_value": t.trigger_value,
                    "created_at": t.created_at,
                    "deadline": t.deadline,
                    "priority": t.priority,
                    "completed": t.completed,
                    "completed_at": t.completed_at,
                }
                for tid, t in self.tasks.items()
            },
            "completed_count": self.completed_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProspectiveMemory":
        obj = cls()
        data = data or {}
        obj.completed_count = data.get("completed_count", 0)
        for tid, t in (data.get("tasks") or {}).items():
            obj.tasks[tid] = ProspectiveTask(
                id=t.get("id", tid),
                description=t.get("description", ""),
                trigger_type=t.get("trigger_type", "event"),
                trigger_value=t.get("trigger_value", ""),
                created_at=t.get("created_at", time.time()),
                deadline=t.get("deadline"),
                priority=t.get("priority", 0.5),
                completed=t.get("completed", False),
                completed_at=t.get("completed_at"),
            )
        return obj


# ════════════════════════════════════════════════════════════
# 5. 精细复述（Elaborative Rehearsal）
# ════════════════════════════════════════════════════════════

class ElaborativeRehearsal:
    """精细复述引擎：回忆时把新信息与已有知识网络关联。

    对应人脑的"深度加工优势"（Craik & Lockhart）：
    - 浅层复述 = 机械重复（效果差）
    - 深层复述 = 语义关联（效果好）

    实现：回忆命中时，自动寻找关联记忆，构建关联网络。
    """

    def __init__(self):
        self.associations: dict[str, list[str]] = {}  # memory_id -> [related_id, ...]
        self.elaboration_count: int = 0

    def elaborate(self, memory_id: str,
                 all_memories: list,
                 cosine_sim_fn: Callable,
                 embed_fn: Callable,
                 threshold: float = 0.3,
                 max_associations: int = 3) -> list[str]:
        """为一条记忆构建关联网络（精细复语的核心）。

        找到与当前记忆语义相关的已有记忆，建立关联——
        回忆时通过关联网络激活更多相关记忆，增强记忆提取路径。
        """
        if not all_memories:
            return []

        target_mem = None
        for m in all_memories:
            if m.id == memory_id:
                target_mem = m
                break

        if target_mem is None:
            return []

        target_embed = target_mem.embedding
        related = []

        for m in all_memories:
            if m.id == memory_id:
                continue
            sim = cosine_sim_fn(target_embed, m.embedding)
            if sim >= threshold:
                related.append((m.id, sim))

        # 取最相关的 top-N
        related.sort(key=lambda x: -x[1])
        top_related = [mid for mid, _ in related[:max_associations]]

        self.associations[memory_id] = top_related
        self.elaboration_count += 1

        return top_related

    def get_associations(self, memory_id: str) -> list[str]:
        """获取一条记忆的关联记忆列表。"""
        return self.associations.get(memory_id, [])

    def to_dict(self) -> dict:
        return {
            "associations": {k: v for k, v in self.associations.items()},
            "elaboration_count": self.elaboration_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ElaborativeRehearsal":
        obj = cls()
        data = data or {}
        obj.associations = dict(data.get("associations") or {})
        obj.elaboration_count = data.get("elaboration_count", 0)
        return obj


# ════════════════════════════════════════════════════════════
# 6. 睡眠记忆分级（Memory Triage）
# ════════════════════════════════════════════════════════════

@dataclass
class TriageResult:
    """睡眠分级结果。"""
    memory_id: str
    importance: float
    emotion_boost: float
    consolidation_score: float  # 巩固优先级分数
    triage_category: str  # "high" | "medium" | "low"

    @property
    def should_consolidate(self) -> bool:
        return self.triage_category != "low"


class MemoryTriage:
    """睡眠记忆分级器：按价值分级处理，优先巩固高价值记忆。

    对应人脑的选择性巩固（selective consolidation）：
    - 情绪标记的记忆优先巩固
    - 与目标相关的记忆优先巩固
    - 近期频繁使用的记忆优先巩固
    - 低价值记忆可被遗忘（减轻认知负担）

    这是去除人类短处的关键：人类睡眠时不会平等对待所有记忆。
    """

    def __init__(self):
        self.triage_history: list[TriageResult] = []

    def triage(self, memories: list,
               goal_topics: list[str] | None = None,
               interest_getter: Callable[[str], float] | None = None,
              importance_weight: float = 0.4,
              emotion_weight: float = 0.3,
              recency_weight: float = 0.2,
              usage_weight: float = 0.1) -> list[TriageResult]:
        """对一批记忆进行睡眠分级。

        评分公式：
        consolidation_score = w1 * importance + w2 * emotion_boost
                            + w3 * recency + w4 * usage
        """
        results = []
        now = time.time()
        goal_topics = goal_topics or []

        for mem in memories:
            # 重要性分
            imp_score = mem.importance

            # 情绪加分（高唤醒→编码加深→优先巩固；复用 emotion.py 的 encoding_factor）
            emotion_boost = 0.0
            if hasattr(mem, 'emotion') and mem.emotion is not None:
                emotion_boost = max(0.0, encoding_factor(mem.emotion) - 1.0)

            # 近因分（越近越高；复用 decay.py 的 DecayScorer）
            if hasattr(mem, 'last_access') and hasattr(mem, 'created_at'):
                recency_score = DecayScorer().decay_only(
                    last_access=mem.last_access, now=now, tau_seconds=86400)
            else:
                recency_score = 0.5

            # 使用频率分
            if hasattr(mem, 'access_count'):
                usage_score = min(1.0, mem.access_count / 10.0)
            else:
                usage_score = 0.5

            # 目标相关加分
            if goal_topics and hasattr(mem, 'mtype'):
                # 语义记忆与目标更相关
                if mem.mtype.value == "semantic":
                    usage_score *= 1.2

            total = (importance_weight * imp_score +
                    emotion_weight * emotion_boost +
                    recency_weight * recency_score +
                    usage_weight * usage_score)

            # 分级
            if total >= 0.6:
                category = "high"
            elif total >= 0.35:
                category = "medium"
            else:
                category = "low"

            result = TriageResult(
                memory_id=mem.id,
                importance=imp_score,
                emotion_boost=emotion_boost,
                consolidation_score=total,
                triage_category=category,
            )
            results.append(result)

        # 按巩固分数排序
        results.sort(key=lambda r: -r.consolidation_score)
        self.triage_history.extend(results)
        return results

    def get_high_priority(self, results: list[TriageResult]) -> list[TriageResult]:
        return [r for r in results if r.triage_category == "high"]

    def get_medium_priority(self, results: list[TriageResult]) -> list[TriageResult]:
        return [r for r in results if r.triage_category == "medium"]

    def get_low_priority(self, results: list[TriageResult]) -> list[TriageResult]:
        return [r for r in results if r.triage_category == "low"]

    def to_dict(self) -> dict:
        return {
            "triage_history": [
                {
                    "memory_id": r.memory_id,
                    "importance": r.importance,
                    "emotion_boost": r.emotion_boost,
                    "consolidation_score": r.consolidation_score,
                    "triage_category": r.triage_category,
                }
                for r in self.triage_history[-50:]
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryTriage":
        obj = cls()
        data = data or {}
        for r in (data.get("triage_history") or []):
            obj.triage_history.append(TriageResult(
                memory_id=r.get("memory_id", ""),
                importance=r.get("importance", 0.0),
                emotion_boost=r.get("emotion_boost", 0.0),
                consolidation_score=r.get("consolidation_score", 0.0),
                triage_category=r.get("triage_category", "low"),
            ))
        return obj


# ════════════════════════════════════════════════════════════
# 7. 间隔重复优化（Spaced Repetition）
# ════════════════════════════════════════════════════════════

@dataclass
class SpacedRepetitionItem:
    """间隔重复条目——基于 SM-2 算法的变体。"""
    memory_id: str
    ease_factor: float = 2.5  # 易度因子（SM-2）
    interval: float = 0.0     # 当前间隔（秒）
    repetitions: int = 0      # 连续成功次数
    next_review: float = 0.0  # 下次复习时间戳
    last_review: float | None = None

    def review(self, quality: float, now: float | None = None) -> None:
        """复习一条记忆（quality: 0~1，回忆成功率）。"""
        now = now or time.time()
        self.last_review = now

        # SM-2 变体
        if quality >= 0.6:  # 成功回忆
            self.repetitions += 1
            if self.repetitions == 1:
                self.interval = 86400  # 1天
            elif self.repetitions == 2:
                self.interval = 86400 * 3  # 3天
            else:
                self.interval *= self.ease_factor
        else:  # 失败
            self.repetitions = 0
            self.interval = 86400  # 重置为1天

        # 更新易度因子
        self.ease_factor = max(1.3, self.ease_factor + (0.1 - (1 - quality) * 0.3))
        self.next_review = now + self.interval

    @property
    def is_due(self) -> bool:
        return time.time() >= self.next_review


class SpacedRepetitionOptimizer:
    """间隔重复优化器：在即将遗忘时精准触发复习。

    对应人脑的"间隔效应"（spacing effect）：
    - 分散复习 > 集中复习
    - 在即将遗忘时复习效果最好
    - 动态调整间隔（越容易的间隔越长）

    去除人类短处：人类常过度学习已掌握的，或忘记复习不熟悉的。
    """

    def __init__(self):
        self.items: dict[str, SpacedRepetitionItem] = {}
        self.review_count: int = 0

    def add_item(self, memory_id: str, ease_factor: float = 2.5) -> SpacedRepetitionItem:
        """添加一条需要间隔重复的记忆。"""
        item = SpacedRepetitionItem(
            memory_id=memory_id,
            ease_factor=ease_factor,
            next_review=time.time(),  # 新记忆立即复习
        )
        self.items[memory_id] = item
        return item

    def get_due_items(self, max_items: int = 5) -> list[SpacedRepetitionItem]:
        """获取到期的复习条目。"""
        due = [item for item in self.items.values() if item.is_due]
        # 按间隔长短排序（间隔越短 = 越需要复习）
        due.sort(key=lambda x: x.interval)
        return due[:max_items]

    def review(self, memory_id: str, quality: float, now: float | None = None) -> None:
        """复习一条记忆。"""
        if memory_id in self.items:
            self.items[memory_id].review(quality, now)
            self.review_count += 1

    def get_stats(self) -> dict:
        """间隔重复统计。"""
        total = len(self.items)
        if total == 0:
            return {"status": "empty"}

        due = len(self.get_due_items(max_items=total))
        avg_ef = sum(i.ease_factor for i in self.items.values()) / total

        return {
            "total_items": total,
            "due_now": due,
            "average_ease": round(avg_ef, 2),
            "total_reviews": self.review_count,
        }

    def to_dict(self) -> dict:
        return {
            "items": {
                mid: {
                    "memory_id": item.memory_id,
                    "ease_factor": item.ease_factor,
                    "interval": item.interval,
                    "repetitions": item.repetitions,
                    "next_review": item.next_review,
                    "last_review": item.last_review,
                }
                for mid, item in self.items.items()
            },
            "review_count": self.review_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpacedRepetitionOptimizer":
        obj = cls()
        data = data or {}
        obj.review_count = data.get("review_count", 0)
        for mid, item in (data.get("items") or {}).items():
            obj.items[mid] = SpacedRepetitionItem(
                memory_id=item.get("memory_id", mid),
                ease_factor=item.get("ease_factor", 2.5),
                interval=item.get("interval", 0.0),
                repetitions=item.get("repetitions", 0),
                next_review=item.get("next_review", time.time()),
                last_review=item.get("last_review"),
            )
        return obj
