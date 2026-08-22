"""记忆数据模型与分层存储：Hot / Warm / Cold 三层 + JSON 持久化。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .embedding import embed_text
from .emotion import Emotion
from .io_utils import FileLock, atomic_write_json


class ConcurrentWriteError(RuntimeError):
    """The persistence file changed after this store loaded it."""


class StoreCorruptionError(RuntimeError):
    """Neither the primary persistence file nor its backup is readable."""


class Tier(str, Enum):
    HOT = "hot"    # 工作记忆：高频使用，直接注入上下文
    WARM = "warm"  # 长时记忆：完整内容，参与检索评分
    COLD = "cold"  # 深藏记忆：压缩摘要，命中才唤醒

    def __str__(self) -> str:  # 便于打印
        return self.value


class MemType(str, Enum):
    """记忆内容类型：决定遗忘曲线时间常数 τ（技能慢、情景快）。"""

    SKILL = "skill"       # 程序性/技能：如做饭、弹琴、编程 → 慢衰减
    SEMANTIC = "semantic" # 语义/事实：如知识、身份、偏好 → 中衰减
    EPISODIC = "episodic" # 情景/事件：如"昨天吃了火锅" → 快衰减

    def __str__(self) -> str:
        return self.value


# 类型识别关键词（中文 + 英文，无 LLM 依赖）
_SKILL_KEYWORDS = [
    "学习", "练习", "学会", "掌握", "熟练", "技能", "技巧", "步骤", "方法",
    "怎么做", "如何", "教程", "流程", "操作", "做法", "菜谱", "食谱",
    "编程", "代码", "弹琴", "钢琴", "骑车", "游泳", "打球", "开车", "做饭",
    "learn", "practice", "skill", "tutorial", "recipe", "how to", "code",
]
_EPISODIC_KEYWORDS = [
    "昨天", "今天", "明天", "上周", "下周", "刚才", "早上", "晚上", "周末",
    "星期", "去了", "吃了", "买了", "看到", "见到", "听说", "发生", "遇到",
    "聚会", "旅行", "电影", "吃饭", "火锅", "出差", "旅游",
    "yesterday", "today", "tomorrow", "last week", "weekend", "happened",
    "saw", "ate", "bought", "met", "event", "trip",
]
_SEMANTIC_KEYWORDS = [
    "定义", "意思", "原理", "因为", "所以", "首都", "位于", "属于", "事实",
    "知识", "概念", "指", "意思是",
    "is ", "means", "defined", "fact", "knowledge", "concept", "because",
]
# 身份/偏好语句是稳定事实，倾向语义类（慢衰减）
_IDENTITY_PATTERNS = [
    "我叫", "我是", "我喜欢", "我不喜欢", "我的名字", "我住在", "我生日",
    "我的生日", "我工作", "我的工作",
]


def classify_memory_with_confidence(content: str, kind: str = "fact") -> tuple[MemType, float]:
    """关键词分类 + 置信度：返回 (类型, 置信度 0~1)。

    对话流水（turn）必然情景类，置信 1.0；其余按关键词打分取最高，
    置信度由命中数与领先幅度（margin）推导。
    """
    if kind == "turn":
        return MemType.EPISODIC, 1.0
    text = content.lower()
    scores = {
        MemType.SKILL: sum(1 for kw in _SKILL_KEYWORDS if kw in text),
        MemType.EPISODIC: sum(1 for kw in _EPISODIC_KEYWORDS if kw in text),
        MemType.SEMANTIC: sum(1 for kw in _SEMANTIC_KEYWORDS if kw in text),
    }
    if any(p in text for p in _IDENTITY_PATTERNS):
        scores[MemType.SEMANTIC] += 2
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return MemType.SEMANTIC, 0.4  # 无命中，低置信默认
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    conf = min(0.9, 0.55 + 0.05 * scores[best] + 0.1 * margin)
    return best, round(conf, 2)


def classify_memory(content: str, kind: str = "fact") -> MemType:
    """关键词分类（无置信度版本，向后兼容）。"""
    return classify_memory_with_confidence(content, kind)[0]


# 中文/英文重要性线索：命中即认为这段信息值得长期记住
_IMPORTANT_PATTERNS = [
    "我叫", "我是", "我的名字", "我喜欢", "我不喜欢", "我讨厌", "我最爱",
    "我住在", "我生日", "我的生日", "我工作", "我的工作", "我是做",
    "我养了", "我有", "我的手机", "我的邮箱", "我的地址", "我结婚",
    "我要", "我打算", "我计划", "我决定", "别忘", "记住", "important",
    "my name", "i like", "i love", "i work", "i live", "my birthday",
    "i hate", "i am", "i'm", "remember",
]


def estimate_importance(text: str) -> float:
    """基于关键词线索的启发式重要性打分（0~1）。

    无 LLM 依赖、可运行；工程上可替换为 LLM 判断（见 README）。
    """
    low = text.lower()
    score = 0.0
    for kw in _IMPORTANT_PATTERNS:
        if kw in low:
            score += 0.25
    # 长文本或包含个人细节的，略微加权
    if len(text) >= 40:
        score += 0.1
    return min(1.0, score + 0.1)  # 保底 0.1，避免纯零


@dataclass
class Memory:
    id: str
    content: str
    tier: Tier = Tier.WARM
    kind: str = "fact"  # fact=事实记忆 | turn=对话流水
    mtype: MemType = MemType.SEMANTIC  # 记忆类型：决定 τ（遗忘速度）
    mtype_confidence: float | None = None  # 分类置信度（LLM 或关键词给出）
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.1
    # Cold 层的压缩摘要；Warm 层为 None
    summary: str | None = None
    # Cold 摘要由哪些 Warm 记忆合并而来（"索引"指向底层内容）
    source_ids: list[str] = field(default_factory=list)
    # 深藏后原始内容是否仍保留（无损降权而非删除）
    originals: dict[str, str] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list, repr=False)
    # 观测历史：状态快照 [时间戳, 观测强度, 最后访问, 检索次数, 重要性]
    # 由 Agent 在创建/检索/升级时及每轮对话后的 _observe() 记录，
    # 用于画曲线与验证预测贴合度（fit_report）
    history: list[list] = field(default_factory=list, repr=False)
    # 记忆再巩固：回忆后进入"可塑窗口"的截止时间戳（0 = 不在窗口内）
    labile_until: float = 0.0
    # 再巩固修订次数与滚动日志 [时间, 可塑性, 漂移幅度, 重要性变化]
    revision_count: int = 0
    revisions: list[list] = field(default_factory=list, repr=False)
    # 类型迁移日志 [时间, "episodic→semantic"|"semantic→episodic", 语义化评分]
    # （情景语义化：被反复检索的 episodic 逐渐固化为 semantic，低频反向淡化）
    migrations: list[list] = field(default_factory=list, repr=False)
    # 技能类一致性校验日志 [时间, 触发查询, 结论, 相似度]
    # 结论：consistent=一致 / unknown=证据不足 / conflict=冲突留痕 / corrected=已修正
    checks: list[list] = field(default_factory=list, repr=False)
    # 上次从 Cold 唤醒的时间戳（None = 从未被唤醒）。
    # 长生命周期记忆的"复活"标记：/memories 据此展示继承的修订/历史条数，
    # 唤醒 → 再压回 Cold 往返不清除（同一记忆的复苏史可追溯）。
    awakened_at: float | None = None
    # 情绪调制信号（Emotion 或 None）。写入时标注/推断，检索时用于τ调制与一致性过滤
    emotion: object = None  # memagent.emotion.Emotion 类型；用 object 避免循环导入
    # 唤醒偏差观测日志 [时间戳, 实测偏差, 类型预期偏差, 唤醒时刻类型,
    # 埋藏时长 Δt, 埋藏时检索次数]：偏差 = 实测跳升强度 − 模型延续预测（未唤醒
    # 假想），并按该类型实测可塑性因子调制（dev 同时编码 τ 失准与可塑性，见
    # Agent._observe_awakening）。后两列供 τ↔可塑性联合估计器按衰减公式精确
    # 重算 τ 校正；旧四元组（无后两列）回退近似校正。
    awakenings: list[list] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.embedding:
            base = self.summary if (self.tier == Tier.COLD and self.summary) else self.content
            self.embedding = embed_text(base)

    def touch(self, now: float | None = None) -> None:
        """检索命中：测试效应——强化记忆（次数 +1、时间刷新）。

        now 由 Agent 注入（模拟时钟/对照实验），默认用真实时间。
        """
        self.access_count += 1
        self.last_access = now if now is not None else time.time()

    def demote_to_cold(
        self,
        summary: str,
        sources: list["Memory"] | None = None,
    ) -> None:
        """降级压缩：Warm → Cold，保留摘要与原始内容（无损降权）。

        originals 采用**合并**而非覆盖：被唤醒（Cold→Warm）后再次压回的记忆
        保留前一轮的深藏细节，Cold↔Warm 往返不丢数据（循环守护配套）。
        """
        self.tier = Tier.COLD
        self.summary = summary
        self.source_ids = [s.id for s in (sources or [self])]
        self.originals = {
            **self.originals,
            **{s.id: s.content for s in (sources or [self])},
        }
        self.embedding = embed_text(summary)  # 索引向量改为指向摘要

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier.value
        d["mtype"] = self.mtype.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        d = dict(d)
        d["tier"] = Tier(d["tier"])
        if "mtype" in d:
            d["mtype"] = MemType(d["mtype"])
        # JSON serializes Emotion as a dict; restore the runtime value used by
        # retrieval and tau modulation.
        emotion = d.get("emotion")
        if isinstance(emotion, dict):
            try:
                d["emotion"] = Emotion(
                    valence=emotion["valence"],
                    arousal=emotion["arousal"],
                    self_relevance=emotion["self_relevance"],
                    label=emotion.get("label", "neutral"),
                ).clamp()
            except (KeyError, TypeError, ValueError):
                d["emotion"] = None
        return cls(**d)


class MemoryStore:
    """三层记忆仓库，负责增删、升降级、持久化。

    meta 保存仓库级元数据（如学习器持久化的 learned_tau），随记忆一起落盘。
    """

    def __init__(self, path: str | None = None):
        self._memories: dict[str, Memory] = {}
        self.meta: dict = {}
        self.path = path
        self._file_signatures: dict[str, tuple[int, int] | None] = {}
        self._recovered_paths: set[str] = set()
        if path:
            self.load()

    # ---------- 基础操作 ----------

    def add(
        self,
        content: str,
        importance: float | None = None,
        tier: Tier = Tier.WARM,
        kind: str = "fact",
        mtype: MemType | None = None,
        mtype_confidence: float | None = None,
        now: float | None = None,
    ) -> Memory:
        """now: 注入时间戳（模拟时钟），默认真实时间。"""
        mem = Memory(
            id=uuid.uuid4().hex[:12],
            content=content,
            tier=tier,
            kind=kind,
            mtype=mtype or classify_memory(content, kind),
            mtype_confidence=mtype_confidence,
            importance=estimate_importance(content) if importance is None else importance,
            created_at=now if now is not None else time.time(),
            last_access=now if now is not None else time.time(),
        )
        self._memories[mem.id] = mem
        return mem

    def get(self, mem_id: str) -> Memory | None:
        return self._memories.get(mem_id)

    def remove(self, mem_id: str) -> bool:
        return self._memories.pop(mem_id, None) is not None

    def all(self) -> list[Memory]:
        return list(self._memories.values())

    def by_tier(self, tier: Tier) -> list[Memory]:
        return [m for m in self._memories.values() if m.tier is tier]

    def __len__(self) -> int:
        return len(self._memories)

    # ---------- 升降级 ----------

    def promote(self, mem: Memory, tier: Tier, now: float | None = None) -> None:
        """升级：Warm → Hot。Hot 层直接进上下文，不再参与向量检索。"""
        mem.tier = tier
        mem.touch(now)

    def demote_to_cold(self, mem: Memory, summary: str) -> None:
        """兼容入口：委托给 Memory.demote_to_cold。"""
        mem.demote_to_cold(summary)

    def awaken(self, mem: Memory, now: float | None = None) -> Memory:
        """唤醒：Cold → Warm，用摘要内容重建一条记忆（回忆即重建）。

        继承 mtype / kind / mtype_confidence / created_at / 观测轨迹 history /
        originals / 再巩固修订日志 revisions / 唤醒偏差观测 awakenings——重建是
        同一记忆的复活：类型决定遗忘速度（τ）与再巩固因子；created_at 保持原
        出生时间；history 让强度曲线与学习器（fit_report / 语义化评分，都从
        history 推导）不因唤醒而断层；originals 让合并簇的深藏细节随唤醒延续
        （配合 recall 的 move 语义，Cold 移除后不丢数据）；revisions 与
        awakenings 让可塑性学习器（learn_plasticity）的回忆事件与唤醒信号记录
        随唤醒延续（滚 12 条，多次 Cold↔Warm 往返的观测不断层）。
        access_count 在原值上 +1 = 本次唤醒本身也是一次检索（测试效应）。
        """
        new = Memory(
            id=uuid.uuid4().hex[:12],
            content=mem.summary or mem.content,
            tier=Tier.WARM,
            kind=mem.kind,
            mtype=mem.mtype,
            mtype_confidence=mem.mtype_confidence,
            importance=mem.importance,
            access_count=mem.access_count + 1,
            created_at=mem.created_at,
            last_access=now if now is not None else time.time(),
        )
        new.history = list(mem.history)        # 继承观测轨迹（复制，避免共享引用）
        new.originals = dict(mem.originals)    # 继承深藏细节（原 Cold 将被移除）
        new.revisions = list(mem.revisions)    # 继承再巩固修订日志（可塑性学习不断层）
        new.revision_count = mem.revision_count
        new.awakenings = list(mem.awakenings)  # 继承唤醒偏差观测（唤醒信号不断层，与修订同语义）
        # 同一记忆复活：情绪调制（τ 因子与一致性过滤）、技能校验/类型迁移履历
        # 随唤醒延续——否则一次 Cold↔Warm 往返就丢失这些状态
        new.emotion = mem.emotion
        new.checks = list(mem.checks)
        new.migrations = list(mem.migrations)
        new.awakened_at = now if now is not None else time.time()  # 复活标记
        self._memories[new.id] = new
        return new

    # ---------- 持久化 ----------

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        if not path:
            return
        target = Path(path).resolve()
        key = str(target)
        with FileLock(str(target) + ".lock"):
            current = self._signature(target)
            expected = self._file_signatures.get(key, current)
            if current != expected:
                raise ConcurrentWriteError(
                    f"persistence changed since load; refusing to overwrite: {target}"
                )
            payload = {
                "meta": self.meta,
                "memories": [m.to_dict() for m in self._memories.values()],
            }
            recovered = key in self._recovered_paths
            atomic_write_json(target, payload, backup=not recovered)
            if recovered:
                atomic_write_json(target.with_suffix(target.suffix + ".bak"), payload)
                self._recovered_paths.discard(key)
            self._file_signatures[key] = self._signature(target)

    def load(self, path: str | None = None) -> None:
        path = path or self.path
        if not path:
            return
        target = Path(path).resolve()
        key = str(target)
        if not target.exists():
            self._file_signatures[key] = None
            return
        try:
            with target.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            backup = target.with_suffix(target.suffix + ".bak")
            try:
                with backup.open(encoding="utf-8") as f:
                    data = json.load(f)
                self._recovered_paths.add(key)
            except (FileNotFoundError, json.JSONDecodeError) as backup_error:
                raise StoreCorruptionError(
                    f"invalid persistence file and backup: {target}"
                ) from backup_error
        if isinstance(data, dict):
            self.meta = data.get("meta") or {}
            records = data.get("memories", [])
        else:  # 旧格式：裸记忆数组
            self.meta = {}
            records = data
        self._memories = {d["id"]: Memory.from_dict(d) for d in records}
        self._file_signatures[key] = self._signature(target)

    @staticmethod
    def _signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size
