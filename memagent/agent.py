"""对话 Agent：检索式回复 + 分层记忆 + 睡眠巩固 + 命令行交互。

无 LLM 依赖即可运行（模板回复）；重要性打分与回复生成均预留了
可替换成 LLM 的钩子（见 README）。
"""

from __future__ import annotations

import math
import random
import re
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .compression import extractive_summary, merge_similar
from .compat import call_responder
from .decay import DecayScorer, ScorerConfig
from .embedding import cosine_similarity, embed_text, ngrams, normalize
from .innate import INNATE_DEFAULTS, InnateBounds, TIME_SCALE, default_innate_bounds
from .emotion import (
    Emotion, BASIC_EMOTIONS, infer_emotion,
    tau_factor, encoding_factor, congruence_factor,
)
# 情绪参数约定：
#   emotion 不传（_UNSET）→ 自动从文本关键词推断
#   emotion=None         → 显式要求无情绪调制
#   emotion=Emotion(...) → 显式标注
_UNSET = object()
from .graph import KnowledgeGraph
from .growth import GrowthEngine
from .cognition import Cognition
from .curiosity import CuriosityDrivenExplore
from .analogy import AnalogyTransfer
from .social import SocialLearner
from .interest import InterestVector
from .io_utils import FileLock, LockTimeoutError, atomic_write_text
from .llm import LLMClassifier
from .memory import MemType, Memory, MemoryStore, Tier
from .synonyms import SHORT_QUERY_LEN, expand_query, substring_priority_order

QUERY_BOOST_HOT = 1.25   # Hot 记忆（工作记忆）的检索加成
TURN_PENALTY = 0.5       # 对话流水记忆的检索惩罚（事实记忆权重更高）
STRENGTH_FLOOR = 0.2     # 检索强度下限，避免相似度被完全吞掉
DEDUP_THRESHOLD = 0.92   # 与已有记忆几乎相同 → 合并（去重）


def _safe_title(title: str) -> str:
    """作品名 → 安全目录名（去掉非法路径字符与空白）。"""
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", title) or "unnamed"


@dataclass
class AgentConfig:
    hot_after_access: int = 5               # 检索 ≥ 该次数 → 升级 Hot
    cold_max_access: int = 2                # 且累计检索 ≤ 该次数（低频）才压缩
    max_hot: int = 10                       # Hot 层容量
    k: int = 3                              # 回复引用相关记忆条数
    sleep_interval_turns: int = 8           # 每 N 轮对话自动睡眠巩固一次
    query_expansion: bool = True            # 检索同义扩展（问法与记忆措辞不同也能命中）
    rerank_short_query: bool = True         # 短查询子串优先重排总开关（< rerank_short_len 字）
    rerank_short_len: int = SHORT_QUERY_LEN  # 短词阈值：查询少于该字数视为"短"，可自定义
    # --- 按记忆类型分遗忘曲线：技能慢、语义中、情景快 ---
    # 所有时间参数已在人类级秒数上乘以 TIME_SCALE（默认 1/86400，1 agent-秒≈1人类-天）
    # 使学习器几秒即可跑完完整遗忘周期，测试可行
    tau_by_type: dict = field(default_factory=lambda: {
        MemType.SKILL: 60 * 24 * 3600 * TIME_SCALE,   # 技能：慢（60天→~3530s agent-时间）
        MemType.SEMANTIC: 14 * 24 * 3600 * TIME_SCALE,  # 语义：中（14天）
        MemType.EPISODIC: 3 * 24 * 3600 * TIME_SCALE,   # 情景：快（3天）
    })
    tau_seconds: float = 7 * 24 * 3600 * TIME_SCALE     # 回退 τ（7天→~4.2s）
    # --- 参数自适应学习器：按实测 τ 自动校准配置 ---
    tau_learning: bool = True               # 总开关（关闭则保留显式配置）
    tau_min_segments: int = 3               # 每类型最少干净衰减段才更新
    tau_learning_rate: float = 0.3          # 最大学习率（EMA α，再乘置信度）
    tau_min_seconds: float = 1.0 * TIME_SCALE        # 学习后 τ 下限
    tau_max_seconds: float = 10 * 365 * 86400 * TIME_SCALE  # 学习后 τ 上限
    # 唤醒偏差作为 τ 的第二观测源：实测跳升深于类型预期（dev > expected）→
    # 该类型衰减比信念快 → τ 下调（与干净段反推互补，见 _tau_awakening_estimate）。
    tau_from_awakenings: bool = True
    tau_min_awakenings: int = 3            # 每类型至少 N 次唤醒观测才参与 τ 估计
    awakening_tau_gain: float = 1.0        # 偏差比值放大指数：τ_est = τ × (expected/dev)^gain
    # 真实遗忘环境：观测采样用它、预测用 tau_by_type。留空=观测与预测一致；
    # 设置后可通过贴合度报告验证 τ 配置是否失准（见 fit_report）。
    true_tau_by_type: dict = field(default_factory=dict)
    cold_after_seconds: float | None = None  # 绝对压缩阈值；None → 按类型 τ 推导
    cold_after_tau: float = 2.0             # 无访问超过 2×τ 即可压缩进 Cold
    # --- 睡眠回放（锐波涟漪模拟）：白天经历按时间顺序重放 = 再激活 ---
    # 每次重放 access_count +1 并记录观测采样（强度微调 + 语义化评分贡献）；
    # 完整睡眠（sleep() 不带时长）重放全部候选；sleep(duration) 按
    # replay_per_second 折算预算，未回放的候选次日"模糊"（importance 缩放）。
    replay: bool = True
    replay_window_seconds: float = 24 * 3600 * TIME_SCALE  # "白天经历"窗口（24h→1s）
    replay_per_second: float = 1.0            # 睡眠每秒可重放的条数（中断时预算折算）
    replay_fog_factor: float = 0.9            # 未回放候选的次日模糊系数（importance 缩放）
    # --- 记忆再巩固：回忆后按重要程度微调原始记忆 ---
    reconsolidate: bool = True              # 总开关
    reconsolidation_window: float = 6 * 3600 * TIME_SCALE  # 可塑窗口时长（6h→~0.067s）
    labile_bonus: float = 1.0               # 可塑窗口内的漂移幅度加成
    content_drift: float = 0.15             # 单次回忆的语义漂移幅度（低重要性更明显）
    importance_drift: float = 0.05          # 单次回忆的重要性微调幅度
    freeze_importance: float = 0.8          # 重要性 ≥ 该值 → 核心记忆，完全冻结
    importance_floor: float = 0.05          # 重要性漂移下限
    # 按类型的再巩固因子：技能类回忆时高度稳定，情景类易被情境改写
    reconsolidation_by_type: dict = field(default_factory=lambda: {
        MemType.SKILL: {"drift": 0.15, "importance": 0.2},    # 技能：几乎不漂移
        MemType.SEMANTIC: {"drift": 1.0, "importance": 1.0},  # 语义：基准
        MemType.EPISODIC: {"drift": 2.5, "importance": 1.5},  # 情景：易被情境改写
    })
    # --- 类型迁移：情景记忆语义化 ---
    # 被反复检索的 episodic 记忆逐渐固化为 semantic（"我经常去爬山"替代
    # 50 次具体爬山）；低频 semantic 反向淡化为 episodic。评分 = 近期检索事件
    # 的指数衰减加权和（从观测历史推导，无需额外状态），双阈值滞回防振荡。
    semanticize: bool = True                  # 总开关
    semanticization_tau_seconds: float = 7 * 24 * 3600 * TIME_SCALE  # 检索事件"新鲜度"衰减
    semanticize_threshold: float = 3.0        # episodic 评分 ≥ 该值 → 语义化
    desemanticize_threshold: float = 0.8      # semantic 评分 < 该值 → 淡化为 episodic
    # --- 再巩固因子自适应：从观测估计每类型实际可塑性 ---
    # 像学习器调 τ 一样自动校准 reconsolidation_by_type：修订日志记录每次回忆
    # 事件实际应用的因子（实测可塑性），EMA + 置信度门控把配置因子推向实测值。
    plasticity_learning: bool = True          # 总开关
    plasticity_min_events: int = 3            # 每类型每通道至少 N 次回忆事件才更新
    plasticity_learning_rate: float = 0.3     # EMA 最大学习率（再乘置信度）
    plasticity_min: float = 0.0               # 因子下限（0 = 完全稳定）
    plasticity_max: float = 5.0               # 因子上限
    # 唤醒偏差作为可塑性观测：recall 时记录实测跳升 vs 模型延续预测的偏差，
    # 换算成漂移因子代理样本喂给 learn_plasticity（唤醒越剧烈 → 可塑性越活跃）。
    plasticity_from_awakenings: bool = True
    awakening_dev_gain: float = 2.0           # 偏差偏离典型值的放大系数（可调）
    # --- τ↔可塑性联合估计器：一次唤醒事件同时更新 τ 与 drift ---
    # dev 现在同时编码 τ 失准与可塑性（观测层调制，见 _observe_awakening）：
    #   dev = base(τ真实) · (1 + g·(p实测 − 1))
    #   expected = base(τ模型) · (1 + g·(p信念 − 1))
    # 两式同构 → 比值 = [base(τ)/base(τ)]·[可塑性因子] 乘法可分；单事件两个未知
    # （τ 失准、可塑性偏差）数学上不可完全分离 → 顺序归因 + 跨轮耦合：τ 通道
    # 先拿比值（埋藏深度主导）；drift 通道拿 τ 校正后的残余（上一轮 τ 估计反哺），
    # 而 expected 的可塑性刻度随信念收敛 → 比值中的可塑性因子逐年消融 →
    # 两路 EMA 相互加速收敛（彼此的进展清洗对方的信号，见 _joint_awakening_estimates）。
    joint_awakening: bool = True              # 联合估计器总开关（关闭 → 回退两路独立代理）
    awakening_plasticity_gain: float = 0.3    # 唤醒跳升中可塑性的调制系数 g
    # 真实可塑性环境：实际应用再巩固时用它（隐藏），学习器据此校准配置因子；
    # 留空 = 实际与配置一致（自洽，无可学），设置后可验证/校准配置因子。
    true_reconsolidation_by_type: dict = field(default_factory=dict)
    # --- 出厂边界（innate bounds）：进化硬约束，学习器只能微调不可翻转方向 ---
    # 类比人脑的出厂硬件：恐惧 τ 几乎无限、技能漂移接近零——经验数据不能颠覆这些方向。
    # 详见 memagent.innate 模块。全局 tau_min/max / plasticity_min/max 仍作为最终安全网生效。
    innate_bounds: dict = field(default_factory=lambda: dict(INNATE_DEFAULTS))
    # --- 场景重建：检索命中时把相关记忆片段组合成连贯场景 ---
    # 人脑回忆的是片段组合（场景重建）而非单条事实；compose_scene 以检索命中
    # 为种子，扩展彼此相关的片段（嵌入相似或共享 n-gram，二者取大——字符嵌入
    # 对措辞不同的相关中文片段相似度低，纯余弦会漏掉同场景但不同词的片段），
    # 按经历顺序（created_at）拼成叙事。
    scene_reconstruction: bool = True        # 总开关（compose_scene / respond 场景展示）
    scene_similarity: float = 0.2            # 片段相关阈值（max(余弦, n-gram Jaccard)）
    scene_max_fragments: int = 6             # 场景最多片段数（种子优先）
    scene_time_window: float = 90 * 24 * 3600  # 片段出生时间跨度上限（跨年不属于同一场景）
    scene_reconsolidates: bool = True        # 场景重建对扩展片段是否再巩固（重建会微调片段）
    # --- 自主演化：人设随记忆 + 联网资料自主成长 ---
    # evolve() 以当前人设档案 + 最近记忆 + 联网搜索为上下文，让 LLM 提出新的
    # 作品设定并 remember_setting 入库——下一轮 persona_sheet() 即包含新设定。
    # evolve_on_sleep=True 时每次睡眠巩固自动触发一次演化（默认关，避免在无
    # 意图的场景静默调用 LLM；交互 CLI 配 persona 或 --auto 模式会自动开启）。
    evolve_on_sleep: bool = False
    evolve_search: bool = True            # 演化时是否联网搜索创作资料
    evolve_max_settings: int = 6          # 单次演化最多吸收的新设定条数
    # --- 每日创作产出：设定演化 → 正文写作闭环 ---
    # write_chapter() 以当前人设档案为上下文续写下一章，落盘 works/<书名>/ 目录，
    # 并把章节进度回写进设定记忆（“已连载 N 章”）。--auto 模式每轮先演化再写作。
    chapter_words: int = 2200             # 单章目标字数（约）——千字章装不下剧情，
                                          # 只能装"一段对话+一个钩子"（旧书实测根因之一）
    chapter_min_ratio: float = 0.9        # 低于目标字数 90% 视为截断响应，不落盘
    chapter_save_dir: str = "works"       # 章节落盘根目录（相对 cwd）
    chapter_with_web: bool = False        # 写作时是否联网查资料（灵感参考）
    llm_long_timeout: float = 150.0       # 演化/写作等长任务请求超时（秒）
    llm_long_max_tokens: int = 6144       # 长文本输出上限（默认 1024 会把章节正文拦腰截断，触发写章级整轮重试）
    chapter_retries: int = 3              # 写章整轮重试次数（空/截断/未收尾时重跑，章号不占号）
    chapter_retry_delay: float = 30.0     # 写章级重试间隔（秒）
    llm_retries: int = 3                  # 单次 LLM 调用空回复/异常重试次数
    llm_retry_delay: float = 8.0          # 单次调用重试间隔（秒）

    def tau_for(self, mtype: MemType) -> float:
        """某类型记忆的遗忘时间常数 τ。"""
        return self.tau_by_type.get(mtype, self.tau_seconds)

    def reconsolidation_factor(self, mtype: MemType, channel: str) -> float:
        """某类型记忆在再巩固中的配置因子（模型信念，channel: "drift" | "importance"）。"""
        d = self.reconsolidation_by_type.get(mtype, {})
        return float(d.get(channel, 1.0))

    def applied_factor(self, mtype: MemType, channel: str) -> float:
        """实际应用的再巩固因子：配置了 true_reconsolidation_by_type（隐藏的
        真实可塑性环境）时用它，否则用配置因子（模型信念）。

        与 true_tau_by_type 同理：观测按"真实"环境发生，预测按配置——
        两者的偏差就是因子失准的信号，供 learn_plasticity 校准。
        """
        t = self.true_reconsolidation_by_type.get(mtype)
        if t is not None and channel in t:
            return float(t[channel])
        return self.reconsolidation_factor(mtype, channel)


@dataclass
class Retrieved:
    memory: Memory
    relevance: float      # 查询相似度
    strength: float       # 记忆强度（遗忘曲线得分归一化后）
    total: float          # relevance × strength
    via_summary: bool     # 是否命中 Cold 摘要


@dataclass
class SceneFragment:
    """场景里的一个记忆片段。"""

    memory: Memory
    rel: float            # 与场景的相关度（种子=检索 rel；扩展=与种子的最大相似度）
    strength: float       # 片段当前强度
    via_summary: bool     # Cold 摘要片段（可 /recall 唤醒细节）
    role: str = ""        # 时序角色：开头 / 中间 / 结尾

    @property
    def text(self) -> str:
        """片段展示文本：Warm 用完整内容，Cold 用压缩摘要。"""
        return self.memory.summary or self.memory.content


@dataclass
class Scene:
    """组合而成的连贯场景：相关记忆片段按经历顺序拼接。"""

    query: str
    title: str
    fragments: list[SceneFragment]
    narrative: str        # 时序连接词拼成的连贯叙事
    strength: float       # rel 加权平均强度（场景整体显著性）
    coherence: float      # 片段间平均最大相似度（连贯性度量）
    created: float        # 重建时刻

    @property
    def count(self) -> int:
        return len(self.fragments)


def _fragment_text(m: Memory) -> str:
    """片段文本：Warm 用完整内容，Cold 用压缩摘要（与索引向量一致）。"""
    return m.summary or m.content


def _fragment_relatedness(a: Memory, b: Memory, require_shared: bool = False) -> float:
    """两个记忆片段的相关度：嵌入余弦与 n-gram 共享度取大。

    字符 n-gram 嵌入对"措辞不同但同主题"的中文片段相似度很低（
    「我在西湖边散步」vs「西湖边的风很舒服」余弦仅 0.42），纯余弦会
    把同场景但不同词的片段漏掉；共享 n-gram（Jaccard）补充主题词重叠
    信号（同场景片段常共享地点/人物等词）。文本统一取 summary-or-content
    （Cold 的嵌入与索引都指向摘要，n-gram 也必须用摘要，否则两套语义脱节）。

    require_shared=True（扩展门控）时**必须共享至少一个 n-gram** 才算
    相关——哈希嵌入的余弦在 0.1~0.25 区间有纯碰撞噪声（实测措辞完全
    无关的片段也能到 0.224），不要求共享主题词会把噪声片段拉进场景。
    """
    cos = cosine_similarity(a.embedding, b.embedding)
    ga, gb = set(ngrams(_fragment_text(a))), set(ngrams(_fragment_text(b)))
    if ga and gb:
        inter = ga & gb
        if not inter and require_shared:
            return 0.0
        jac = len(inter) / len(ga | gb)
    else:
        jac = 0.0
    return max(cos, jac)


def format_scene(scene: Scene) -> str:
    """场景的中文展示（CLI /scene 与 demo 使用）。"""
    lines = [
        f"场景 · {scene.title}（{scene.count} 个片段 · "
        f"连贯度 {scene.coherence:.2f} · 强度 {scene.strength:.2f}）",
        f"  重建叙事：{scene.narrative}",
    ]
    for f in scene.fragments:
        note = " [Cold 摘要]" if f.via_summary else ""
        lines.append(f"  [{f.role}] {f.text}{note}（强度 {f.strength:.2f}）")
    return "\n".join(lines)


_AWAKENING_EPS = 1e-6  # 唤醒信号方向判别容差（与 recall_curve_check 一致）


def awakening_signal_stats(agent: "MemoryAgent",
                           window_seconds: float | None = None,
                           since: float | None = None,
                           until: float | None = None,
                           now: float | None = None,
                           exclude: set | None = None) -> dict:
    """从记忆库扫描 awakenings，按类型聚合实测/预期偏差分布与信号方向一致性。

    时间窗：默认全时段。`window_seconds` = 只看最近 N 秒（相对 now）；或直接给
    绝对窗口 `since`（含）/ `until`（不含，awakenings[0] 为事件时刻）做任意时段
    对比。`exclude` = {(memory_id, 事件在 _awakening_events 过滤列表中的序号)}
    的排除集——排除事件不参与统计（冲突剔除假设检验：仪表盘圈出的可疑事件
    排除后重算方向一致性）。每类型返回 {"events", "dev": [min, 中位, max],
    "expected": [min, 中位, max], "ratio_med": 中位 dev/expected,
    "up"/"down"/"flat": 方向计数, "dominant": 主导方向, "consistency":
    主导方向占比}；无观测的类型返回 {"events": 0}；旧格式三元组（无类型预期
    偏差）跳过。方向语义：dev > expected → up（唤醒比类型预期剧烈 → τ 应下调）；
    < → down（温和 → 反向校准）；≈ → flat（自洽，无学习信号）。这是唤醒信号
    统计的单一事实源（CLI /types /signal、仪表盘画像列、收工验证共用）。
    """
    now = now if now is not None else agent._now()
    if window_seconds is not None:
        since = now - window_seconds
    exclude = exclude or set()
    rows: dict[str, list] = {t.value: [] for t in MemType}
    for m in agent.store.all():
        filtered = [aw for aw in m.awakenings if len(aw) >= 4]
        for f_idx, aw in enumerate(filtered):
            if (m.id, f_idx) in exclude:
                continue
            if aw[3] not in rows:
                continue
            ts = float(aw[0])
            if since is not None and ts < since:
                continue
            if until is not None and ts >= until:
                continue
            dev, exp = float(aw[1]), float(aw[2])
            rows[aw[3]].append((dev, exp))
    out: dict = {}
    for t, rs in rows.items():
        if not rs:
            out[t] = {"events": 0}
            continue
        devs = [d for d, _ in rs]
        exps = [e for _, e in rs]
        dirs = [
            "up" if d > e + _AWAKENING_EPS else ("down" if d < e - _AWAKENING_EPS else "flat")
            for d, e in rs
        ]
        counts = {k: dirs.count(k) for k in ("up", "down", "flat")}
        dominant = max(counts, key=counts.get)
        out[t] = {
            "events": len(rs),
            "dev": [round(min(devs), 4), round(statistics.median(devs), 4),
                    round(max(devs), 4)],
            "expected": [round(min(exps), 4), round(statistics.median(exps), 4),
                         round(max(exps), 4)],
            "ratio_med": (round(statistics.median(d / e for d, e in rs), 3)
                           if all(e > 0 for _, e in rs) else None),
            **counts,
            "dominant": dominant,
            "consistency": round(counts[dominant] / len(rs), 2),
        }
    return out


def awakening_signal_periods(agent: "MemoryAgent",
                             recent_seconds: float = 30 * 86400 * TIME_SCALE,
                             now: float | None = None,
                             exclude: set | None = None) -> dict:
    """对比最近 N 天与更早两段时期的唤醒信号——方向一致性是否随时间漂移。

    recent = 近 recent_seconds 内的唤醒观测，earlier = 更早的全部观测。
    `exclude` = {(memory_id, 事件序号)} 排除集（同 awakening_signal_stats）。
    每类型返回 {"recent": 近期统计, "earlier": 早期统计, "verdict": 判定,
    "direction_changed": 主导方向是否翻转, "consistency_delta": 一致性差值
    （近期 - 早期）}；verdict ∈ 稳定 | 方向翻转 | 一致性变化 | 仅近期有观测 |
    仅早期有观测 | 无观测（方向翻转 = 两段主导方向不同；一致性变化 = 方向未变
    但一致性差 ≥ 0.2；其余为稳定）。用于判断"信号是否随时间漂移"——早期与近期
    结论一致 = 校准方向稳定，翻转 = 类型行为发生了真实变化（需重新审视配置）。
    """
    now = now if now is not None else agent._now()
    cutoff = now - recent_seconds
    recent = awakening_signal_stats(agent, since=cutoff, now=now, exclude=exclude)
    earlier = awakening_signal_stats(agent, until=cutoff, now=now, exclude=exclude)
    out: dict = {"now": now, "recent_seconds": recent_seconds,
                 "cutoff": cutoff, "by_type": {}}
    for t in MemType:
        r, e = recent[t.value], earlier[t.value]
        n_r, n_e = r.get("events", 0), e.get("events", 0)
        if n_r == 0 and n_e == 0:
            out["by_type"][t.value] = {"recent": r, "earlier": e,
                                        "verdict": "无观测",
                                        "direction_changed": False,
                                        "consistency_delta": None}
            continue
        if n_r == 0:
            out["by_type"][t.value] = {"recent": r, "earlier": e,
                                        "verdict": "仅早期有观测",
                                        "direction_changed": False,
                                        "consistency_delta": None}
            continue
        if n_e == 0:
            out["by_type"][t.value] = {"recent": r, "earlier": e,
                                        "verdict": "仅近期有观测",
                                        "direction_changed": False,
                                        "consistency_delta": None}
            continue
        changed = r["dominant"] != e["dominant"]
        delta = round(r["consistency"] - e["consistency"], 2)
        if changed:
            verdict = "方向翻转"
        elif abs(delta) >= 0.2:
            verdict = "一致性变化"
        else:
            verdict = "稳定"
        out["by_type"][t.value] = {
            "recent": r, "earlier": e, "verdict": verdict,
            "direction_changed": changed, "consistency_delta": delta,
        }
    return out


# ---------- τ 学习器两路信号健康检查 ----------


def _tau_dir(tau_est: float, cfg_tau: float) -> str:
    """干净段反推的方向：实测 τ < 配置 → 该类型忘得比信念快 → τ 应下调。"""
    if abs(tau_est - cfg_tau) / max(cfg_tau, 1e-9) < 0.005:
        return "flat"
    return "down" if tau_est < cfg_tau else "up"


def _ratio_dir(ratio: float) -> str:
    """唤醒偏差的方向：比值 > 1 → 唤醒比类型预期剧烈 → 忘得比信念快 → τ 应下调。"""
    if abs(ratio - 1.0) < 0.005:
        return "flat"
    return "down" if ratio > 1.0 else "up"


def _adjust_suggestion(h: dict) -> str:
    """从两路信号一致性推导行动建议（供导出 CSV / JSON / 仪表盘 / 收工验证共用）。

    - agree + 同向非 flat → 直接给出调整方向：down = τ↓（配置偏大）、
      up = τ↑（配置偏小）；agree + 双 flat → 已校准（无需动作）；
    - conflict → 需检查（两路互相矛盾，先排查观测污染/事件注入再调参）；
    - one_sided → 需补观测（只有一路信号，无法交叉印证，积累另一路）；
    - no_data → 无信号（不动作）。
    """
    cs = h["consistency"]
    if cs == "no_data":
        return "无信号"
    if cs == "conflict":
        return "需检查"
    if cs == "one_sided":
        return "需补观测"
    d = h["clean"]["direction"] or h["awakening"]["direction"]
    if d == "down":
        return "τ↓"
    if d == "up":
        return "τ↑"
    return "已校准"


def _suggest_confidence(h: dict, tau_min_segments: int,
                        tau_min_awakenings: int) -> str:
    """建议的置信度（证据强度）：agree 时按两路观测条数给 强/中/弱——避免
    单条观测就建议调参；非 agree 固定值（冲突/无信号 → —、单源 → 弱）。

    - 强：干净段 ≥ 2×tau_min_segments **且** 唤醒 ≥ 2×tau_min_awakenings
      （两路都充分采样，建议可信）；
    - 中：两路都过各自门控（≥ tau_min_segments / ≥ tau_min_awakenings），
      未达强；
    - 弱：agree 但唤醒观测不足（< tau_min_awakenings）——单条观测不足以
      调参，建议先积累观测再校准。
    """
    cs = h["consistency"]
    if cs == "conflict" or cs == "no_data":
        return "—"
    if cs == "one_sided":
        return "弱"
    cl_n, aw_n = h["clean"]["n"], h["awakening"]["n"]
    if cl_n >= 2 * tau_min_segments and aw_n >= 2 * tau_min_awakenings:
        return "强"
    if cl_n >= tau_min_segments and aw_n >= tau_min_awakenings:
        return "中"
    return "弱"


_TAU_DIR_CN = {
    "down": "应下调（配置偏大）",
    "up": "应上调（配置偏小）",
    "flat": "无明显偏差（已校准）",
}

# 行动项中文语义（τ↓/τ↑/需检查）——actions 数组与终端行动清单共用
_ADJUST_CN = {
    "τ↓": "配置偏大 · 忘得比信念快",
    "τ↑": "配置偏小 · 忘得比信念慢",
    "需检查": "两路信号冲突，先排查再调参",
}


def _event_ratio_dir(ratio):
    """单条唤醒事件的比值方向（终端明细 / warnings / actions 共用闸门
    1.05/0.95）：>1.05 → down（应下调）、<0.95 → up（应上调）、其余 flat、
    无比值 legacy。"""
    if ratio is None:
        return "legacy"
    if ratio > 1.05:
        return "down"
    if ratio < 0.95:
        return "up"
    return "flat"


def _awakening_row_map(agent: "MemoryAgent", exclude: set | None = None) -> dict[tuple, int]:
    """事件级导出行号表：{(memory_id, ts): 行号}——全局按 (ts, memory_id)
    排序、表头第 1 行，与 `--export-signals` 的 events CSV / 仪表盘证据行
    行号**同一算法（收敛于此，单一实现）**。与事件级导出同门控（len≥4）；
    exclude = {(memory_id, 事件序号)} 排除集（排除后行号自然重排）。
    """
    exclude = exclude or set()
    events = []
    for mem in sorted(agent.store.all(), key=lambda m: m.id):
        filtered = [aw for aw in mem.awakenings if len(aw) >= 4]
        for f_idx, aw in enumerate(filtered):
            if (mem.id, f_idx) in exclude:
                continue
            events.append({"memory_id": mem.id, "ts": aw[0]})
    events.sort(key=lambda e: (e["ts"], e["memory_id"]))
    return {(e["memory_id"], e["ts"]): i + 2 for i, e in enumerate(events)}


def _conflict_events(agent: "MemoryAgent", mtype: MemType,
                     row_map: dict | None = None,
                     exclude: set | None = None) -> list[dict]:
    """该类型的冲突成因唤醒事件明细（warnings/actions 共用）：
    [{memory_id, index, ts, ts_relative_seconds, dev, expected, ratio,
    direction, row}]。

    - index = 事件在该记忆 `_awakening_events` 过滤列表中的序号（与主图
      data-evi 同语义——仪表盘点击可定位到对应唤醒点）；
    - row = 事件级导出行号（_awakening_row_map 同算法，直接进 health——
      导出/仪表盘/终端零重复）；
    - exclude = {(memory_id, 事件序号)} 排除集（排除后从明细与行号重算中消失）；
    - 行类型取日志记录的 mtype（与唤醒信号统计同门控：len≥4、dev/expected
      都 > 0 才给比值）；direction 用 _event_ratio_dir 闸门。
    按 (ts, memory_id) 排序，与事件级导出同序。
    """
    row_map = row_map if row_map is not None else _awakening_row_map(agent, exclude)
    exclude = exclude or set()
    now = agent._now()
    evs: list[dict] = []
    for mem in agent.store.all():
        filtered = [aw for aw in mem.awakenings if len(aw) >= 4]
        for idx, aw in enumerate(filtered):
            if (mem.id, idx) in exclude or aw[3] != mtype.value:
                continue
            ts, dev, expected = aw[0], float(aw[1]), float(aw[2])
            ratio = round(dev / expected, 4) if dev > 0 and expected > 0 else None
            evs.append({"memory_id": mem.id, "index": idx, "ts": ts,
                        "ts_relative_seconds": round(ts - now, 1),
                        "dev": round(dev, 4), "expected": round(expected, 4),
                        "ratio": ratio, "direction": _event_ratio_dir(ratio),
                        "row": row_map.get((mem.id, ts))})
    evs.sort(key=lambda e: (e["ts"], e["memory_id"]))
    return evs


def clash_event_keys(agent: "MemoryAgent", health: dict,
                     exclude: set | None = None) -> set:
    """按 dir ≠ clean 自动圈定冲突成因事件（与仪表盘「选反向」同判定：
    `_event_ratio_dir` 1.05/0.95 事件方向 + clean 方向存在且 dir != clean，
    含 flat ≠ clean）。返回 {(memory_id, 事件序号)}——供 --exclude-clashes
    自动剔除重判，选反向在 CLI 侧的一键执行。exclude = 已在排除集内的
    事件不再重复圈定。"""
    exclude = exclude or set()
    keys: set = set()
    for t in MemType:
        h = health.get("by_type", {}).get(t.value) or {}
        clean = (h.get("clean") or {}).get("direction")
        if not clean:
            continue
        for e in _conflict_events(agent, t, exclude=exclude):
            if e["direction"] != "legacy" and e["direction"] != clean:
                keys.add((e["memory_id"], e["index"]))
    return keys


def aggregation_for(agent: "MemoryAgent", health: dict, spec: dict) -> dict:
    """把一次"Shift 多选"聚合结论写成 health.aggregations 条目——复刻仪表盘
    判定（中位比值方向用 _ratio_dir 1.0±0.005、事件方向用 _event_ratio_dir
    1.05/0.95），同一记忆库上导出与仪表盘结论逐字一致，CI 可回放人工排查。
    单一事实源：session_memory 的 --aggregations 导出与仪表盘回放共用。
    spec = {"mtype", "events": ['memory_id:序号', ...]}。

    条目含方向占比分布（与仪表盘色条同口径，以有比值事件为基数）：
    all_dist / all_n / all_dist_pct（全体）与 selected_dist / selected_n_ratio /
    selected_dist_pct（选中集）——外部工具无需再翻事件 CSV 即可复刻色条。"""
    mtype = spec["mtype"]
    h = health.get("by_type", {}).get(mtype) or {}
    clean = (h.get("clean") or {}).get("direction")
    evs = _conflict_events(agent, MemType(mtype))
    key_of = lambda e: f"{e['memory_id']}:{e['index']}"
    sel_set = set(spec.get("events", []))
    sel_evs = [e for e in evs if key_of(e) in sel_set]
    rem = [e for e in evs if key_of(e) not in sel_set]
    med = lambda lst: (statistics.median([e["ratio"] for e in lst
                                          if e["ratio"] is not None])
                       if any(e["ratio"] is not None for e in lst) else None)
    all_med, rem_med = med(evs), med(rem)
    # 方向占比分布（与仪表盘色条同口径：以有比值事件为基数）——外部工具可直接复刻
    def _dist_of(events):
        d = {"up": 0, "down": 0, "flat": 0}
        for e in events:
            if e["ratio"] is not None:
                d[e["direction"]] += 1
        return d

    def _pct_of(d, total):
        if not total:
            return {"up": 0, "down": 0, "flat": 0}
        return {k: round(v / total * 100) for k, v in d.items()}

    all_ratio_evs = [e for e in evs if e["ratio"] is not None]
    dist_all = _dist_of(evs)
    dist = _dist_of(sel_evs)
    n_sel_ratio = sum(dist.values())
    rd = _ratio_dir(rem_med) if rem_med is not None else None
    if rem_med is None:
        verdict, verdict_text = "insufficient", "— 剩余观测不足，无法判定"
    elif clean and rd == clean:
        verdict, verdict_text = "resolved", "✔ 移除后两路一致——冲突消除"
    elif clean and rd and rd != clean:
        verdict, verdict_text = "still_conflict", "✘ 移除后仍冲突"
    else:
        verdict, verdict_text = "unknown", "— 方向未判定"
    return {
        "mtype": mtype,
        "events": sorted(sel_set),
        "selected_n": len(sel_evs),
        "selected_dist": dist,
        "selected_n_ratio": n_sel_ratio,
        "selected_dist_pct": _pct_of(dist, n_sel_ratio),
        "all_dist": dist_all,
        "all_n": len(all_ratio_evs),
        "all_dist_pct": _pct_of(dist_all, len(all_ratio_evs)),
        "all_median_ratio": all_med,
        "all_direction": _ratio_dir(all_med) if all_med is not None else None,
        "remaining_n": len(rem),
        "remaining_median_ratio": rem_med,
        "remaining_direction": rd,
        "clean_direction": clean,
        "verdict": verdict,
        "verdict_text": verdict_text,
    }


def aggregation_recompute(agent: "MemoryAgent", keys: list[str],
                         recent_seconds: float) -> dict:
    """resolved 聚合自动附带排除后重算——与 --exclude-events 同链路：把聚合
    圈出的事件子集（memory_id:序号）作为排除集，重算唤醒统计/漂移/health。
    人工在仪表盘圈定的冲突成因一步变成「剔除后健康证据包」，CI 读 JSON 直接
    拿到排除后的健康结论（consistency / suggest / warnings / actions）。返回
    {"excluded", "stats", "periods", "health"}；无有效排除集返回 {}。"""
    exc: set = set()
    for key in keys:
        mem_id, _, idx = key.rpartition(":")
        try:
            exc.add((mem_id, int(idx)))
        except ValueError:
            continue
    if not exc:
        return {}
    stats = awakening_signal_stats(agent, exclude=exc)
    periods = awakening_signal_periods(agent, recent_seconds=recent_seconds,
                                       exclude=exc)
    health2 = tau_learner_health(agent, exclude=exc)
    return {
        "excluded": sorted(f"{m}:{i}" for m, i in exc),
        "stats": stats,
        "periods": periods,
        "health": health2,
    }


def _conflict_warning(mtype: str, h: dict) -> dict:
    """冲突类型的告警条目：类型 + 两路原始证据 + 建议（供导出 JSON /
    CI 判定红灯——health.warnings 非空即需处理）。"""
    cl, aw = h["clean"], h["awakening"]
    if cl["direction"] is not None:
        clean_ev = (f"干净段 {cl['n']} 条: 实测τ≈{cl['tau_est']:.2f}s vs 配置 "
                    f"{cl['cfg_tau']:.2f}s → {_TAU_DIR_CN[cl['direction']]}")
    else:
        clean_ev = f"干净段 {cl['n']} 条: 方向未判定（观测不足）"
    if aw["direction"] is not None:
        aw_ev = (f"唤醒 {aw['n']} 条: 中位比值 {aw['ratio_med']:.3f} "
                 f"→ {_TAU_DIR_CN[aw['direction']]}")
    else:
        aw_ev = f"唤醒 {aw['n']} 条: 方向未判定（无观测）"
    return {
        "mtype": mtype,
        "consistency": "conflict",
        "clean_evidence": clean_ev,
        "awakening_evidence": aw_ev,
        "suggestion": "两路互相矛盾，学习器按置信度加权折中——建议检查该类型观测"
                       "是否被污染（如检索与衰减混在一起）或事件注入",
    }


def tau_learner_health(agent: "MemoryAgent", exclude: set | None = None) -> dict:
    """τ 学习器健康检查：从记忆库扫描两路信号的**方向一致性**。exclude =
    {(memory_id, 事件序号)} 排除集——排除事件后重算唤醒源与冲突明细（冲突
    剔除假设检验：仪表盘圈出的可疑事件排除后 health 重新判定）。

    - 干净段源（fit_report）：干净衰减段反推的实测 τ vs 配置 τ——实测更小
      → 忘得快 → 方向 down（τ 应下调）；
    - 唤醒源（awakenings 四元组）：中位比值 dev/expected vs 1——> 1 → 埋得比
      类型预期深 → 同样方向 down；
    - 一致性：两路同向 = 互相印证（学习器收敛方向明确）；反向 = 冲突（需检查
      观测污染 / 事件注入）；单源 = 无法交叉印证；无数据 = 无信号。

    门控与学习器一致：干净段 ≥ tau_min_segments 才判方向；旧格式三元组跳过。
    返回 {"by_type": {类型: {"clean": {"n", "tau_est", "cfg_tau", "direction"},
    "awakening": {"n", "ratio_med", "direction"}, "consistency", "suggest",
    "confidence"}},
    "summary": {"agree", "conflict", "one_sided", "no_data"},
    "warnings": [冲突告警 {mtype, consistency, clean_evidence,
    awakening_evidence, suggestion}, ...]（冲突类型才非空，供 CI 按非空判定红灯），
    "actions": [行动项 {mtype, suggest, confidence, reason}, ...]（suggest ∈
    {τ↓, τ↑, 需检查} 的类型——即终端行动清单与 suggest_adjust 列的同一视图，
    供 CI 按非空判定需处理，外部工具无需再跑 CSV 读回）}。
    纯读取不改库。这是两路信号健康检查的单一事实源（导出合表 / 仪表盘画像列 /
    收工验证共用）。confidence（强/中/弱/—）见 _suggest_confidence——agree 按
    观测条数给证据强度，避免单条观测就建议调参。
    """
    fit = agent.fit_report()
    stats = awakening_signal_stats(agent, exclude=exclude)
    out: dict = {"by_type": {}, "summary": {"agree": 0, "conflict": 0,
                                              "one_sided": 0, "no_data": 0}}
    for t in MemType:
        d = fit["by_type"][t.value]
        cfg_tau = agent.cfg.tau_for(t)
        tau_est = d.get("tau_est")
        clean_n = d.get("clean", 0)
        clean_dir = None
        if tau_est is not None and clean_n >= agent.cfg.tau_min_segments:
            clean_dir = _tau_dir(tau_est, cfg_tau)
        aw = stats[t.value]
        n_aw = aw.get("events", 0)
        ratio = aw.get("ratio_med")
        aw_dir = None
        if n_aw and ratio is not None:
            aw_dir = _ratio_dir(ratio)
        if clean_dir is not None and aw_dir is not None:
            consistency = "agree" if clean_dir == aw_dir else "conflict"
        elif clean_dir is not None or aw_dir is not None:
            consistency = "one_sided"
        else:
            consistency = "no_data"
        entry = {
            "clean": {"n": clean_n, "tau_est": tau_est,
                      "cfg_tau": round(cfg_tau, 2), "direction": clean_dir},
            "awakening": {"n": n_aw, "ratio_med": ratio, "direction": aw_dir},
            "consistency": consistency,
        }
        entry["suggest"] = _adjust_suggestion(entry)
        entry["confidence"] = _suggest_confidence(
            entry, agent.cfg.tau_min_segments, agent.cfg.tau_min_awakenings)
        out["summary"][consistency] += 1
        out["by_type"][t.value] = entry
    out["warnings"] = [
        _conflict_warning(t.value, entry)
        for t, entry in ((t, out["by_type"][t.value]) for t in MemType)
        if entry["consistency"] == "conflict"
    ]
    out["actions"] = [
        {"mtype": t.value, "suggest": entry["suggest"],
         "confidence": entry["confidence"],
         "reason": _ADJUST_CN.get(entry["suggest"], "")}
        for t, entry in ((t, out["by_type"][t.value]) for t in MemType)
        if entry["suggest"] in _ADJUST_CN
    ]
    # 冲突成因事件：warnings 与需检查 actions 都挂同一份（含 index 供仪表盘
    # 定位主图唤醒点、row 行号直接进 health——导出/仪表盘/终端零重复）
    conflict_types = [t for t in MemType
                      if out["by_type"][t.value]["consistency"] == "conflict"]
    row_map = _awakening_row_map(agent, exclude) if conflict_types else {}
    for t in conflict_types:
        conflict_evs = _conflict_events(agent, t, row_map, exclude)
        for w in out["warnings"]:
            if w["mtype"] == t.value:
                w["events"] = conflict_evs
        for act in out["actions"]:
            if act["mtype"] == t.value:
                act["events"] = conflict_evs
    return out


class MemoryAgent:
    """分层记忆 Agent。

    sleep() 模拟"睡眠巩固"：把久未访问的低频 Warm 记忆聚类压缩成
    Cold 摘要（海马体索引），并把闲置的 Hot 记忆降级。
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        cfg: AgentConfig | None = None,
        scorer: DecayScorer | None = None,
        persist_path: str | None = None,
        content_updater: Callable[[Memory, str, float], str | None] | None = None,
        content_updaters: dict | None = None,
        classifier=None,
        responder=None,
        now_fn: Callable[[], float] | None = None,
        persona: str | None = None,
    ):
        """content_updater: 可选的再巩固内容编辑钩子 (记忆, 触发查询, 可塑性) -> 新内容或 None。

        内置的再巩固只漂移语义向量与重要性（安全、无损文本）；想真正改写记忆
        文本（如接入 LLM 把回忆情境融进内容）时，提供此钩子即可。
        content_updaters: 按类型分流的钩子注册表 {MemType 或 "skill"/"semantic"/"episodic": fn}，
        类型专属钩子优先，未注册的类型回退通用 content_updater。技能类建议配
        checkers.consistency_checker()——回忆时核对一致性而非情境改写。
        classifier: 类型分类器，默认 LLMClassifier（读 OPENAI_* 环境变量，
        未配置 key 时自动用关键词规则离线分类）。
        responder: 可选的 LLM 回复生成器（memagent.responder.LLMResponder，
        读 OPENAI_* 环境变量）。配置且可用时，respond() 把检索结果注入上下文
        让 LLM 基于记忆回答、无相关记忆时由 LLM 直接回答；未配置或未设 key
        或出错时自动回退内置模板回复。
        now_fn: 可注入的时钟函数（对照实验/模拟时间），默认 time.time。
        persona: 人设（如 "novelist"/"小说家" 映射内置小说家人设，或任意自定义
        文本）。设置后自动创建/配置 LLMResponder，并把 persona_sheet() 的演化
        档案注入每次回复——人设随 remember_setting() 写入的记忆自主演化。
        """
        self.cfg = cfg or AgentConfig()
        self._now = now_fn or time.time
        self.scorer = scorer or DecayScorer(ScorerConfig(tau_seconds=self.cfg.tau_seconds))
        self.store = store if store is not None else MemoryStore(path=persist_path)
        self.content_updater = content_updater
        self.content_updaters = content_updaters or {}
        self.classifier = classifier if classifier is not None else LLMClassifier()
        if persona is not None and responder is None:
            from .responder import LLMResponder

            responder = LLMResponder(persona=persona)
        elif persona is not None and hasattr(responder, "set_persona"):
            responder.set_persona(persona)
        self.persona = persona
        self.responder = responder
        self.last_reconsolidated: list[str] = []
        self.last_scene: Scene | None = None  # 最近一次场景重建结果（respond/CLI 展示）
        self._turns = 0
        # 当前情绪状态：agent 此刻的 (价, 唤醒, 自我相关)，影响检索一致性过滤
        # 由 respond/remember 自动推断，也可手动 set_current_emotion()
        self.current_emotion: Emotion | None = None
        self._learn_history: list = []
        # 兴趣向量（成长方向）。用户 set_growth_direction / 情绪累积 / 检索累积三条路径同时生效
        state = self.store.meta.get("agent_state") or {}
        self.interest = InterestVector.from_dict(state.get("interest"))
        # 知识图谱 + 生长引擎
        self.graph = KnowledgeGraph.from_dict(state.get("graph"))
        self.growth = GrowthEngine.from_dict(state.get("growth"))
        self.growth.set_interest_getter(self.interest.get)
        # 认知高层：技能/目标/自我模型
        self.cognition = Cognition.from_dict(state.get("cognition"))
        # 好奇驱动探索：提问→搜索→写入
        self.curiosity = CuriosityDrivenExplore.from_dict(state.get("curiosity"), agent=self)
        # 类比迁移
        self.analogy = AnalogyTransfer.from_dict(state.get("analogy"), agent=self)
        # 社交学习
        self.social = SocialLearner.from_dict(state.get("social"), agent=self)
        emotion_state = state.get("current_emotion")
        if isinstance(emotion_state, dict):
            try:
                self.current_emotion = Emotion(**emotion_state).clamp()
            except (TypeError, ValueError):
                self.current_emotion = None
        self._plasticity_history: list = []
        # τ↔可塑性联合估计器的跨轮状态：learn_tau 本轮扫描出的 drift 样本
        # （learn_plasticity 直接消费，避免同轮二次扫描污染跨轮耦合）；上一轮
        # 的 τ 参考（drift 通道按衰减公式重算校正用）与可塑性估计（τ 通道剥
        # 可塑性因子用）——两路 EMA 相互加速收敛的耦合点。
        self._joint_drift_samples: dict[str, list] = {}
        self._joint_tau_ref: dict[str, float] = {}
        self._joint_p_ref: dict[str, float] = {}
        # 重启后应用学习器持久化的 τ（学习开启时覆盖显式配置）
        if self.cfg.tau_learning:
            for _t in MemType:
                if _t.value in (self.store.meta.get("learned_tau") or {}):
                    self.cfg.tau_by_type[_t] = float(self.store.meta["learned_tau"][_t.value])
        # 重启后应用学习器持久化的再巩固因子（覆盖对应通道）
        if self.cfg.plasticity_learning:
            for _t in MemType:
                lp = (self.store.meta.get("learned_plasticity") or {}).get(_t.value)
                if lp:
                    merged = dict(self.cfg.reconsolidation_by_type.get(_t, {}))
                    merged.update({k: float(v) for k, v in lp.items()})
                    self.cfg.reconsolidation_by_type[_t] = merged

    # ---------- 写入 ----------

    def classify_text(self, content: str, kind: str = "fact") -> tuple[MemType, float, str]:
        """用当前分类器识别记忆类型，返回 (类型, 置信度, 来源: llm/keyword/turn)。"""
        return self.classifier.classify(content, kind)

    def set_current_emotion(self, emotion: Emotion | None = None) -> None:
        """手动设置 agent 当前情绪状态（用于情绪一致性检索）。None=清空。"""
        self.current_emotion = emotion

    def _persist_agent_state(self) -> None:
        self.store.meta["agent_state"] = {
            "version": 1,
            "interest": self.interest.to_dict(),
            "graph": self.graph.to_dict(),
            "growth": self.growth.to_dict(),
            "cognition": self.cognition.to_dict(),
            "curiosity": self.curiosity.to_dict(),
            "analogy": self.analogy.to_dict(),
            "social": self.social.to_dict(),
            "current_emotion": asdict(self.current_emotion) if self.current_emotion else None,
        }

    def save(self, path: str | None = None) -> None:
        """Persist memories, learned parameters, and all agent growth state."""
        self._persist_agent_state()
        self.store.save(path)

    def _cold_after(self, mem: Memory) -> float:
        """该记忆压缩进 Cold 的闲置阈值：绝对秒数或按类型 τ 推导。"""
        if self.cfg.cold_after_seconds is not None:
            return self.cfg.cold_after_seconds
        return self.cfg.cold_after_tau * self._tau_for(mem)

    def _true_tau_for(self, mem: Memory) -> float:
        """观测用的"真实"遗忘 τ：配置了 true_tau_by_type 时用它（实验模式模拟
        隐藏环境的真实遗忘），否则回落到**模型预测同一口径**（裸类型 τ）。

        回落不能带情绪/兴趣调制（_tau_for）：非实验模式下采样公式本身就是
        "环境"，观测端若比预测端多乘一个因子（neutral 也 ≈×1.10），fit_report
        与唤醒偏差比值都会看到幻影偏差，learn_tau / learn_plasticity 被系统性
        带偏——三处消费方（_record_sample / _observe_awakening / 触底验证）
        的契约都是"未配置时同模型 τ"。
        """
        m = self.cfg.true_tau_by_type
        model_tau = self.cfg.tau_for(mem.mtype)
        if not m:
            return model_tau
        return m.get(mem.mtype, model_tau)

    def set_growth_direction(self, topic: str, intensity: float = 0.8, keywords: list[str] | None = None) -> None:
        """显式指定成长方向。
        主题名+兴趣强度+关键词集合。后续该主题的记忆会被兴趣因子加强。
        """
        self.interest.set(topic, intensity)
        if keywords:
            self.interest.register_topic(topic, keywords)

    def register_topic(self, topic: str, keywords: list[str]) -> None:
        """注册主题关键词（不改变兴趣值）。"""
        self.interest.register_topic(topic, keywords)

    def growth_directions(self) -> list[tuple[str, float]]:
        """当前兴趣排名。"""
        return self.interest.top()

    def remember(
        self,
        content: str,
        importance: float | None = None,
        kind: str = "fact",
        mtype: MemType | None = None,
        emotion: object = _UNSET,
    ) -> Memory:
        """写入 Warm 层。返回写入的 Memory。带去重+情绪调制+兴趣调制。

        情绪参数：不传→自动推断；None→无调制；Emotion→显式标注
        兴趣向量三条路径：显式指定 / 情绪驱动 / 检索累积
        """
        if kind == "turn" and importance is None:
            importance = 0.05
        explicit_emotion = emotion is not _UNSET  # 显式传参才允许覆盖旧记忆的情绪标注
        if emotion is _UNSET:
            emotion = infer_emotion(content)
        if emotion is not None:
            self.current_emotion = emotion.clamp()

        # 兴趣路径2（情绪驱动）：所属主题兴趣值随 arousal 递增
        detected_topics = self.interest.detect_topics(content)
        if detected_topics and emotion is not None:
            delta = self.interest.emotion_delta_per_unit * emotion.arousal
            for t in detected_topics:
                self.interest.update(t, delta)

        # 兴趣编码：主题内记忆 importance 按兴趣值加权
        if importance is None:
            importance = 0.1
        importance *= encoding_factor(emotion)
        if detected_topics:
            topic_boost = 1.0 + self.interest.get(detected_topics[0]) * 0.5
            importance *= min(topic_boost, 1.5)
        # 编码路径放大后钳制到 1.0：防止 skill/semantic 经情绪×兴趣联合调制后超 [0,1]
        # （实测 32 条超上限）。衰减公式中 importance 作为权重分量，>1 会扭曲 strength 比例。
        importance = min(importance, 1.0)

        mtype_conf: float | None = None
        if mtype is None:
            mtype, mtype_conf, _ = self.classifier.classify(content, kind)
        cv = embed_text(content)  # 查询向量只算一次（循环内重复嵌入是 O(N) 写入开销）
        cand = [m for m in self.store.all() if m.tier is not Tier.COLD]
        for m in cand:
            if cosine_similarity(cv, embed_text(m.content)) >= DEDUP_THRESHOLD:
                m.touch(self._now())
                self._record_sample(m)
                if importance > m.importance:
                    m.importance = importance
                if explicit_emotion:
                    m.emotion = emotion
                elif m.emotion is None:
                    m.emotion = emotion  # 自动推断只补空，不抹掉已有标注（如恐惧）
                return m
        mem = self.store.add(
            content, importance=importance, kind=kind,
            mtype=mtype, mtype_confidence=mtype_conf, now=self._now(),
        )
        mem.emotion = emotion
        self._record_sample(mem)
        return mem

    def remember_setting(self, content: str, importance: float = 0.6) -> Memory:
        """写入一条"设定"记忆（kind="setting"）——人设演化的原料。

        小说家场景下：作品名、世界观、人物关系、境界体系、时间线、伏笔、
        章节进度等长期设定都该用本方法入库；persona_sheet() 会把这些设定
        按重要性排序注入每次回复的 system prompt，让创作人设随设定累积
        自主演化、跨会话保持一致。
        """
        return self.remember(content, kind="setting", importance=importance)

    def remember_skill(self, content: str, importance: float = 0.7) -> Memory:
        """写入一条"技能/方法论"记忆（kind="skill"）——随创作经验沉淀的规则。

        自评模块 critique.py 会把每次写作的改进点以本方法入库；技能记忆
        遗忘最慢（MemType.SKILL τ 最长），因此沉淀下来的写作规则会长期生效，
        成为 agent 持续变强的载体。

        写入时强制 access_count=2，避免被压缩逻辑当作"从未使用过的新记忆"
        卷入冷压缩；同时靠 importance>=0.8 双保险进入 candidates 排除门限。
        """
        m = self.remember(content, kind="skill", importance=importance)
        m.access_count = max(m.access_count, 2)  # 保护：标记为"已使用"
        return m

    def persona_sheet(self, limit: int = 8) -> str | None:
        """当前人设演化档案：按重要性取前 limit 条 kind="setting" 记忆。

        返回多行文本（每条一行），供注入 LLM 回复的 system prompt；
        没有任何设定记忆时返回 None（不注入，保持原行为）。
        """
        settings = [m for m in self.store.all() if m.kind == "setting"]
        if not settings:
            return None
        # “已连载 N 章”进度条是作品名唯一权威载体，排最前（与写章守卫同口径）：
        # 长期运行后多条设定可能都饱和在 importance=1.0，纯重要性排序会因并列
        # 不稳定而把进度条挤出前 limit，导致 _work_title() 误判书名。
        settings.sort(key=lambda m: ("已连载" not in m.content, -m.importance))
        lines = [f"• {m.content}" for m in settings[:limit]]
        return "\n".join(lines)

    def _research_query(self) -> str:
        """演化时的联网研究主题：优先取人设档案里的作品名，否则用类型泛词。"""
        sheet = self.persona_sheet() or ""
        import re as _re

        m = _re.search(r"《([^》]+)》", sheet)
        if m:
            return f"《{m.group(1)}》 小说 世界观 设定 灵感"
        return "玄幻仙侠小说 修炼体系 境界设定 创作灵感"

    def _absorb_settings(self, reply: str, max_settings: int) -> list[str]:
        """从 LLM 演化回复中提取设定行并入库（去重、长度门控、条数上限）。"""
        added: list[str] = []
        for raw in reply.splitlines():
            line = raw.strip().lstrip("-•*0123456789.、）)] ")
            for prefix in ("设定：", "设定:", "新设定：", "新设定:"):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            else:
                continue  # 只吸收以「设定：」开头的行（演化指令已要求该格式）
            if len(line) < 6:
                continue
            if any(
                cosine_similarity(embed_text(line), embed_text(m.content)) >= DEDUP_THRESHOLD
                for m in self.store.all()
                if m.kind == "setting"
            ):
                continue
            self.remember_setting(line, importance=0.7)
            added.append(line)
            if len(added) >= max_settings:
                break
        return added

    def evolve(self, max_settings: int | None = None, with_web: bool | None = None) -> dict:
        """自主演化：反思记忆 + 联网查资料 → LLM 提出新设定 → 入库。

        返回演化报告 {ok, reason?, query, web_n, added, skipped}。
        未配 LLM（无 key / responder 不可用）时 ok=False 静默跳过——
        演化是可选的成长能力，不影响检索/回复等主链路。
        """
        if self.responder is None or not self.responder.available:
            return {"ok": False, "reason": "no-llm", "query": "", "web_n": 0,
                    "added": [], "skipped": []}
        max_settings = self.cfg.evolve_max_settings if max_settings is None else max_settings
        with_web = self.cfg.evolve_search if with_web is None else with_web
        sheet = self.persona_sheet() or "（尚未有任何设定）"
        mems = sorted(
            (m for m in self.store.all() if m.kind != "turn"),
            key=lambda m: m.importance, reverse=True,
        )[:10]
        context = "\n".join(f"- {m.content}" for m in mems) or "（记忆库为空）"
        query = self._research_query()
        web_block, web_n = "（未联网）", 0
        if with_web:
            try:
                from .websearch import search_web

                results = search_web(query, n=4)
                web_n = len(results)
                web_block = "\n".join(
                    f"- {r['title']}（{r['url']}）: {r['snippet'][:120]}"
                    for r in results
                ) or "（无搜索结果）"
            except Exception:
                web_block = "（联网搜索不可用）"
        instruction = (
            f"【自主演化任务】你是正在自主成长的作家。基于以下你的记忆、当前人设档案"
            f"与搜索到的创作资料，提出 {max_settings} 条新的作品设定或发展（新人物/"
            f"新伏笔/世界观补全/剧情走向/修炼体系细节/人物关系变化等）。要求：与已有"
            f"设定自洽、不重复已有内容、每条具体可执行。只输出设定本身，每条一行，"
            f"以「设定：」开头。\n\n"
            f"—— 当前人设档案 ——\n{sheet}\n\n"
            f"—— 最近记忆 ——\n{context}\n\n"
            f"—— 联网资料（查询：{query}）——\n{web_block}"
        )
        try:
            reply = call_responder(
                self.responder, instruction, memories=None,
                timeout=self.cfg.llm_long_timeout,
            )
        except Exception as e:
            return {"ok": False, "reason": f"llm-error:{e}", "query": query,
                    "web_n": web_n, "added": [], "skipped": []}
        added = self._absorb_settings(reply, max_settings)
        return {"ok": True, "query": query, "web_n": web_n,
                "added": added, "skipped": max(0, max_settings - len(added))}

    # ---------- 每日创作产出 ----------

    def _work_title(self) -> str:
        """从人设档案提取作品名（《…》），没有则用默认名。"""
        sheet = self.persona_sheet() or ""
        import re as _re

        m = _re.search(r"《([^》]+)》", sheet)
        return m.group(1).strip() if m else "未命名作品"

    def _work_dir(self, title: str) -> str:
        """作品落盘目录 works/<书名>/chapters/（自动创建）。"""
        d = Path(self.cfg.chapter_save_dir) / _safe_title(title) / "chapters"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def _next_chapter(self, title: str) -> int:
        """下一章编号：扫描落盘目录里已有的 第N章.md 文件，取最大 +1。"""
        d = Path(self.cfg.chapter_save_dir) / _safe_title(title) / "chapters"
        nums: list[int] = []
        if d.is_dir():
            for f in d.glob("第*章.md"):
                m = re.search(r"第(\d+)章", f.name)
                if m:
                    nums.append(int(m.group(1)))
        return (max(nums) + 1) if nums else 1

    def _last_chapter_tail(self, title: str, chars: int = 200) -> str:
        """上一章结尾片段（保持章节连续性）：取最新一章文件的尾部。"""
        no = self._next_chapter(title) - 1
        if no < 1:
            return "（这是第一章）"
        d = Path(self.cfg.chapter_save_dir) / _safe_title(title) / "chapters"
        f = d / f"第{no}章.md"
        if not f.is_file():
            return "（这是第一章）"
        return f.read_text(encoding="utf-8")[-chars:].strip()

    def _update_chapter_progress(self, title: str, no: int) -> None:
        """把“已连载 N 章”写进设定记忆（同作品去重替换，保持档案干净）。"""
        tag = f"《{title}》：已连载 {no} 章"
        for m in self.store.all():
            if m.kind == "setting" and "已连载" in m.content and title in m.content:
                m.content = tag
                m.embedding = embed_text(tag)  # 内容变了必须同步检索向量
                return
        self.remember_setting(tag, importance=0.95)

    def call_with_retry(self, prompt: str, *, min_len: int = 1,
                        timeout: float | None = None,
                        attempts: int | None = None,
                        retry_delay: float | None = None,
                        max_tokens: int | None = None,
                        **kw) -> str:
        """带空回复/异常重试的 LLM 调用（写章管线专用，所有作品共用）。

        responder.respond() 对 reasoning-only 空回复会抛异常而不是返回空串，
        直接调用会导致整轮失败（日志里的"LLM 回复为空"）。本封装：空/短回复或
        异常时最多重试 attempts 次（默认 cfg.llm_retries），成功才返回，全失败
        抛最后一个错误。

        min_len 区分调用类型（短回复门槛）：
        - 标题/剧情目标/架构更新等短回复必须用 min_len=1——否则 6-12 字的标题会被
          误判为空重试到失败（第 54 章空标题的根因）；
        - 正文也用 min_len=1：调用级重试只兜"抛异常"（reasoning-only 空回复），
          长度不足由 _write_chapter_locked 既有检查分类，写章级重试整轮重跑。

        kw 原样透传给 call_responder（memories/persona_extras 等）。
        """
        attempts = self.cfg.llm_retries if attempts is None else attempts
        retry_delay = self.cfg.llm_retry_delay if retry_delay is None else retry_delay
        last_err: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                reply = call_responder(
                    self.responder, prompt, **kw, timeout=timeout,
                    max_tokens=max_tokens,
                )
                if reply and len(reply.strip()) >= min_len:
                    return reply
                last_err = RuntimeError(
                    "LLM 回复为空（可能只有 reasoning，没有最终 content）"
                )
            except Exception as e:
                last_err = e
            if attempt < attempts:
                print(
                    f"  LLM 回复为空/异常（第 {attempt}/{attempts} 次），"
                    f"{retry_delay:g}s 后重试：{str(last_err)[:60]}",
                    flush=True,
                )
                time.sleep(retry_delay)
        raise last_err or RuntimeError("LLM 回复为空")

    def ensure_title_in_sheet(self) -> str | None:
        """写章守卫：确保作品名记忆始终在人设档案（persona_sheet 前 8）内。

        伏笔/新设定会以高重要性不断写入，可能把作品名记忆挤出前 8，导致
        `_work_title()` 误判为“未命名作品”、把章节写进错误目录。本守卫每次
        写章前/后调用：若作品名记忆不在前 8，自动把其重要性提到第 8 名之上
        （+0.2 余量），保证 `_work_title()` 永远拿到正确书名。

        返回：守卫生效后的作品名；没有任何《…》设定记忆时返回 None，
        调用方走原有“未命名作品”兜底。
        """
        import re as _re

        settings = [m for m in self.store.all() if m.kind == "setting"]
        if not settings:
            return None

        def _title_of(m: Memory) -> str | None:
            mm = _re.search(r"《([^》]+)》", m.content)
            if not mm:
                return None
            t = mm.group(1).strip()
            return None if t == "未命名作品" else t

        cands = [(m, t) for m in settings if (t := _title_of(m))]
        if not cands:
            return None
        # 优先“已连载 N 章”进度条（作品名唯一权威载体），其次按重要性
        cands.sort(key=lambda x: ("已连载" not in x[0].content, -x[0].importance))
        mem, title = cands[0]

        # 与 persona_sheet 同口径：进度条优先，其次重要性（并列时排序才稳定）
        top = sorted(settings, key=lambda m: ("已连载" not in m.content, -m.importance))[:8]
        if mem in top:
            return title
        eighth = min(m.importance for m in top)
        # 钳制到 [0,1]：importance >1 会扭曲强度比例，且 ≥freeze_importance 后记忆永久冻结
        new_imp = min(1.0, max(eighth + 0.2, mem.importance * 1.5))
        print(
            f"[写章守卫] 作品名《{title}》被挤出人设档案前 8（importance "
            f"{mem.importance:.3f} → {new_imp:.3f}），已自动提升",
            flush=True,
        )
        mem.importance = new_imp
        self.save()
        return title

    def write_chapter(
        self,
        target_words: int | None = None,
        with_web: bool | None = None,
        save_dir: str | None = None,
    ) -> dict:
        """Run one serialized chapter-writing transaction per work."""
        if self.responder is None or not self.responder.available:
            return {"ok": False, "reason": "no-llm"}
        self.ensure_title_in_sheet()  # 写前守卫：书名被挤出档案前 8 时自动提升
        title = self._work_title()
        if save_dir:
            supplied = Path(save_dir)
            work_root = supplied.parent if supplied.name == "chapters" else supplied
        else:
            work_root = Path(self.cfg.chapter_save_dir) / _safe_title(title)
        try:
            with FileLock(work_root / ".writer.lock", timeout=0.5):
                # 写章级重试：空/截断/未收尾时整轮重跑（章号从磁盘推断，失败不占号）
                result: dict | None = None
                retries = self.cfg.chapter_retries
                for attempt in range(1, retries + 1):
                    result = self._write_chapter_locked(
                        target_words, with_web, save_dir,
                    )
                    if result.get("ok") or result.get("reason") not in (
                        "llm-incomplete", "llm-empty", "llm-truncated",
                    ):
                        break
                    print(
                        f"  写章未完成（{result.get('reason')}，第 {attempt}/{retries} 次），"
                        f"{self.cfg.chapter_retry_delay:g}s 后重试…",
                        flush=True,
                    )
                    time.sleep(self.cfg.chapter_retry_delay)
        except LockTimeoutError:
            return {"ok": False, "reason": "writer-busy", "title": title}
        if result and result.get("ok"):
            self.ensure_title_in_sheet()  # 写完复检：架构更新等新增记忆若挤出书名，自动提升
        return result or {"ok": False, "reason": "no-result", "title": title}

    def _write_chapter_locked(
        self,
        target_words: int | None = None,
        with_web: bool | None = None,
        save_dir: str | None = None,
    ) -> dict:
        """基于当前人设档案续写下一章正文，落盘 works/<书名>/chapters/，
        并把章节进度回写进设定记忆。返回 {ok, title, chapter, path, words, preview}。

        连续性三要素：
        - 人设档案（persona_sheet）注入 system prompt——设定/人物/伏笔全程生效；
        - 上一章结尾片段注入——剧情无缝衔接；
        - 章节号从落盘文件推断——重启/多进程也单调递增，进度不靠 LLM 记忆。
        未配 LLM 时静默跳过（ok=False），不影响其他链路。
        """
        if self.responder is None or not self.responder.available:
            return {"ok": False, "reason": "no-llm"}
        target_words = self.cfg.chapter_words if target_words is None else target_words
        with_web = self.cfg.chapter_with_web if with_web is None else with_web
        self.ensure_title_in_sheet()  # 写前守卫（_write_chapter_locked 也可能被直接调用）
        title = self._work_title()
        chapter = self._next_chapter(title)
        tail = self._last_chapter_tail(title)
        sheet = self.persona_sheet() or "（还没有设定——先用 /evolve 或 remember_setting 建立设定）"

        web_block = "（未联网）"
        if with_web:
            try:
                from .websearch import search_web

                results = search_web(f"{title} 小说 写作 灵感", n=3)
                web_block = "\n".join(
                    f"- {r['title']}（{r['url']}）: {r['snippet'][:100]}"
                    for r in results
                ) or "（无搜索结果）"
            except Exception:
                web_block = "（联网搜索不可用）"

        # 已沉淀的写作改进规则（来自 critique.py 的自评沉淀）——
        # 这些是 agent 从过去写作里学到的具体规则，写作时必须遵守。
        from .critique import writing_improvements as _wi
        skill_block = _wi(self)

        # 剧情目标：从架构文档生成，让本章不再只是"接上一章"而是"推进某个明确目标"
        from .architecture import next_chapter_goal as _ncg
        work_dir = Path(save_dir or self._work_dir(title))
        goal = _ncg(self, title, chapter, work_dir)
        goal_block = goal if goal else "（无明确目标——自由推进剧情）"

        prompt = (
            f"【写作任务】续写你的小说《{title}》的下一章（第 {chapter} 章）。"
            f"目标约 {target_words} 字。\n\n"
            f"—— 当前设定档案（必须严格遵守，不推翻已确认事实）——\n{sheet}\n\n"
            f"—— 上一章结尾（从此处继续，无缝衔接）——\n{tail}\n\n"
            f"—— 已沉淀的写作改进规则（你从过去写作中学到的，务必遵守）——\n{skill_block}\n\n"
            f"—— 读者友好度硬规则（以下 4 条必须遵守，任何新术语/新角色/旧设定首次出现都要照做）——\n"
            f"① 新术语首次出现必须用破折号或括号加 5-15 字内嵌解释，格式「塔纹——他掌心自幼就有的九道银色裂痕」，不可裸用名词。\n"
            f"② 跨章引用的旧设定必须给 5-15 字轻量提示，格式「锈脉——上次断魂崖那根贯穿他血脉的暗管」，不可假设读者记得。\n"
            f"③ 新角色第一次出场必须给 3 个可识别标签（外貌+身份+与主角的关系）。\n"
            f"④ 对话中禁止堆砌术语；每次使用术语后必须夹带人话解释或角色反应。\n\n"
            f"本章剧情目标（务必推动其达成）——\n{goal_block}\n\n"
            f"—— 联网参考（可选灵感，不照抄）——\n{web_block}\n\n"
            "要求：保持设定连续、推进剧情与人物成长、伏笔照应；"
            "成年角色亲密关系可自然展开（双方自愿、均成年）。"
            "只输出正文本身，不要任何解释、标题或前后缀。"
        )
        try:
            # min_len=1：调用级重试只兜"抛异常"（reasoning-only 空回复）；
            # 长度不足/未收尾交给下方既有检查分类（llm-empty/truncated/incomplete），
            # 再由 write_chapter 的写章级重试整轮重跑——保持既有的失败原因契约。
            text = self.call_with_retry(
                prompt, min_len=1, timeout=self.cfg.llm_long_timeout,
                persona_extras=sheet,
                max_tokens=self.cfg.llm_long_max_tokens,
            )
        except Exception as e:
            return {"ok": False, "reason": f"llm-error:{e}"}
        text = text.strip()
        minimum = max(50, int(target_words * self.cfg.chapter_min_ratio))
        if len(text) < minimum:
            reason = "llm-empty" if len(text) < 50 else "llm-truncated"
            return {
                "ok": False,
                "reason": reason,
                "text": text[:100],
                "actual_words": len(text),
                "minimum_words": minimum,
            }
        if not re.search(r'[。！？!?….](?:["”’」』）】])?$', text):
            return {
                "ok": False,
                "reason": "llm-incomplete",
                "text": text[-100:],
                "actual_words": len(text),
                "minimum_words": minimum,
            }
        # 生成章节标题：写完正文后追问 LLM 一句
        chap_title = ""
        try:
            title_reply = self.call_with_retry(
                f"你是一名小说编辑。请给下面这段章节正文拟一个 6-12 字的章节标题"
                f"（古风仙侠味，含蓄不直白，一个画面感强的短句或成语化短语）。\n\n"
                f"{text[:1500]}\n\n只输出标题本身，不要引号、不要解释。",
                min_len=1, timeout=30.0,
            )
            tt = title_reply.strip().strip("《》「」\"")
            if 4 <= len(tt) <= 14:
                chap_title = tt
        except Exception:
            pass

        d = save_dir or self._work_dir(title)
        # 读者友好度 post-process：术语表从作品目录 term_explanations.json 加载
        # （出厂婴儿原则——包内零词表；作品未配置 = 功能静默关闭）
        try:
            from .reader_postproc import inject_explanations as _ri
            from .reader_postproc import load_terms_for_work as _load_terms

            _terms = _load_terms(d)
            if _terms:
                text, _inj = _ri(text, _terms)
        except Exception:
            pass  # 词表缺失/损坏不阻断写章主流程
        f = Path(d) / f"第{chapter}章.md"
        header = f"# 《{title}》第{chapter}章"
        if chap_title:
            header += f" {chap_title}"
        header += "\n\n"
        try:
            atomic_write_text(f, header + text + "\n", overwrite=False)
        except FileExistsError:
            return {"ok": False, "reason": "chapter-conflict", "path": str(f)}
        self._update_chapter_progress(title, chapter)
        return {
            "ok": True, "title": title, "chapter": chapter,
            "chapter_title": chap_title,
            "path": str(f), "words": len(text),
            "preview": text[:60],
        }

    # ---------- 检索 ----------

    def _tau_for(self, mem: Memory) -> float:
        """该记忆的遗忘时间常数（类型 + 情绪 + 兴趣三重调制）。"""
        base_tau = self.cfg.tau_for(mem.mtype)
        tau = base_tau * tau_factor(mem.emotion)
        for t in self.interest.detect_topics(mem.content):
            tau *= 1.0 + self.interest.get(t)
        return tau

    def strength_at_state(
        self,
        mtype: MemType,
        last_access: float,
        access_count: int,
        importance: float,
        t: float,
        tau_override: float | None = None,
    ) -> float:
        """任意状态下 t 时刻的遗忘曲线强度（预测用，可覆盖 τ）。"""
        tau = tau_override or self.cfg.tau_for(mtype)
        s = self.scorer.score(
            last_access=last_access,
            access_count=access_count,
            importance=importance,
            now=t,
            tau_seconds=tau,
        )
        denom = (
            self.scorer.cfg.w_recency
            + self.scorer.cfg.w_freq
            + self.scorer.cfg.w_importance
        )
        return min(1.0, max(STRENGTH_FLOOR, s.total / denom))

    def _strength_at(self, mem: Memory, t: float) -> float:
        """t 时刻的模型预测强度，τ 按记忆类型取：技能慢衰减、情景快衰减。"""
        return self.strength_at_state(
            mem.mtype, mem.last_access, mem.access_count, mem.importance, t
        )

    def _strength(self, mem: Memory) -> float:
        """当前时刻的记忆强度。"""
        return self._strength_at(mem, self._now())

    def _record_sample(self, mem: Memory) -> None:
        """记录一条观测采样（状态快照，用于曲线与贴合度验证）。

        格式 [时间戳, 观测强度, 最后访问, 检索次数, 重要性]：强度按"真实"τ
        采样（有 true_tau_by_type 时），后三列供回放预测与统计干扰。
        """
        now = self._now()
        s = self.strength_at_state(
            mem.mtype, mem.last_access, mem.access_count, mem.importance, now,
            tau_override=self._true_tau_for(mem),
        )
        mem.history.append([now, round(s, 4), mem.last_access, mem.access_count, mem.importance])

    def _semanticization_score(self, mem: Memory) -> float:
        """语义化评分：近期检索事件的指数衰减加权和（无需额外状态）。

        从观测历史推导：相邻快照间 access_count 增大即发生过一次"使用"
        （检索命中/去重强化/升级等），按距离现在的时间指数衰减加权——
        越近的检索贡献越大，评分高 = 该记忆近期被反复使用。
        """
        now = self._now()
        tau = self.cfg.semanticization_tau_seconds
        score = 0.0
        prev: int | None = None
        for row in mem.history:
            ts, _s, _la, acc, _imp = row
            if prev is not None and acc > prev:
                score += math.exp(-(now - ts) / tau)
            prev = acc
        return score

    def _observe(self) -> int:
        """持续观测：给**所有**记忆记录当前强度采样（不只是本轮检索命中的），

        这样每条记忆的真实遗忘轨迹都会被跟踪。返回本轮新采样条数。
        """
        now = self._now()
        n = 0
        for mem in self.store.all():
            if mem.history and abs(mem.history[-1][0] - now) < 0.05:
                continue  # 本轮已采样过（如刚被检索命中）
            self._record_sample(mem)
            n += 1
        return n

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        """检索：对 Warm/Cold 全量做 相似度 × 记忆强度 排序。

        查询在打分前统一归一化（strip + 小写）——与 n-gram 嵌入的归一化语义
        对齐，所有文本级检查（子串重排、再巩固钩子）与 rel 计算全链路大小写
        无关；用户原句保留在调用方（respond/对话注入），此处只做检索内部归一化。

        Cold 记忆用摘要向量参与检索——命中即"索引触发"，此时
        via_summary=True，表示需要时可唤醒底层细节。

        短查询（少于 AgentConfig.rerank_short_len 字，默认 3）自动做**子串优先
        重排**（见 synonyms.substring_priority_order，可用 AgentConfig 开关/调阈值）：
        内容/摘要含查询词的记忆排最前，消除哈希嵌入泛化命中把无关记忆顶到
        前面的干扰——所有下游入口（respond/对话注入/主题检索）统一受益。
        """
        k = k or self.cfg.k
        # 查询归一化（打分前单点）：strip + 小写。n-gram 嵌入内部已小写
        # （ngrams 里 text.lower()），此处让所有**文本级检查**（子串重排、再巩固
        # 内容钩子收到的查询）与 rel 计算的语义完全一致——长英文查询「Python」查
        # 「python」全链路大小写无关。
        query = query.strip().lower()
        qv = embed_text(query)
        # 查询同义扩展：生成变体（人称互换/同义词替换），对每条记忆取最大相似度。
        # 原始查询恒在变体里，故扩展只会提高真实相关记忆的 rel，不会变差。
        if self.cfg.query_expansion:
            qvs = [embed_text(q) for q in expand_query(query)]
        else:
            qvs = [qv]
        self.last_reconsolidated = []
        hits: list[Retrieved] = []
        for mem in self.store.all():
            if mem.tier is Tier.HOT:
                boost = QUERY_BOOST_HOT
            else:
                boost = 1.0
            rel = max(cosine_similarity(q, mem.embedding) for q in qvs) * boost
            if mem.kind == "turn":
                rel *= TURN_PENALTY
            strength = self._strength(mem)
            # 情绪一致性调制：当前情绪与记忆情绪相似 → 提升检索分数
            congruence = congruence_factor(self.current_emotion, mem.emotion)
            hits.append(
                Retrieved(
                    memory=mem,
                    relevance=rel,
                    strength=strength,
                    total=rel * strength * congruence,
                    via_summary=(mem.tier is Tier.COLD),
                )
            )
        hits.sort(key=lambda h: h.total, reverse=True)
        # 短查询（< rerank_short_len 字）子串优先重排：哈希嵌入的泛化命中常把不含
        # 查询词的记忆顶到前面（短词 rel 被碰撞主导），子串优先让内容真正含查询词
        # 的记忆排最前（组内按 rel×强度=total 降序——见 score_of，兼顾 rel 信号）。
        # 与 remember_agent / session_memory 共用 synonyms.substring_priority_order
        # 单点实现——排序发生在测试效应之前，命中强化也跟随子串优先；开关/阈值见
        # AgentConfig（rerank_short_query / rerank_short_len），长查询不重排，
        # 行为与旧版完全一致。
        if self.cfg.rerank_short_query:
            hits, _ = substring_priority_order(
                hits, query,
                content_of=lambda h: h.memory.content + (h.memory.summary or ""),
                strength_of=lambda h: h.strength,
                # 组内按 rel×强度（total）而非纯强度排序：低相关但高强度的
                # 含词记忆不再压过更高相关（但强度略低）的含词记忆——重排
                # 保留子串优先的组边界，组内排序尊重 rel 信号。
                score_of=lambda h: h.total,
                short_len=self.cfg.rerank_short_len,
            )
        # 记录检索命中：测试效应（仅对真正相关的）
        # 兴趣路径3（检索累积）：命中主题的兴趣值随检索分数递增
        for h in hits[: max(k * 2, 1)]:
            if h.total > 0.05:
                content_topics = self.interest.detect_topics(h.memory.content)
                for t in content_topics:
                    delta = self.interest.access_delta * h.total
                    self.interest.update(t, delta)
                h.memory.touch(self._now())
                self._record_sample(h.memory)
                if h.memory.tier is Tier.WARM and h.memory.access_count >= self.cfg.hot_after_access:
                    self._promote_hot(h.memory)
                self._reconsolidate(h.memory, query, qv, h.relevance)
        return hits[:k]

    def _reconsolidate(
        self,
        mem: Memory,
        query_text: str,
        query_vec: list[float],
        relevance: float,
    ) -> bool:
        """回忆后的再巩固：按重要程度微调原始记忆，返回是否发生微调。

        认知对应：回忆使记忆进入可塑状态，并以修改后的形式重新存储。
        - 可塑性 lability = 1 − importance：重要性越高越稳定；
        - importance ≥ freeze_importance 的记忆完全冻结（核心记忆）；
        - 语义漂移：记忆向量向本次回忆情境靠拢（低重要性更明显）；
        - 重要性微调：被高度相关的查询命中会巩固（+），弱相关轻微去巩固（−）；
        - 可塑窗口（labile_until）内漂移幅度放大，模拟再巩固窗口。
        """
        if not self.cfg.reconsolidate:
            return False
        if mem.importance >= self.cfg.freeze_importance:
            return False  # 核心记忆：完全稳定
        now = self._now()

        # 0) 校验型内容钩子（技能类一致性校验）：回忆只核对不改写——
        #    跳过向量漂移与重要性微调，内容/向量全不动，结论记入 mem.checks。
        #    程序性记忆抵抗修改：技能回忆是验证，不是吸收情境。
        #    例外：钩子判定冲突且提供修正时返回新内容——此时才真正改写。
        hook = self._content_hook_for(mem)
        if hook is not None and getattr(hook, "is_checker", False):
            new_content = hook(mem, query_text, 1.0 - mem.importance)
            if new_content and new_content != mem.content:
                mem.content = new_content
                mem.embedding = normalize(embed_text(new_content))  # 内容为准，重建向量
                mem.revision_count += 1
                mem.revisions.append([
                    round(now, 1), round(1.0 - mem.importance, 2), 0.0, 0.0,
                    mem.mtype.value,
                    round(self.cfg.applied_factor(mem.mtype, "drift"), 4),
                    round(self.cfg.applied_factor(mem.mtype, "importance"), 4),
                ])
                del mem.revisions[:-12]
                self.last_reconsolidated.append(mem.id)
                return True
            return False

        # 先判断是否本就在既有窗口内，再开新窗口——否则判断恒真，窗口机制失效
        in_window = now <= mem.labile_until
        if not in_window:
            mem.labile_until = now + self.cfg.reconsolidation_window
        lability = 1.0 - mem.importance
        boost = 1.0 + (self.cfg.labile_bonus if in_window else 0.0)
        changed = False

        # 1) 语义微调：向量向回忆情境漂移（幅度按类型因子缩放：技能稳、情景易改写）
        #    实际因子 = applied_factor（有 true_reconsolidation_by_type 时用它）
        f_drift = self.cfg.applied_factor(mem.mtype, "drift")
        drift = lability * self.cfg.content_drift * boost * f_drift
        if drift > 0:
            mem.embedding = normalize(
                [e + drift * q for e, q in zip(mem.embedding, query_vec)]
            )
            changed = True

        # 2) 重要性微调（强度公式含 importance，故同时微调了强度；同样按类型缩放）
        f_imp = self.cfg.applied_factor(mem.mtype, "importance")
        imp_delta = (
            lability * self.cfg.importance_drift * boost * (2.0 * relevance - 1.0)
            * f_imp
        )
        if imp_delta != 0:
            new_imp = min(1.0, max(self.cfg.importance_floor, mem.importance + imp_delta))
            if new_imp != mem.importance:
                mem.importance = new_imp
                changed = True

        # 3) 可选的内容级编辑钩子（按类型分流：技能=一致性校验，情景=情境改写）
        if hook is not None and drift > 0:
            new_content = hook(mem, query_text, lability)
            if new_content and new_content != mem.content:
                mem.content = new_content
                mem.embedding = normalize(embed_text(new_content))  # 内容为准，重建向量
                changed = True

        if changed:
            mem.revision_count += 1
            # 修订日志：前 4 列同旧格式，后 3 列为事件发生时类型 + 实际应用的
            # drift/importance 因子（实测可塑性，供 learn_plasticity 估计）
            mem.revisions.append([
                round(now, 1), round(lability, 2), round(drift, 3), round(imp_delta, 4),
                mem.mtype.value, round(f_drift, 4), round(f_imp, 4),
            ])
            del mem.revisions[:-12]  # 只保留最近 12 条修订日志
            self.last_reconsolidated.append(mem.id)
        return changed

    def _content_hook_for(self, mem: Memory) -> Callable | None:
        """按类型解析内容钩子：类型专属优先（content_updaters），否则回退通用钩子。"""
        if self.content_updaters:
            h = self.content_updaters.get(mem.mtype)
            if h is None:
                h = self.content_updaters.get(mem.mtype.value)
            if h is not None:
                return h
        return self.content_updater

    def _promote_hot(self, mem: Memory) -> None:
        """升级到 Hot 层；超容量时把最旧的一条踢回 Warm。"""
        hot = self.store.by_tier(Tier.HOT)
        if len(hot) >= self.cfg.max_hot:
            oldest = min(hot, key=lambda m: m.last_access)
            oldest.tier = Tier.WARM
        mem.tier = Tier.HOT
        mem.touch(self._now())
        self._record_sample(mem)

    # ---------- 回复合成（模板；可替换为 LLM 回复生成器） ----------

    def _template_reply(self, relevant: list[Retrieved], scene: Scene | None = None) -> str:
        """内置模板回复：无相关记忆时诚实说不了解，有则引用记忆供确认。

        scene 存在时展示重建出的连贯场景（片段组合）而非单条命中——
        回忆以场景为单位呈现，Cold 片段标注可 /recall 唤醒细节。"""
        parts: list[str] = []
        if scene is not None:
            parts.append(f"我记得一段连贯的场景：{scene.title}")
            parts.append(f"  {scene.narrative}")
            if any(f.via_summary for f in scene.fragments):
                parts.append("  （含 Cold 摘要片段，可 /recall 唤醒细节）")
            parts.append("这和你说的有关吗？")
        elif not relevant:
            parts.append("（没有找到相关的旧记忆）我还不太了解这个，你能多说一点吗？")
        else:
            parts.append("我记得：")
            for h in relevant[:2]:
                m = h.memory
                if h.via_summary:
                    parts.append(f"  • [索引:{m.id[:6]}] {m.summary}（这是压缩摘要，可用 /recall {m.id[:6]} 唤醒细节）")
                else:
                    parts.append(f"  • {m.content}")
            parts.append("这和你说的有关吗？")
        return "\n".join(parts)

    def _generate_reply(self, query: str, relevant: list[Retrieved], scene: Scene | None = None) -> str:
        """生成回复：配置了 LLM 回复生成器且可用时，把检索结果注入上下文
        让 LLM 基于记忆回答；无相关记忆时由 LLM 直接回答（闲聊模式）。
        未配置 / 未设 key / 网络或解析出错 → 自动回退模板回复。"""
        if self.responder is not None and self.responder.available:
            try:
                mems = [
                    # 与模板路径一致：Cold 命中（via_summary）注入摘要文本
                    # 而非深藏 content——命中词在摘要里，content 可能不含
                    (h.memory.summary or h.memory.content, h.memory.mtype.value, h.strength)
                    for h in relevant[:3]
                ]
                if scene is not None:
                    # 场景叙事作为最高优先级的上下文注入（连贯场景优于碎片列表）
                    mems.insert(0, (f"场景：{scene.narrative}", "scene", scene.strength))
                # 人设演化档案：kind="setting" 的设定记忆按重要性注入 system
                # prompt——人设随记忆自主演化（无设定时不注入，行为不变）
                return call_responder(
                    self.responder, query, memories=mems or None,
                    persona_extras=self.persona_sheet(),
                )
            except Exception:
                pass  # 回退模板
        return self._template_reply(relevant, scene=scene)

    def respond(self, query: str, k: int | None = None) -> tuple[str, list[Retrieved]]:
        """返回 (回复文本, 被引用的记忆)。"""
        hits = self.retrieve(query, k)
        scene = self.compose_scene(query, k, hits=hits)  # 场景重建：相关片段拼成连贯场景
        self.remember(f"用户说：{query}", kind="turn")  # 对话本身也进记忆（去重）
        self._turns += 1
        if self._turns % self.cfg.sleep_interval_turns == 0:
            self.sleep()

        # 生长步进：每次对话都是观察 → 触发预测验证 / 模式提取 / 概念形成
        detected_topics = self.interest.detect_topics(query)
        self.growth.grow_step(query, topic=detected_topics[0] if detected_topics else None)

        relevant = [h for h in hits if h.total > 0.05]
        reply = self._generate_reply(query, relevant, scene=scene)
        self._observe()  # 持续观测：每轮对话后给所有记忆采样
        self.interest.apply_decay()
        return reply, hits

    # ---------- 睡眠巩固 ----------

    def sleep(self, duration: float | None = None) -> dict:
        """睡眠巩固：
        0) **回放**：按白天经历的时间顺序重放最近活跃的记忆——每次重放即一次
           再激活（access_count +1 + 观测采样：强度微调、语义化评分贡献）。
           完整睡眠（duration=None）重放全部候选；中断（duration 秒）时按
           replay_per_second 折算预算只重放一部分，**未回放的候选次日更模糊**
           （importance × replay_fog_factor）。重放不改 last_access——否则每次
           睡眠都重置衰减时钟，记忆会变得不朽。
        1) 闲置过久的 Hot 记忆降级为 Warm；
        2) 久未访问的低频 Warm 记忆聚类 → 压缩为 Cold 摘要。
        返回一份"梦境报告"。
        """
        now = self._now()
        report: dict = {"hot_demoted": [], "cold_compressed": 0, "clusters": 0,
                        "migrations": 0, "replay_candidates": 0, "replayed": [],
                        "replayed_count": 0, "unreplayed_count": 0, "fogged": []}

        # 0) 睡眠回放：先于迁移/压缩——被重放的记忆获得再激活强化（可能因此
        #    逃过压缩、推动情景语义化）；候选 = 非 Cold、非对话流水、最近窗口内
        #    活跃过的记忆，按经历时间（last_access）排序，早→晚。
        if self.cfg.replay:
            candidates = [
                m for m in self.store.all()
                if m.tier is not Tier.COLD and m.kind != "turn"
                and now - m.last_access <= self.cfg.replay_window_seconds
            ]
            candidates.sort(key=lambda m: m.last_access)
            budget = (len(candidates) if duration is None
                      else max(0, int(duration * self.cfg.replay_per_second)))
            replayed, unreplayed = candidates[:budget], candidates[budget:]
            for mem in replayed:
                mem.access_count += 1       # 再激活（测试效应；不改 last_access）
                self._record_sample(mem)    # 观测采样 → 曲线 / 语义化评分可见
                report["replayed"].append(mem.content[:20])
            for mem in unreplayed:
                if mem.importance >= self.cfg.freeze_importance:
                    continue  # 核心记忆豁免：连乘模糊系数会击穿冻结保护（0.9² 即跌破 0.8）
                old_imp = mem.importance
                mem.importance = max(self.cfg.importance_floor,
                                     old_imp * self.cfg.replay_fog_factor)
                report["fogged"].append({"content": mem.content[:20],
                                          "importance": round(old_imp, 4),
                                          "fogged": round(mem.importance, 4)})
            report["replay_candidates"] = len(candidates)
            report["replayed_count"] = len(replayed)
            report["unreplayed_count"] = len(unreplayed)

        # 0) 类型迁移：被反复检索的 episodic 逐渐语义化（"我经常去爬山" 替代
        #    50 次具体爬山）；低频 semantic 反向淡化为 episodic。
        #    - 双阈值滞回（3.0 / 0.8）避免来回振荡；
        #    - 对话流水（turn）是瞬时记录不迁移；Cold 摘要已压缩不参与；
        #    - 从未被使用过的 semantic（access_count < 2）不淡化——新存的事实
        #      不会一觉醒来就翻成情景。
        if self.cfg.semanticize:
            for mem in self.store.all():
                if mem.tier is Tier.COLD or mem.kind == "turn":
                    continue
                score = self._semanticization_score(mem)
                if mem.mtype is MemType.EPISODIC and score >= self.cfg.semanticize_threshold:
                    mem.mtype = MemType.SEMANTIC
                    mem.mtype_confidence = None  # 类型由迁移决定，不再有分类置信度
                    mem.migrations.append([round(now, 1), "episodic→semantic", round(score, 3)])
                    report["migrations"] += 1
                elif (
                    mem.mtype is MemType.SEMANTIC
                    and mem.access_count >= 2
                    and score < self.cfg.desemanticize_threshold
                ):
                    mem.mtype = MemType.EPISODIC
                    mem.mtype_confidence = None
                    mem.migrations.append([round(now, 1), "semantic→episodic", round(score, 3)])
                    report["migrations"] += 1

        for mem in self.store.by_tier(Tier.HOT):
            if now - mem.last_access > self._cold_after(mem):
                mem.tier = Tier.WARM
                report["hot_demoted"].append(mem.content[:20])

        candidates = [
            m
            for m in self.store.by_tier(Tier.WARM)
            if (now - m.last_access) > self._cold_after(m)
            and m.access_count <= self.cfg.cold_max_access
            and m.importance < 0.8  # 高价值记忆不入冷压缩候选池
        ]
        clusters = merge_similar(candidates)
        for cl in clusters:
            merged = "；".join(m.content for m in cl)
            summary = extractive_summary(merged)
            cl[0].demote_to_cold(summary, sources=cl)
            for other in cl[1:]:
                self.store.remove(other.id)
            report["cold_compressed"] += len(cl)
            report["clusters"] += 1
        self._observe()  # 睡眠巩固后也观测一轮
        self.learn_tau()  # 参数自适应：按观测校准各类型 τ（门控内自动跳过）
        self.learn_plasticity()  # 再巩固因子自适应：按修订日志校准各类型可塑性
        if self.cfg.evolve_on_sleep:
            # 自主演化：睡眠时反思记忆 + 联网查资料 → 沉淀新设定（人设成长）
            ev = self.evolve()
            report["evolved"] = ev.get("added", [])
            report["evolve_query"] = ev.get("query", "")
            report["evolve_web_n"] = ev.get("web_n", 0)
            report["evolve_ok"] = ev.get("ok", False)
        return report

    def recall(self, mem_id_prefix: str) -> Memory | None:
        """唤醒一条 Cold 记忆：摘要 → 重建 Warm 完整记忆（**move 语义**）。

        原 Cold 记忆在唤醒后从仓库移除——重建取代旧索引，而非并存：
        Cold→Warm→（闲置）→Cold 的往返不会产生记忆增殖，重复唤醒同一
        Cold 也不会生成多条 Warm。深藏细节（originals）已由 awaken 继承，
        移除不丢数据。
        """
        for mem in self.store.by_tier(Tier.COLD):
            if mem.id.startswith(mem_id_prefix):
                now = self._now()
                revived = self.store.awaken(mem, now=now)
                self.store.remove(mem.id)  # 原 Cold 移出（唤醒即取代）
                self._record_sample(revived)
                self._observe_awakening(mem, revived, now)
                return revived
        return None

    def _observe_awakening(self, cold: Memory, revived: Memory, now: float) -> None:
        """唤醒偏差观测：实测跳升强度 − 模型延续预测（未唤醒假想）。

        actual    = 唤醒后状态（last_access=now，recency 满格）在 now 的强度
        predicted = 原 Cold 状态（旧 last_access，按 τ 已衰减）在 now 的强度
        实测偏差 dev = actual − predicted：recall 恢复的强度，即"唤醒的剧烈程度"。

        **类型专属预期偏差**：同一事件、同一状态，只把 τ 换成模型信念
        （配置/已学习的类型 τ）再算一遍同样的跳升——expected。Δ 时间由压缩
        时机决定（闲置 > cold_after_tau × τ 才埋藏），所以预期自然包含该类型
        的压缩深度：技能类 τ 大 → 预期唤醒轻，情景类 τ 小 → 预期唤醒重。

        **可塑性调制**（τ↔可塑性联合估计器的观测层）：唤醒跳升按该类型实测
        可塑性因子缩放——dev 同时编码 τ 失准与可塑性：
          dev = base(τ真实) · (1 + g·(p实测 − 1))
          expected = base(τ模型) · (1 + g·(p信念 − 1))
        p实测 = applied_factor（配置了 true_reconsolidation_by_type 时即真实环境，
        否则等于信念——两刻度相同 → 比值纯 τ，旧行为不变）；信念刻度随
        learn_plasticity 收敛 → dev/expected 比值中的可塑性因子逐年消融
        （联合估计器两路 EMA 相互加速的耦合点，见 _joint_awakening_estimates）。
        记入 revived.awakenings 六元组 [时间戳, 实测偏差, 类型预期偏差, 唤醒时刻
        类型, 埋藏时长 Δt, 埋藏时检索次数]——后两列供联合估计器按衰减公式精确
        重算 τ 校正（旧四元组回退近似校正）。记录本身只要任一消费方开启即可。
        """
        if not (self.cfg.plasticity_from_awakenings or self.cfg.tau_from_awakenings):
            return
        tau = self._true_tau_for(revived)
        actual = self.strength_at_state(
            revived.mtype, revived.last_access, revived.access_count,
            revived.importance, now, tau_override=tau,
        )
        predicted = self.strength_at_state(
            revived.mtype, cold.last_access, cold.access_count,
            cold.importance, now, tau_override=tau,
        )
        dev = actual - predicted
        if dev <= 0:
            return
        tau_model = self.cfg.tau_for(revived.mtype)
        expected = self.strength_at_state(
            revived.mtype, revived.last_access, revived.access_count,
            revived.importance, now, tau_override=tau_model,
        ) - self.strength_at_state(
            revived.mtype, cold.last_access, cold.access_count,
            cold.importance, now, tau_override=tau_model,
        )
        g = self.cfg.awakening_plasticity_gain
        p_eff = self.cfg.applied_factor(revived.mtype, "drift")
        p_bel = self.cfg.reconsolidation_factor(revived.mtype, "drift")
        dev *= 1.0 + g * (p_eff - 1.0)
        expected *= 1.0 + g * (p_bel - 1.0)
        revived.awakenings.append(
            [round(now, 1), round(dev, 4), round(expected, 4), revived.mtype.value,
             round(now - cold.last_access, 1), cold.access_count]
        )
        del revived.awakenings[:-12]  # 只保留最近 12 条唤醒观测

    def spontaneous_recall(self, rng: random.Random | None = None) -> Memory | None:
        """心游 / 默认模式网络：无查询时按**强度加权**随机想起一条记忆。

        候选 = 非 Cold（Warm + Hot）——Cold 深藏记忆需要线索才能被唤起（recall()），
        不参与心游。权重 = 当前强度（越牢越容易被想起；触底记忆强度相同、偶尔
        也会冒出来）。被想起的记忆获得再激活测试效应（touch：次数 +1、时间刷新
        + 观测采样）——"反复突然想到的事情记得特别牢"；随即它成为当晚睡眠回放
        的候选（活跃窗口内），形成 心游 → 想起 → 再激活 → 回放 → 更牢 的闭环。
        无查询上下文，故不触发再巩固内容改写、不升级 Hot。rng 可注入（测试/
        对照实验确定性）；空记忆库返回 None。
        """
        rng = rng or random
        now = self._now()
        cands = [m for m in self.store.all() if m.tier is not Tier.COLD]
        if not cands:
            return None
        weights = [max(0.0, self._strength(m)) for m in cands]
        total = sum(weights)
        if total <= 0:
            return None
        r = rng.uniform(0.0, total)
        upto = 0.0
        picked = cands[-1]
        for m, w in zip(cands, weights):
            upto += w
            if r <= upto:
                picked = m
                break
        picked.touch(now)          # 再激活测试效应
        self._record_sample(picked)
        return picked

    def compose_scene(
        self,
        query: str,
        k: int | None = None,
        hits: list[Retrieved] | None = None,
    ) -> Scene | None:
        """场景重建：把检索命中的记忆与彼此相关的片段组合成一个连贯场景。

        认知对应：人脑回忆的是**片段组合**（场景重建）而非单条事实——
        安静时的场景闪回、睡前对白天的运转，都是把多条相关记忆按经历
        顺序重新组合成一段连贯叙事。

        - **种子**：检索命中（total > 0.05，排除对话流水）。未提供 hits 时
          内部调用 retrieve()（种子照常获得检索测试效应与再巩固）；
        - **扩展**：与任一种子"相关"（`_fragment_relatedness`：嵌入余弦与
          n-gram 共享度取大 ≥ scene_similarity）的其它记忆，且出生时间在
          scene_time_window 内（时间连贯性：跨年片段不属于同一场景）；
        - **排序**：按 created_at（经历顺序）从早到晚，赋予 开头/中间/结尾
          时序角色；叙事用时序连接词拼接（先是…；接着…；最后…）；
        - **整体度量**：strength = rel 加权平均强度（场景显著性），
          coherence = 片段间平均最大相关度（连贯性）；
        - **测试效应**：被纳入场景的扩展片段同样 touch + 观测采样（回忆
          即强化）；scene_reconsolidates=True 时扩展片段（非 Cold）同样走
          再巩固——每次重建都是对原始记忆的一次微调（再巩固窗口语义）；
        - **Cold 摘要片段**：via_summary=True，叙事用摘要文本（可 /recall
          唤醒细节）；相关片段不足 2 个或开关关闭返回 None。
        """
        if not self.cfg.scene_reconstruction:
            self.last_scene = None
            return None
        now = self._now()
        if hits is None:
            hits = self.retrieve(query, k)
        seeds = [h for h in hits if h.total > 0.05 and h.memory.kind != "turn"]
        if not seeds:
            self.last_scene = None
            return None
        anchor = statistics.median([h.memory.created_at for h in seeds])
        frags: list[SceneFragment] = [
            SceneFragment(
                memory=h.memory, rel=h.relevance, strength=h.strength,
                via_summary=h.via_summary,
            )
            for h in seeds
        ]
        included = {f.memory.id for f in frags}
        expansions: list[tuple[Memory, float]] = []
        for h in seeds:
            for m in self.store.all():
                if m.id in included or m.kind == "turn":
                    continue
                if abs(m.created_at - anchor) > self.cfg.scene_time_window:
                    continue
                sim = _fragment_relatedness(h.memory, m, require_shared=True)
                if sim >= self.cfg.scene_similarity:
                    expansions.append((m, sim))
                    included.add(m.id)
        # 种子优先，扩展按与种子的相关度降序截断到上限
        expansions.sort(key=lambda p: p[1], reverse=True)
        room = max(0, self.cfg.scene_max_fragments - len(frags))
        for m, sim in expansions[:room]:
            frags.append(
                SceneFragment(
                    memory=m, rel=sim, strength=self._strength(m),
                    via_summary=(m.tier is Tier.COLD),
                )
            )
        if len(frags) < 2:
            self.last_scene = None
            return None
        # 按经历顺序（created_at）排序，赋时序角色
        frags.sort(key=lambda f: f.memory.created_at)
        n = len(frags)
        for i, f in enumerate(frags):
            f.role = "开头" if i == 0 else ("结尾" if i == n - 1 else "中间")
        narrative = "；".join(
            f"{('先是' if i == 0 else ('接着' if i < n - 1 else '最后'))}「{f.text}」"
            for i, f in enumerate(frags)
        ) + "。"
        title_frag = max(frags, key=lambda f: f.rel)
        title = title_frag.text if len(title_frag.text) <= 14 else title_frag.text[:14] + "…"
        w = sum(f.rel for f in frags)
        strength = sum(f.strength * f.rel for f in frags) / w if w > 0 else 0.0
        coherence = statistics.mean(
            max(_fragment_relatedness(f.memory, g.memory) for g in frags if g.memory.id != f.memory.id)
            for f in frags
        )
        # 测试效应：扩展片段获得再激活（种子已在 retrieve 里 touch/再巩固）
        seed_ids = {h.memory.id for h in seeds}
        qv = embed_text(query.strip().lower())
        for f in frags:
            if f.memory.id in seed_ids:
                continue
            f.memory.touch(now)
            self._record_sample(f.memory)
            if self.cfg.scene_reconsolidates and f.memory.tier is not Tier.COLD:
                self._reconsolidate(f.memory, query, qv, f.rel)
        scene = Scene(
            query=query, title=title, fragments=frags, narrative=narrative,
            strength=strength, coherence=coherence, created=now,
        )
        self.last_scene = scene
        return scene

    def plot_by_type(self, path: str = "memories_by_type.svg", horizon_seconds: float | None = None) -> str:
        """按类型分面板导出曲线：技能/语义/情景各一张子图，共享横轴对比遗忘斜率。"""
        from .visualize import render_svg_by_type

        return render_svg_by_type(self, path=path, horizon_seconds=horizon_seconds)

    def plot_tau_convergence(self, base_path: str = "tau_convergence") -> list[str]:
        """导出 learn_tau 两路信号的收敛轨迹：SVG 图 + 轮次明细 CSV。

        图：每个有学习记录的类型一张面板——上子图 τ（对数轴）画配置 τ 的
        EMA 轨迹（实线）与两路独立估计（干净段紫虚线 / 唤醒橙虚线）逼近真实
        τ（灰虚线）；下子图画唤醒中位比值 dev/expected 随轮次逼近 1。
        两条 τ_est 都向真实 τ 收敛 = 两路信号互相印证。
        """
        from .visualize import export_tau_trajectory, render_tau_convergence

        return [
            render_tau_convergence(self, f"{base_path}.svg"),
            export_tau_trajectory(self, f"{base_path}.csv"),
        ]

    def plot_interactive(self, path: str = "memories_dashboard.html",
                         horizon_seconds: float | None = None,
                         aggregations: list[dict] | None = None) -> str:
        """导出多视图仪表盘（单文件 HTML，浏览器打开）。

        四个联动视图：强度曲线（线宽=重要性、环标=检索事件）、记忆地图气泡图、
        层级×类型分布、最强记忆 Top5；点击任意面板全局高亮，层级切换全局生效。
        `aggregations` = [{"mtype", "events": ['memory_id:序号', ...]}, ...]
        （与 --aggregations 同格式）——回放历史聚合结论（health.aggregations，
        含 verdict 徽章 + resolved 自动附带剔除后证据包），点击可在主图高亮
        对应事件子集。
        """
        from .interactive import render_interactive_html

        return render_interactive_html(self, path=path,
                                       horizon_seconds=horizon_seconds,
                                       aggregations=aggregations)

    # ---------- 参数自适应学习器 ----------

    def learn_tau(self, force: bool = False) -> dict:
        """根据实测采样与预测的偏差自动调整各类型 τ（遗忘曲线学习器）。

        信号：两路互补的"预测 vs 实际"观测（_combined_tau_estimate 合并）——
        ① fit_report 从干净衰减段反推的实测 τ（观测时长充分时）；
        ② 唤醒偏差代理：实测跳升深于类型预期（dev > expected）→ 该类型衰减比
        信念快 → τ 下调（见 _tau_awakening_estimate）。更新规则：
        - 门控：任一源充足（干净段 ≥ tau_min_segments 或唤醒观测 ≥ tau_min_awakenings），
          且合并 τ_est 与配置 τ 偏差 > 0.5%；
        - EMA：τ_new = (1−α)·τ_old + α·τ_est，α = tau_learning_rate × 置信度；
        - 置信度 = 两源各自置信度之和（封顶 1，观测越充分越敢动）；
        - 夹在 [tau_min_seconds, tau_max_seconds]，结果持久化到 store.meta。
        返回 {"updated": [...], "skipped": [...]}。
        """
        report: dict = {"updated": [], "skipped": []}
        if not self.cfg.tau_learning and not force:
            return report
        r = self.fit_report()
        self._joint_drift_samples = {}  # 新一轮：重置（learn_plasticity 消费本轮样本）
        for t in MemType:
            d = r["by_type"][t.value]
            old_tau = self.cfg.tau_for(t)
            joint = None
            if self.cfg.joint_awakening:
                # 联合估计器：同一批唤醒事件 → τ 与 drift 两路估计。τ 通道喂给
                # 本学习器；drift 样本存起来供 learn_plasticity 直接消费（避免
                # 同轮二次扫描破坏跨轮耦合的 τ 参考）。
                tau_ests, drift_ests, n_aw, ratios = self._joint_awakening_estimates(t)
                self._joint_drift_samples[t.value] = drift_ests
                joint = (tau_ests, n_aw, ratios)
            src = self._tau_source_estimates(t, d, joint=joint)
            tau_est, confidence = self._combined_tau_estimate(t, d, src=src)
            if tau_est is None:
                report["skipped"].append({"type": t.value, "reason": "观测不足"})
                continue
            alpha = self.cfg.tau_learning_rate * confidence
            new_tau = (1.0 - alpha) * old_tau + alpha * tau_est
            new_tau = min(self.cfg.tau_max_seconds, max(self.cfg.tau_min_seconds, new_tau))
            # 出厂边界钳制：学习器只能在进化预设的区间内微调，不可翻转方向
            innate = self.cfg.innate_bounds.get(t)
            if innate and not innate.frozen:
                new_tau = min(innate.tau_max, max(innate.tau_min, new_tau))
            if innate and innate.frozen:
                # 出厂冻结类型：完全不可学习，保持出厂 τ
                new_tau = old_tau
            if abs(new_tau - old_tau) / max(old_tau, 1e-9) < 0.005:
                report["skipped"].append({"type": t.value, "reason": "偏差过小"})
                continue
            self.cfg.tau_by_type[t] = new_tau
            # 学习历史 11 列：前 6 列同旧格式（时间/类型/旧τ/合并估计/新τ/置信度），
            # 7~9 列为**两路信号的独立估计**（干净段 τ_est / 唤醒 τ_est / 唤醒中位
            # 比值 dev÷expected）；10~11 列为本次更新实际使用的**唤醒信号原始值**
            # （中位 dev / 中位 expected）——方向可复盘（dev > expected → 该类型
            # 埋得比信念深 → 下调 τ），比值趋 1 = 已校准。旧 9 列格式向后兼容。
            dev_m, exp_m, ratio_m = self._awakening_signal_medians(t)
            self._learn_history.append(
                [self._now(), t.value, round(old_tau, 2), round(tau_est, 2),
                 round(new_tau, 2), round(confidence, 3),
                 round(src["clean"]["est"], 2) if src["clean"] else None,
                 round(src["awakening"]["est"], 2) if src["awakening"] else None,
                 round(src["awakening"]["ratio"], 4) if src["awakening"] else None,
                 round(dev_m, 4) if dev_m is not None else None,
                 round(exp_m, 4) if exp_m is not None else None,
                 ]
            )
            report["updated"].append(
                {"type": t.value, "old_tau": round(old_tau, 2), "tau_est": round(tau_est, 2),
                 "new_tau": round(new_tau, 2), "confidence": round(confidence, 3)}
            )
        self._persist_learned_tau()
        return report

    def _tau_awakening_estimate(self, mtype: MemType) -> tuple[float | None, int, float | None]:
        """唤醒偏差 → τ 代理估计（第二观测源，与干净段反推互补）。

        实测跳升深于类型预期（dev > expected）→ 该类型埋得比信念更深 →
        真实衰减比信念快 → τ 应下调：
            est = τ_model × (expected / dev)^awakening_tau_gain
        浅埋区间约等于真实 τ 的反演；深埋时 dev、expected 同饱和于强度下限
        → 比值 → 1 → 估计 → τ_model（保守无信号）；自洽环境（无 true_tau）
        实测 == 预期 → 估计 == τ_model（不误调）。按类型归组取中位数抗离群。
        返回 (中位数估计, 事件数, 中位比值 dev/expected)——比值 > 1 表示唤醒
        比类型预期剧烈（τ 应下调），收敛轨迹图按轮次展示它逼近 1。
        无观测或开关关闭返回 (None, 0, None)。
        """
        if not self.cfg.tau_from_awakenings:
            return None, 0, None
        ests: list[float] = []
        ratios: list[float] = []
        for mem in self.store.all():
            for aw in mem.awakenings:
                if len(aw) < 4 or aw[3] != mtype.value:
                    continue
                dev, expected = float(aw[1]), float(aw[2])
                if dev <= 0 or expected <= 0:
                    continue
                ratios.append(dev / expected)
                est = self.cfg.tau_for(mtype) * (expected / dev) ** self.cfg.awakening_tau_gain
                est = min(self.cfg.tau_max_seconds, max(self.cfg.tau_min_seconds, est))
                ests.append(est)
        if not ests:
            return None, 0, None
        return (
            statistics.median(ests), len(ests), statistics.median(ratios),
        )

    def _joint_awakening_estimates(
        self, mtype: MemType,
    ) -> tuple[list[float], list[float], int, list[float]]:
        """τ↔可塑性联合估计器：一次唤醒事件 → τ 与 drift 两个估计（跨轮耦合）。

        观测（见 _observe_awakening）：
            dev      = base(τ真实) · (1 + g·(p实测 − 1))
            expected = base(τ模型) · (1 + g·(p信念 − 1))
        比值 = [base(τ真实)/base(τ模型)] · [可塑性因子]——单事件两个未知（τ 失准、
        可塑性偏差）数学上不可分离，采用**双向跨轮耦合**（两路 EMA 相互加速）：

        - τ 通道：比值先剥掉**上一轮估计的可塑性因子**再反演 τ（否则纯可塑性
            失准会被误读为 τ 失准而误调）：
              L_τ = (dev/expected) · (S_信念/P_上一轮)   → τ_est = τ模型·(1/L_τ)^gain
            其中 S_信念 = 1+g(p信念−1)（记录时已知）、P_上一轮 = 1+g(p_est_上一轮−1)
            ——可塑性估计收敛 → P_上一轮 → 实测刻度 → L_τ → 纯 τ 比值；
        - drift 通道：拿 τ 解释不了的残余——用**去可塑性**后的 dev 做衰减公式的
            精确反演得 τ 参考（R(τ参考) 使 dev 中 τ 分量归零）：
              R_t = dev/P_上一轮 − f   → τ_ref（对数反演，物理不可行回退 τ模型）
              expected_corr = expected − (R(τ模型) − R(τ_ref))·S_信念
              drift_ratio   = dev / expected_corr
              p_est = 1 + (drift_ratio·S_信念 − 1)/g              # 调制反演
            纯 τ 失准（可塑性校准）：R_t = R(τ真实) → expected_corr = dev →
            drift_ratio=1 → drift 不动（消除旧双计数：旧代理把整个比值同时判给
            τ 与 drift）；纯可塑性失准（τ 校准）：P_上一轮→实测刻度 → R_t = R(τ模型)
            → 无 τ 校正 → drift_ratio = 可塑性因子 → p_est = 实测因子。

        识别边界（诚实记录）：单事件无法分离 τ 与可塑性，首轮无 P_上一轮 知识
        （退化为顺序归因）；可塑性估计滞后时两路存在短暂互扰，随估计收敛消融。
        返回 (τ估计列表, drift估计列表, 事件数, 比值列表)。开关门控与两路独立
        代理一致（tau_from_awakenings / plasticity_from_awakenings）。
        """
        g = self.cfg.awakening_plasticity_gain
        if g <= 0:
            return [], [], 0, []
        sc = self.scorer.cfg
        denom = sc.w_recency + sc.w_freq + sc.w_importance
        w_rec = sc.w_recency / denom
        w_freq = sc.w_freq / denom
        tau_ests: list[float] = []
        drift_ests: list[float] = []
        ratios: list[float] = []
        refs: list[float] = []
        for mem in self.store.all():
            for aw in mem.awakenings:
                if len(aw) < 4 or aw[3] != mtype.value:
                    continue
                dev, expected = float(aw[1]), float(aw[2])
                if dev <= 0 or expected <= 0:
                    continue
                tau_model = self.cfg.tau_for(mtype)
                p_b = self.cfg.reconsolidation_factor(mtype, "drift")
                S = 1.0 + g * (p_b - 1.0)                     # 记录时的信念刻度
                p_prev = self._joint_p_ref.get(mtype.value, p_b)  # 上一轮可塑性估计
                P_prev = 1.0 + g * (p_prev - 1.0)             # 上一轮估计的实测刻度
                if self.cfg.tau_from_awakenings:
                    # ① τ 通道：剥掉上一轮估计的可塑性因子 → 纯 τ 比值
                    # （纯 τ 失准时 P_prev≈S → 退化为旧比值，行为逐位兼容；
                    # 纯可塑性失准时比值被清洗 → τ 不被误调）
                    ratio = (expected / dev) * (P_prev / S)
                    tau_est = tau_model * ratio ** self.cfg.awakening_tau_gain
                    tau_est = min(self.cfg.tau_max_seconds,
                                  max(self.cfg.tau_min_seconds, tau_est))
                    tau_ests.append(tau_est)
                    ratios.append(dev / expected)
                if not self.cfg.plasticity_from_awakenings:
                    continue
                # ② drift 通道：τ 参考（**上一轮**事件批的精确反演中位数——不是
                # 本轮事件的，否则会把本轮可塑性信号吸收进 τ 而饿死 drift）按
                # 衰减公式重算校正后的预期跳升 → 残余 → 调制反演。
                tau_prev = self._joint_tau_ref.get(mtype.value, tau_model)
                dt = float(aw[4]) if len(aw) > 4 and aw[4] is not None else None
                n_cold = int(aw[5]) if len(aw) > 5 and aw[5] is not None else None
                f = (w_freq * math.exp(-(n_cold or 0) / sc.kappa)
                     * (1.0 - math.exp(-1.0 / sc.kappa)))
                tau_ref = None
                if dt and dt > 0:
                    # τ 参考按**信念刻度**反演（不用 P_prev——那是 drift 自己的
                    # 输出，喂回去会形成自放大回路）。纯 τ 失准时 dev/S = R(τ真实)
                    # → τ_ref 精确 → 残余归零（双计数消除）；可塑性失准时反演被
                    # 可塑性因子抬出物理界 → 回退不校正（残余 = 全比值）。
                    r_t = dev / S - f
                    if 0.0 < r_t <= w_rec:
                        arg = 1.0 - r_t / w_rec
                        if 0.0 < arg < 1.0:
                            tau_ref = -dt / math.log(arg)
                            tau_ref = min(self.cfg.tau_max_seconds,
                                          max(self.cfg.tau_min_seconds, tau_ref))
                if tau_ref is not None:
                    refs.append(tau_ref)
                # 残余：模型按 τ_prev 延续应看到的跳升 vs 实测（含可塑性调制）。
                # 直接从 τ_ref 重算 base（(R(τ_ref)+f)·S）而非在记录时的 expected
                # 上做差——后者依赖事件记录时的旧 τ，跨轮事件池里会残留旧 τ 的
                # 校正误差（τ-only 场景 drift 被旧事件抬高）。纯 τ 失准 + τ_ref
                # 精确时：expected_corr = (R(τ真实)+f)·S = dev → 残余归零。
                if dt and dt > 0:
                    r_ref = w_rec * (1.0 - math.exp(-dt / max(tau_prev, 1e-9)))
                    expected_corr = (r_ref + f) * S
                else:  # 旧四元组：线性区近似（R ∝ 1/τ）
                    expected_corr = expected * (tau_model / max(tau_prev, 1e-9))
                if expected_corr > 0:
                    drift_ratio = dev / expected_corr
                    p_est = 1.0 + (drift_ratio * S - 1.0) / g
                    p_est = min(self.cfg.plasticity_max,
                                max(self.cfg.plasticity_min, p_est))
                    drift_ests.append(p_est)
        if refs:
            self._joint_tau_ref[mtype.value] = statistics.median(refs)
        if drift_ests:
            self._joint_p_ref[mtype.value] = statistics.median(drift_ests)
        return tau_ests, drift_ests, len(tau_ests), ratios

    def _tau_source_estimates(
        self, mtype: MemType, seg: dict, joint: tuple | None = None,
    ) -> dict:
        """两路 τ 观测的独立估计（合并与轨迹可视化共用同一门控）。

        返回 {"clean": {"est", "conf", "n"} | None,
        "awakening": {"est", "conf", "n", "ratio"} | None}——每路各自过
        门控（干净段 ≥ tau_min_segments / 唤醒观测 ≥ tau_min_awakenings），
        缺源为 None。收敛轨迹图按轮次记录两路的独立估计与唤醒比值。
        joint: 联合估计器开启时由 learn_tau 预计算的 (τ估计列表, 事件数, 比值
        列表)——传 None 则走两路独立代理（_tau_awakening_estimate）。
        """
        src: dict = {"clean": None, "awakening": None}
        seg_est = seg.get("tau_est")
        if seg_est is not None and seg["clean"] >= self.cfg.tau_min_segments:
            conf = min(1.0, seg["clean_seconds"] / max(self.cfg.tau_for(mtype), 1e-9))
            src["clean"] = {"est": float(seg_est), "conf": conf, "n": seg["clean"]}
        if joint is not None:
            tau_ests, n_aw, ratios = joint
            if tau_ests and n_aw >= self.cfg.tau_min_awakenings:
                conf = min(1.0, n_aw / (2.0 * self.cfg.tau_min_awakenings))
                src["awakening"] = {
                    "est": float(statistics.median(tau_ests)), "conf": conf,
                    "n": n_aw,
                    "ratio": (float(statistics.median(ratios)) if ratios else None),
                }
        else:
            aw_est, n_aw, ratio = self._tau_awakening_estimate(mtype)
            if aw_est is not None and n_aw >= self.cfg.tau_min_awakenings:
                conf = min(1.0, n_aw / (2.0 * self.cfg.tau_min_awakenings))
                src["awakening"] = {"est": aw_est, "conf": conf, "n": n_aw, "ratio": ratio}
        return src

    def _combined_tau_estimate(
        self, mtype: MemType, seg: dict, src: dict | None = None,
    ) -> tuple[float | None, float]:
        """合并两路 τ 观测：干净段反推（拟合报告）+ 唤醒偏差代理。

        各自带置信度（观测时长占比 / 事件数占比），按置信度加权平均；总置信度
        取两源之和（封顶 1）——两路信号互相印证时更敢动。任一源不足则只用
        另一源。src 可预传（learn_tau 已算好，避免重复扫描）。
        返回 (τ 估计, 置信度)；两者都无返回 (None, 0)。
        """
        src = src if src is not None else self._tau_source_estimates(mtype, seg)
        ests: list[tuple[float, float]] = []  # (估计, 置信度权重)
        # 只收 conf > 0 的源：conf=0 的观测（如干净段时长被舍入成 0）本就不该
        # 参与加权，留着会让 w=0 除零（TIME_SCALE=1 测试环境曾复现）。
        for s in (src["clean"], src["awakening"]):
            if s is not None and s["conf"] > 0:
                ests.append((s["est"], s["conf"]))
        if not ests:
            return None, 0.0
        w = sum(c for _, c in ests)
        combined = sum(e * c for e, c in ests) / w
        return combined, min(1.0, w)

    def _persist_learned_tau(self) -> None:
        self.store.meta["learned_tau"] = {t.value: self.cfg.tau_for(t) for t in MemType}

    def _awakening_signal_medians(self, mtype: MemType) -> tuple[float | None, float | None, float | None]:
        """该类型本轮全部唤醒事件的中位信号（供学习历史复盘）。

        与学习器共用同一门控：只收 len≥4 的唤醒行、按唤醒时刻类型归组、
        dev/expected 都 > 0 才参与（旧格式三元组跳过）。返回
        (中位 dev, 中位 expected, 中位比值 dev/expected)——每项独立取中位数
        （比值的"中位数的中位数"语义与收敛轨迹图的 src["awakening"]["ratio"]
        一致）；无事件时全 None。
        """
        devs: list[float] = []
        exps: list[float] = []
        ratios: list[float] = []
        for mem in self.store.all():
            for aw in mem.awakenings:
                if len(aw) < 4 or aw[3] != mtype.value:
                    continue
                dev, expected = float(aw[1]), float(aw[2])
                if dev <= 0 or expected <= 0:
                    continue
                devs.append(dev)
                exps.append(expected)
                ratios.append(dev / expected)
        if not devs:
            return None, None, None
        return (
            statistics.median(devs), statistics.median(exps), statistics.median(ratios),
        )

    # ---------- 再巩固因子自适应学习器 ----------

    def _plasticity_samples(self) -> dict:
        """从再巩固修订日志 + 唤醒偏差观测估计每类型两通道的实测因子。

        修订日志每行含事件发生时的类型与实际应用的因子（见 _reconsolidate）——
        按事件时的类型归组，记忆迁移后不会被误归属到新类型。旧格式行（无因子
        记录）跳过。唤醒偏差观测（awakenings）换算成漂移因子代理样本，同样按
        唤醒时刻的类型归组——两路信号汇入 drift 通道的同一事件池。

        联合估计器开启时走 _joint_awakening_estimates 的跨轮耦合样本（learn_tau
        本轮已算好则直接消费——同一批事件的 τ 与 drift 归因一致，避免同轮二次
        扫描污染 τ 参考；独立调用 learn_plasticity 时现算）。关闭时回退旧代理
        （_awakening_drift_estimate：把整个比值判给可塑性——双计数，保留给对照）。
        """
        out = {t.value: {"drift": [], "importance": []} for t in MemType}
        for mem in self.store.all():
            for rev in mem.revisions:
                if len(rev) < 7:
                    continue  # 旧格式修订日志，无法测量
                mtype, f_drift, f_imp = rev[4], rev[5], rev[6]
                if mtype in out:
                    out[mtype]["drift"].append(float(f_drift))
                    out[mtype]["importance"].append(float(f_imp))
        if self.cfg.joint_awakening:
            for t in MemType:
                drift_s = self._joint_drift_samples.get(t.value)
                if not drift_s:
                    # learn_plasticity 独立调用：本轮尚无联合扫描 → 现算（并更新
                    # 跨轮 τ 参考——以本次调用为新一轮）。
                    _, drift_s, _, _ = self._joint_awakening_estimates(t)
                    self._joint_drift_samples[t.value] = drift_s
                out[t.value]["drift"].extend(drift_s)
        elif self.cfg.plasticity_from_awakenings:
            for mem in self.store.all():
                for aw in mem.awakenings:
                    if len(aw) < 4:
                        continue  # 旧格式三元组（无类型预期偏差），无法换算
                    mtype = aw[3]
                    if mtype not in out:
                        continue
                    out[mtype]["drift"].append(
                        self._awakening_drift_estimate(mtype, float(aw[1]), float(aw[2]))
                    )
        return out

    def _awakening_drift_estimate(self, mtype: str, dev: float, expected: float) -> float:
        """唤醒偏差 → 漂移因子代理估计（一维观测换算）。

        信号 = 实测偏差相对**该类型预期偏差**（模型自身预测）的偏离：
        est = belief × (1 + gain × (dev / expected − 1))。
        expected 由模型信念 τ 在同一唤醒事件上算出——类型专属、自带压缩时机
        （技能类 τ 大 → 预期唤醒轻；情景类 τ 小 → 预期唤醒重）。dev > expected
        → 环境让该类型埋得比信念更深（τ 失准）→ 唤醒更剧烈 → 可塑性上调；
        反之下调。单一类型有观测即可校准（预期来自模型而非跨类型对比）。
        深埋时实测与预期都饱和于强度下限 → 比值 → 1 → 保守无信号（无偏）。
        按当前信念换算，夹在 [plasticity_min, plasticity_max]。
        """
        belief = self.cfg.reconsolidation_factor(MemType(mtype), "drift")
        if expected <= 0:
            return belief  # 无预期可参照 → 保守不改
        est = belief * (1.0 + self.cfg.awakening_dev_gain * (dev / expected - 1.0))
        return min(self.cfg.plasticity_max, max(self.cfg.plasticity_min, est))

    def learn_plasticity(self, force: bool = False) -> dict:
        """根据修订日志 + 唤醒偏差观测的实测可塑性校准各类型再巩固因子（与 learn_tau 同构）。

        信号：① 每次回忆事件实际应用的因子（有 true_reconsolidation_by_type 时
        为真实环境值，否则等于配置）——它偏离配置因子的程度即"预测偏差"；
        ② 唤醒偏差观测（awakenings）换算的漂移因子代理——实测跳升相对该类型
        预期偏差（模型自身预测，含类型 τ 与压缩时机）越剧烈，该类型可塑性越
        活跃（见 _awakening_drift_estimate）。
        两路信号汇入同一事件池后走统一的 EMA 更新。更新：
        - 门控：每类型每通道事件数 ≥ plasticity_min_events 且偏差 > 1%；
        - EMA：factor_new = (1−α)·factor_old + α·median(实测)，α = 学习率 × 置信度；
        - 置信度 = min(1, 事件数 / (2 × min_events))（事件越多越敢动）；
        - 夹在 [plasticity_min, plasticity_max]，结果持久化到 store.meta。
        返回 {"updated": [...], "skipped": [...]}。
        """
        report: dict = {"updated": [], "skipped": []}
        if not self.cfg.plasticity_learning and not force:
            return report
        samples = self._plasticity_samples()
        # 每类型本轮唤醒信号中位数（dev/expected/比值）——记入学习历史复盘
        aw_signals: dict[str, tuple] = {
            t.value: self._awakening_signal_medians(t) for t in MemType
        }
        for t in MemType:
            for channel in ("drift", "importance"):
                evs = samples[t.value][channel]
                if len(evs) < self.cfg.plasticity_min_events:
                    report["skipped"].append(
                        {"type": t.value, "channel": channel,
                         "reason": f"事件 {len(evs)} < {self.cfg.plasticity_min_events}"}
                    )
                    continue
                est = statistics.median(evs)
                belief = self.cfg.reconsolidation_factor(t, channel)
                confidence = min(1.0, len(evs) / (2.0 * self.cfg.plasticity_min_events))
                alpha = self.cfg.plasticity_learning_rate * confidence
                new = (1.0 - alpha) * belief + alpha * est
                new = min(self.cfg.plasticity_max, max(self.cfg.plasticity_min, new))
                # 出厂边界钳制：再巩固因子同样受进化预设区间约束
                innate = self.cfg.innate_bounds.get(t)
                if innate and not innate.frozen:
                    bound_min = {
                        "drift": innate.drift_min,
                        "importance": innate.importance_min,
                    }.get(channel, self.cfg.plasticity_min)
                    bound_max = {
                        "drift": innate.drift_max,
                        "importance": innate.importance_max,
                    }.get(channel, self.cfg.plasticity_max)
                    new = min(bound_max, max(bound_min, new))
                if innate and innate.frozen:
                    new = belief  # 出厂冻结：保持出厂因子
                if abs(new - belief) / max(belief, 1e-9) < 0.01:
                    report["skipped"].append(
                        {"type": t.value, "channel": channel, "reason": "偏差过小"}
                    )
                    continue
                merged = dict(self.cfg.reconsolidation_by_type.get(t, {}))
                merged[channel] = new
                self.cfg.reconsolidation_by_type[t] = merged
                # 学习历史 10 列：前 7 列同旧格式（时间/类型/通道/旧因子/估计/
                # 新因子/置信度），后 3 列为本次更新实际使用的**唤醒信号原始值**
                # （中位 dev / 中位 expected / 中位比值）——可复盘"这次更新是
                # 由多剧烈的唤醒驱动"（dev > expected → 可塑性上调的触发信号）。
                dev_m, exp_m, ratio_m = aw_signals[t.value]
                self._plasticity_history.append(
                    [self._now(), t.value, channel, round(belief, 4), round(est, 4),
                     round(new, 4), round(confidence, 3),
                     round(dev_m, 4) if dev_m is not None else None,
                     round(exp_m, 4) if exp_m is not None else None,
                     round(ratio_m, 4) if ratio_m is not None else None,
                     ]
                )
                report["updated"].append(
                    {"type": t.value, "channel": channel, "old": round(belief, 4),
                     "est": round(est, 4), "new": round(new, 4),
                     "confidence": round(confidence, 3)}
                )
        self._persist_learned_plasticity()
        return report

    def _persist_learned_plasticity(self) -> None:
        """只持久化显式配置过的类型/通道——未配置的类型重启后保持默认因子。"""
        self.store.meta["learned_plasticity"] = {
            t.value: {k: float(v) for k, v in d.items()}
            for t in MemType
            if (d := self.cfg.reconsolidation_by_type.get(t))
        }

    # ---------- 持续观测与贴合度 ----------

    def fit_report(self) -> dict:
        """预测 vs 真实遗忘的贴合度报告（详见 visualize.fit_report）。"""
        from .visualize import fit_report

        return fit_report(self)

    def format_fit(self) -> str:
        from .visualize import format_fit_report

        return format_fit_report(self)

    # ---------- 可视化 ----------

    def plot_curves(self, base_path: str = "memories_curves", horizon_seconds: float | None = None) -> list[str]:
        """导出各层记忆的强度曲线：.svg 曲线图 + .csv / .json 数据。

        返回生成的文件路径列表。
        """
        from .visualize import (
            default_horizon,
            export_awakenings_csv,
            export_csv,
            export_json,
            render_svg,
            render_svg_by_type,
        )

        now = self._now()
        horizon = horizon_seconds or default_horizon(self)
        files = [
            render_svg(self, f"{base_path}.svg", horizon_seconds=horizon, now=now),
            render_svg_by_type(self, f"{base_path}_by_type.svg", horizon_seconds=horizon, now=now),
            export_csv(self, f"{base_path}.csv", horizon_seconds=horizon, now=now),
            # 唤醒事件明细（实测/预期/比值），与曲线 CSV 按 memory_id 连接
            export_awakenings_csv(self, f"{base_path}_awakenings.csv", now=now),
            export_json(self, f"{base_path}.json", horizon_seconds=horizon, now=now),
        ]
        return files

    # ---------- 状态 ----------

    def profile_table(self) -> str:
        """记忆类型画像表格：各类型 τ / 再巩固因子 / 压缩阈值 + 唤醒信号列 +
        τ 两路信号列（干净段 / 唤醒方向 + 一致性）。

        唤醒信号列来自真实记忆库的 awakenings 扫描（方向 + 一致性）；两路信号
        列来自 tau_learner_health（单一事实源）——与仪表盘画像面板、
        --export-signals 的 CSV 合表同源，终端 / 仪表盘 / CSV 三处输出一致。"""
        from .profiles import format_profiles

        return format_profiles(self.cfg, awakening_signal_stats(self),
                               tau_learner_health(self))

    def signal_drift_table(self, days: float = 30.0) -> str:
        """唤醒信号漂移对比表（CLI /signal）：最近 N 天 vs 更早的方向一致性。"""
        from .profiles import TYPE_LABELS  # 函数级导入：agent 被多方早期导入

        p = awakening_signal_periods(self, recent_seconds=days * 86400 * TIME_SCALE)
        lines = [
            f"唤醒信号漂移（近 {days:g} 天 vs 更早——方向一致性是否随时间变化）",
            f"{'类型':<6}{'近期（方向·一致性·条数）':<26}早期（方向·一致性·条数）  判定",
        ]
        _dir_label = {"up": "↑上调", "down": "↓下调", "flat": "=持平"}
        for t in MemType:
            d = p["by_type"][t.value]
            r, e, verdict = d["recent"], d["earlier"], d["verdict"]
            r_txt = (f"{_dir_label[r['dominant']]}·{r['consistency']:.0%}（{r['events']}条）"
                     if r.get("events") else "无观测")
            e_txt = (f"{_dir_label[e['dominant']]}·{e['consistency']:.0%}（{e['events']}条）"
                     if e.get("events") else "无观测")
            mark = "⚠" if verdict in ("方向翻转", "一致性变化") else "✔"
            lines.append(f"{TYPE_LABELS[t]:<6}{r_txt:<26}{e_txt:<20}{mark} {verdict}")
        return "\n".join(lines)

    def stats(self) -> str:
        hot, warm, cold = (
            len(self.store.by_tier(Tier.HOT)),
            len(self.store.by_tier(Tier.WARM)),
            len(self.store.by_tier(Tier.COLD)),
        )
        skill = sum(1 for m in self.store.all() if m.mtype is MemType.SKILL)
        sem = sum(1 for m in self.store.all() if m.mtype is MemType.SEMANTIC)
        epi = sum(1 for m in self.store.all() if m.mtype is MemType.EPISODIC)
        return (
            f"记忆总数 {hot + warm + cold} = Hot {hot} / Warm {warm} / Cold {cold} "
            f"（技能 {skill} / 语义 {sem} / 情景 {epi} · 已对话 {self._turns} 轮）"
        )

    # ---------- CLI ----------

    HELP = """可用命令：
  /help            显示帮助
  /stats           查看各层记忆数量
  /memories        列出所有记忆（含层级与强度）；/memories <关键词> 按内容搜索
  /sleep           手动触发睡眠巩固（压缩低频记忆 + 情景记忆语义化迁移）
  /mind            心游：无查询时按强度加权自发想起一条记忆（再激活测试效应）
  /scene <查询>    场景重建：把相关记忆片段拼成连贯场景（片段组合回忆）
  /recall <id>     唤醒一条 Cold 摘要记忆
  /forget <id>     彻底删除一条记忆
  /plot            导出强度曲线（.svg 主图 + 按类型面板 + .csv + .json）并打印贴合度报告
  /ploti           导出多视图仪表盘（单文件 HTML：曲线/气泡/分布/Top列表/类型对比联动）
  /observe         观测一轮（所有记忆采样）并打印当前贴合度
  /classify <文本>  用分类器（LLM 或关键词回退）识别记忆类型
  /persona          查看当前人设与演化档案（remember_setting 写入的设定记忆）
  /evolve           自主演化：反思记忆 + 联网查资料 → 提出并入库新设定
  /write [字数]     基于当前人设档案续写下一章（落盘 works/<书名>/chapters/）
  /web <查询>       联网搜索（Bing → DuckDuckGo 备用）
  /models           查看 LLM 模型池状态（429 自动切换次数/最近限流）
  /learn            根据观测自动校准各类型 τ 与再巩固因子（参数自适应学习器）
  /tauplot          导出 learn_tau 两路信号（干净段/唤醒偏差）的收敛轨迹图 + 轮次明细
  /types            查看记忆类型画像（各类型 τ / 再巩固因子 / 压缩阈值 / 唤醒信号 / τ 两路信号列）
  /signal [近N天]   唤醒信号漂移：对比最近 N 天与更早的方向一致性（默认 30 天）
  /save            持久化到磁盘
  /quit            退出
  其他输入         作为对话内容"""

    def cli_loop(self, prompt: str = "你> ") -> None:
        from .cli import enable_utf8

        enable_utf8()
        print("记忆 Agent 已启动（无 LLM 依赖，模板回复）。输入 /help 查看命令。")
        while True:
            try:
                text = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not text:
                continue
            if text == "/quit":
                self.save()
                print("记忆已保存，再见！")
                break
            if text == "/help":
                print(self.HELP)
                continue
            if text == "/stats":
                print(self.stats())
                continue
            if text.startswith("/memories"):
                parts = text.split(maxsplit=1)
                self._print_memories(parts[1] if len(parts) == 2 else None)
                continue
            if text == "/sleep":
                report = self.sleep()
                print(f"睡眠巩固完成：重放 {report['replayed_count']} 条白天经历（再激活），"
                      f"Hot 降级 {len(report['hot_demoted'])} 条，"
                      f"压缩 {report['cold_compressed']} 条记忆为 {report['clusters']} 条 Cold 摘要，"
                      f"类型迁移 {report['migrations']} 条")
                continue
            if text == "/mind":
                m = self.spontaneous_recall()
                if m is None:
                    print("（记忆库为空，无事可想起）")
                else:
                    print(f"心游：突然想起「{m.content}」（次数={m.access_count} "
                          f"强度={self._strength(m):.2f}——再激活测试效应）")
                continue
            if text.startswith("/scene"):
                parts = text.split(maxsplit=1)
                if len(parts) == 2:
                    scene = self.compose_scene(parts[1])
                    if scene is None:
                        print(f"「{parts[1]}」未能重建出场景（相关记忆不足 2 条或开关关闭）")
                    else:
                        print(format_scene(scene))
                else:
                    print("用法：/scene <查询> —— 把相关记忆片段拼成连贯场景")
                continue
            if text == "/plot":
                files = self.plot_curves()
                print("已导出：" + "，".join(files))
                print(self.format_fit())
                continue
            if text == "/observe":
                n = self._observe()
                print(f"已观测一轮（新采样 {n} 条）")
                print(self.format_fit())
                continue
            if text == "/ploti":
                p = self.plot_interactive()
                print(f"已导出交互式曲线：{p}（浏览器打开，支持缩放/点击高亮/层级切换）")
                continue
            if text.startswith("/classify"):
                parts = text.split(maxsplit=1)
                if len(parts) == 2:
                    mt, conf, src = self.classify_text(parts[1])
                    print(f"「{parts[1]}」→ {mt.value}（置信 {conf:.2f}，来源 {src}）")
                else:
                    print("用法：/classify <文本>")
                continue
            if text == "/types":
                print(self.profile_table())
                continue
            if text.startswith("/web"):
                parts = text.split(maxsplit=1)
                if len(parts) == 2:
                    from .websearch import search_web

                    try:
                        results = search_web(parts[1], n=5)
                    except Exception as e:
                        # 网络异常不冲出 REPL（否则未 save 直接退出）
                        print(f"（联网搜索失败：{str(e)[:80]}）")
                        continue
                    if not results:
                        print("（联网搜索无结果或不可用）")
                    for r in results[:5]:
                        print(f"  • {r['title']}")
                        print(f"    {r['url']}")
                        if r.get("snippet"):
                            print(f"    {r['snippet'][:140]}")
                else:
                    print("用法：/web <查询> —— 联网搜索（Bing → DuckDuckGo）")
                continue
            if text.startswith("/write"):
                parts = text.split(maxsplit=1)
                try:
                    words = int(parts[1]) if len(parts) == 2 else None
                except ValueError:
                    words = None
                print("写作中（基于当前人设档案续写下一章）…")
                w = self.write_chapter(target_words=words)
                if not w.get("ok"):
                    print(f"  未完成：{w.get('reason')}")
                else:
                    print(f"  《{w['title']}》第 {w['chapter']} 章已写完（{w['words']} 字）")
                    print(f"  存档: {w['path']}")
                    print(f"  开头: {w['preview']}…")
                continue
            if text == "/evolve":
                print("自主演化中（反思记忆 + 联网查资料 → 沉淀新设定）…")
                ev = self.evolve()
                if not ev.get("ok"):
                    print(f"  未执行：{ev.get('reason')}（配 OPENAI_API_KEY 后可用）")
                else:
                    print(f"  研究主题: {ev['query']}（联网资料 {ev['web_n']} 条）")
                    if ev["added"]:
                        print(f"  新增设定 {len(ev['added'])} 条:")
                        for s in ev["added"]:
                            print(f"    • {s}")
                    else:
                        print("  没有可吸收的新设定（与已有设定重复或 LLM 未产出）")
                    print(f"  当前档案共 {len([m for m in self.store.all() if m.kind == 'setting'])} 条设定")
                continue
            if text == "/persona":
                if self.persona:
                    print(f"\n--- 人设（{self.persona[:24]}{'…' if len(self.persona) > 24 else ''}）---")
                else:
                    print("\n--- 人设 ---（未设置，可用 MemoryAgent(persona='novelist') 或 OPENAI_PERSONA 环境变量）")
                sheet = self.persona_sheet()
                if sheet:
                    print("你的身份档案（随设定记忆自主演化）:")
                    for line in sheet.splitlines():
                        print(f"  {line}")
                else:
                    print("  （还没有设定记忆——用 remember_setting() 写入作品设定后，") 
                    print("    这里会随记忆累积出可注入的人设档案）")
                continue
            if text == "/models":
                print("\n--- LLM 模型池（429 自动切换）---")
                if self.responder is not None and self.responder.available:
                    st = self.responder.pool_status()
                    print(f"  当前模型: {st['active']}")
                    print(f"  模型池: {', '.join(st['pool'])}")
                    print(f"  429 切换次数: {st['failover_count']}")
                    if st["recent_429"]:
                        for m, t in st["recent_429"]:
                            print(f"    限流: {m} @ {time.strftime('%H:%M:%S', time.localtime(t))}")
                    else:
                        print("  （无 429 记录）")
                else:
                    print("  未配置 LLM 回复生成器（OPENAI_API_KEY）——回复走内置模板")
                continue
            if text.startswith("/signal"):
                parts = text.split(maxsplit=1)
                days = 30.0
                if len(parts) == 2:
                    try:
                        days = float(parts[1])
                    except ValueError:
                        print(f"用法：/signal [近N天]（默认 30），「{parts[1]}」不是数字")
                        continue
                print(self.signal_drift_table(days))
                continue
            if text == "/tauplot":
                files = self.plot_tau_convergence()
                print("已导出 τ 收敛轨迹：" + "，".join(files)
                      + "（两路信号按轮次逼近真实 τ，比值趋 1 = 已校准）")
                continue
            if text == "/learn":
                from .visualize import fmt_duration

                trep = self.learn_tau()
                if trep["updated"]:
                    for u in trep["updated"]:
                        print(
                            f"  τ {u['type']}: {fmt_duration(u['old_tau'])} → "
                            f"{fmt_duration(u['new_tau'])}（实测 {fmt_duration(u['tau_est'])}，"
                            f"置信 {u['confidence']:.2f}）"
                        )
                else:
                    why = "；".join(f"{s['type']}: {s['reason']}" for s in trep["skipped"]) or "各类型已稳定"
                    print(f"τ 学习：无参数更新（{why}）")
                prep = self.learn_plasticity()
                if prep["updated"]:
                    for u in prep["updated"]:
                        print(
                            f"  因子 {u['type']}.{u['channel']}: {u['old']:.2f} → {u['new']:.2f}"
                            f"（实测 {u['est']:.2f}，置信 {u['confidence']:.2f}）"
                        )
                else:
                    why = "；".join(f"{s['type']}.{s['channel']}: {s['reason']}" for s in prep["skipped"]) or "各类型已稳定"
                    print(f"因子学习：无参数更新（{why}）")
                continue
            if text.startswith("/recall"):
                parts = text.split()
                if len(parts) == 2 and self.recall(parts[1]):
                    print(f"已唤醒记忆 {parts[1]}，详情可 /memories 查看")
                else:
                    print("未找到该 Cold 记忆（用 /memories 查看 ID）")
                continue
            if text.startswith("/forget"):
                parts = text.split()
                if len(parts) == 2 and self.store.remove(parts[1]):
                    print(f"已删除记忆 {parts[1]}")
                else:
                    print("未找到该记忆")
                continue
            if text == "/save":
                self.save()
                print("已保存。")
                continue

            reply, hits = self.respond(text)
            print(f"Agent> {reply}")
            if hits:
                print(f"  （检索到 {len([h for h in hits if h.total > 0.05])} 条相关记忆）")
            if self.last_reconsolidated:
                print(f"  （回忆再巩固：按重要程度微调了 {len(self.last_reconsolidated)} 条记忆）")

    def find_memories(self, text: str) -> list[Memory]:
        """按内容搜索记忆：content / 摘要 / 原始内容包含全部关键词（不区分大小写，
        空格分隔多词 = 同时包含），返回按重要性排序的记忆列表。

        用于长会话里快速定位一条特定记忆（CLI：/memories <关键词>）。
        """
        words = [w.lower() for w in text.split() if w]
        if not words:
            return []
        hits: list[Memory] = []
        for m in self.store.all():
            parts = [m.content, m.summary or ""]
            parts.extend(m.originals.values())  # Cold 摘要的原始内容
            hay = " ".join(parts).lower()
            if all(w in hay for w in words):
                hits.append(m)
        hits.sort(key=lambda m: (-m.importance, -m.last_access))
        return hits

    def _print_memories(self, filter_text: str | None = None) -> None:
        now = self._now()
        if filter_text:
            mems = self.find_memories(filter_text)
            print(f"匹配「{filter_text}」的记忆 {len(mems)} 条：")
            if not mems:
                print("  （无匹配）")
                return
        else:
            mems = sorted(self.store.all(), key=lambda m: (-m.importance, -m.last_access))
        for mem in mems:
            strength = self._strength(mem)
            star = "★" * max(1, round(mem.importance * 4))
            if mem.importance >= self.cfg.freeze_importance:
                badge = "冻结"
            elif now <= mem.labile_until:
                badge = "可塑"
            else:
                badge = "稳定"
            conf = f" 置信={mem.mtype_confidence:.2f}" if mem.mtype_confidence is not None else ""
            mig = f" 迁移={len(mem.migrations)}" if mem.migrations else ""
            chk = f" 校验={len(mem.checks)}" if mem.checks else ""
            # 唤醒自 Cold 的标记：修订/历史都是继承自 Cold 的（长生命周期可追溯）；
            # 普通记忆保持原样只显示修订计数。
            if mem.awakened_at is not None:
                rev = f"唤醒自Cold(修订={mem.revision_count} 历史={len(mem.history)})"
            else:
                rev = f"修订={mem.revision_count}"
            print(
                f"[{mem.tier.value:4}] [{mem.mtype.value:8}] {mem.id} 强度={strength:.2f} "
                f"次数={mem.access_count} 重要={mem.importance:.2f} {rev}"
                f"{conf}{mig}{chk} [{badge}] {star} {mem.summary or mem.content[:40]}"
            )
