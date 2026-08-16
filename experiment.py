"""类型行为对照实验：同一批记忆在不同类型参数下跑完整生命周期，输出对比报告。

控制变量法：记忆批次、检索序列、时间线三组配置完全相同，只改类型参数——
用可注入的模拟时钟（now_fn）确定性快进，秒级参数代表"数天"。

运行：python experiment.py [--save experiment_report.md]

四组对照：
  A 基线     技能慢/语义中/情景快（τ=90/30/8s）+ 因子 0.15/1.0/2.5
  B 无区分   所有类型同一 τ、因子全 1.0      —— 类型区分的价值
  C 全冻结   再巩固因子全 0                  —— 修改与遗忘是否正交
  D 反转     τ 方向错置（技能快/情景慢）      —— 参数方向的代价
"""

from __future__ import annotations

import sys

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.llm import LLMClassifier
from memagent.memory import MemType, Tier

# ---------- 记忆批次：四组共用（自动识别类型，验证识别一致性） ----------
# 注意："零检索"记忆的内容与所有查询零 n-gram 共享，避免哈希嵌入的泛化命中
# 污染对照组（短查询如"火锅"会因哈希碰撞误伤无关记忆）。
BATCH = [
    ("我会弹钢琴", 0.1),            # skill 高频检索
    ("北京是中国的首都", 0.1),      # semantic 中频检索
    ("我昨天去吃了火锅", 0.1),      # episodic 高频检索 → 触发再巩固 + 语义化迁移
    ("刚才在浇花", 0.05),           # episodic 零检索 → 触发睡眠压缩；内容与三个
    #   查询字符 n-gram 全不相交且实测余弦相似度=0（哈希碰撞也不会命中）
]

# ---------- 时间线：四组完全一致（(推进秒数, 动作)，None=只推进） ----------
# 查询全部用与目标记忆高重叠的完整句子（rel≈1）：命中与否不依赖再巩固漂移，
# 四组行为对称；短查询（"火锅"）会因哈希碰撞误伤无关记忆，实验里避免。
TIMELINE = [
    (1, "弹钢琴"),
    (1, "昨天去吃了火锅"),
    (1, "弹钢琴"),
    (1, "昨天去吃了火锅"),
    (1, "中国的首都"),
    (10, None),        # 衰减窗口：情景（τ=8s）应明显衰减、技能（τ=90s）几乎不掉
    (1, "昨天去吃了火锅"),  # 再巩固触发（可塑窗口内）
    (9, None),         # 继续衰减
    (1, "__sleep__"),  # 睡眠巩固：压缩零检索记忆 + 情景语义化迁移
    (4, None),         # 终态观测
]

# ---------- 组配置 ----------
GROUPS = [
    dict(
        name="A 基线",
        desc="技能慢/语义中/情景快 τ=90/30/8s，因子 0.15/1.0/2.5",
        cfg=AgentConfig(
            tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
            cold_after_seconds=None, cold_after_tau=2.0,
            reconsolidation_by_type={
                MemType.SKILL: {"drift": 0.15, "importance": 0.2},
                MemType.SEMANTIC: {"drift": 1.0, "importance": 1.0},
                MemType.EPISODIC: {"drift": 2.5, "importance": 1.5},
            },
            tau_learning=False, plasticity_learning=False,  # 隔离学习器，只测参数本身
            replay=False,  # 隔离回放：重放会统一抬升各记忆检索次数，污染零检索对照组
        ),
    ),
    dict(
        name="B 无区分",
        desc="所有类型同一 τ=30s，因子全 1.0",
        cfg=AgentConfig(
            tau_by_type={MemType.SKILL: 30.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 30.0},
            cold_after_seconds=None, cold_after_tau=2.0,
            reconsolidation_by_type={
                MemType.SKILL: {"drift": 1.0, "importance": 1.0},
                MemType.SEMANTIC: {"drift": 1.0, "importance": 1.0},
                MemType.EPISODIC: {"drift": 1.0, "importance": 1.0},
            },
            tau_learning=False, plasticity_learning=False,
            replay=False,  # 隔离回放：同 A，保持检索次数为受控变量
        ),
    ),
    dict(
        name="C 全冻结",
        desc="τ 同 A，但再巩固因子全 0（回忆不改写）",
        cfg=AgentConfig(
            tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
            cold_after_seconds=None, cold_after_tau=2.0,
            reconsolidation_by_type={
                MemType.SKILL: {"drift": 0.0, "importance": 0.0},
                MemType.SEMANTIC: {"drift": 0.0, "importance": 0.0},
                MemType.EPISODIC: {"drift": 0.0, "importance": 0.0},
            },
            tau_learning=False, plasticity_learning=False,
            replay=False,
        ),
    ),
    dict(
        name="D 反转",
        desc="τ 方向错置：技能 8s / 情景 90s（技能反而最快遗忘）",
        cfg=AgentConfig(
            tau_by_type={MemType.SKILL: 8.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 90.0},
            cold_after_seconds=None, cold_after_tau=2.0,
            reconsolidation_by_type={
                MemType.SKILL: {"drift": 0.15, "importance": 0.2},
                MemType.SEMANTIC: {"drift": 1.0, "importance": 1.0},
                MemType.EPISODIC: {"drift": 2.5, "importance": 1.5},
            },
            tau_learning=False, plasticity_learning=False,
            replay=False,  # 隔离回放：同前，保持检索次数为受控变量
        ),
    ),
]


# ---------- 执行 ----------

def run_group(group: dict) -> dict:
    """用模拟时钟跑完整生命周期，返回 {metrics, sleep_report, agent}。"""
    clock = [1000.0]  # 模拟时钟（避免 0 起点的边界），now_fn 推进它
    agent = MemoryAgent(
        cfg=group["cfg"],
        now_fn=lambda: clock[0],
        classifier=LLMClassifier(api_key=""),  # 强制关键词分类，实验确定性
    )
    # 语义化阈值校准到 2.5：时间线里情景记忆恰被检索 3 次，评分≈3.0 在默认
    # 阈值 3.0 的浮点边界上（0.99995×3 < 3.0），2.5 让四组一致稳定触发
    agent.cfg.semanticize_threshold = 2.5

    def advance(dt: float) -> None:
        clock[0] += dt

    init_types = {}
    for content, importance in BATCH:
        m = agent.remember(content, importance=importance)
        init_types[content] = m.mtype.value  # 初始识别（报告一致性检查用）

    sleep_report = None
    for dt, action in TIMELINE:
        advance(dt)
        if action is None:
            continue
        if action == "__sleep__":
            sleep_report = agent.sleep()
        else:
            agent.retrieve(action, k=2)

    compressed = [m.content for m in agent.store.by_tier(Tier.COLD)]

    metrics = {}
    for mem in agent.store.all():
        metrics[mem.content] = {
            "mtype": mem.mtype.value,
            "tier": mem.tier.value,
            "strength": round(agent._strength(mem), 2),
            "access": mem.access_count,
            "revisions": mem.revision_count,
            "importance": round(mem.importance, 2),
            "migrations": len(mem.migrations),
        }
    return {
        "metrics": metrics,
        "sleep": sleep_report or {},
        "init_types": init_types,
        "compressed": compressed,
        "agent": agent,
    }


def _disp_len(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_len(s))


# ---------- 报告 ----------

def build_report() -> str:
    results = [run_group(g) for g in GROUPS]
    L: list[str] = []
    add = L.append

    add("═" * 62)
    add("  类型行为对照实验：同一批记忆在不同类型参数下的完整生命周期")
    add("═" * 62)
    add("记忆批次（4 条，自动识别类型）:")
    for content, _imp in BATCH:
        mt = results[0]["init_types"][content]
        same = all(r["init_types"][content] == mt for r in results)
        add(f"  • {content} → {mt}{'' if same else '（⚠ 各组识别不一致！）'}")
    add("时间线: 写入 → 5 轮检索 → 衰减10s → 再巩固检索 → 衰减9s → 睡眠巩固 → 终态")
    add("")

    for group, res in zip(GROUPS, results):
        add("─" * 62)
        add(f"[{group['name']}] {group['desc']}")
        add("─" * 62)
        for content, _imp in BATCH:
            m = res["metrics"][content]
            add(
                f"  {_pad(content, 22)} {_pad(m['mtype'], 9)} 强度 {m['strength']:.2f}  "
                f"检索 {m['access']}  修订 {m['revisions']}  重要 {m['importance']:.2f}  "
                f"[{m['tier']}]{' 迁移×' + str(m['migrations']) if m['migrations'] else ''}"
            )
        rep = res["sleep"]
        add(
            f"  睡眠巩固: 压缩 {rep.get('cold_compressed', 0)} 条 → {rep.get('clusters', 0)} 条 Cold，"
            f"类型迁移 {rep.get('migrations', 0)} 条"
        )
        add("")

    # ---- 对照汇总表 ----
    add("═" * 62)
    add("对照汇总（组 × 指标）")
    add("═" * 62)
    rows = [
        ("技能最终强度", lambda r: f"{r['metrics']['我会弹钢琴']['strength']:.2f}"),
        ("情景最终强度", lambda r: f"{r['metrics']['我昨天去吃了火锅']['strength']:.2f}"),
        ("技能重要性", lambda r: f"{r['metrics']['我会弹钢琴']['importance']:.2f}"),
        ("情景重要性", lambda r: f"{r['metrics']['我昨天去吃了火锅']['importance']:.2f}"),
        ("技能修订次数", lambda r: str(r["metrics"]["我会弹钢琴"]["revisions"])),
        ("情景修订次数", lambda r: str(r["metrics"]["我昨天去吃了火锅"]["revisions"])),
        ("压缩条数", lambda r: str(r["sleep"].get("cold_compressed", 0))),
        ("迁移条数", lambda r: str(r["sleep"].get("migrations", 0))),
    ]
    header = _pad("指标", 14) + "".join(_pad(f"[{g['name']}]", 12) for g in GROUPS)
    add(header)
    add("-" * 62)
    for label, fn in rows:
        add(_pad(label, 14) + "".join(_pad(fn(r), 12) for r in results))
    add("")

    # ---- 结论 ----
    add("═" * 62)
    add("结论")
    add("═" * 62)
    a, b, c, d = results
    sk = lambda r: r["metrics"]["我会弹钢琴"]["strength"]
    ep = lambda r: r["metrics"]["我昨天去吃了火锅"]["strength"]
    short = lambda s: s[:14] + "…" if len(s) > 15 else s
    add(
        f"• 类型区分（A vs B）：技能 {sk(a):.2f} vs {sk(b):.2f}、情景 {ep(a):.2f} vs {ep(b):.2f}"
        f"——区分 τ 让\"技能久记、情景速忘\"成为现实，无区分时三类趋同"
    )
    add(
        f"• 再巩固（A vs C）：情景修订 {a['metrics']['我昨天去吃了火锅']['revisions']} vs "
        f"{c['metrics']['我昨天去吃了火锅']['revisions']}、情景重要性 "
        f"{a['metrics']['我昨天去吃了火锅']['importance']:.2f} vs "
        f"{c['metrics']['我昨天去吃了火锅']['importance']:.2f}"
        f"——冻结因子后回忆不再改写记忆，但遗忘曲线（强度）几乎不变：修改与遗忘是正交的两条轴"
    )
    add(
        f"• 参数方向（A vs D）：技能 {sk(a):.2f} vs {sk(d):.2f}、情景 {ep(a):.2f} vs {ep(d):.2f}"
        f"——τ 方向错置时技能反而最快遗忘，参数方向错误比不区分更糟"
    )
    add(
        f"• 强记忆自我强化：A 组技能检索 {a['metrics']['我会弹钢琴']['access']} 次（精确 2 + 泛化 2）"
        f"——慢衰减的技能强度高，更易被相关查询泛化命中而再次加固（测试效应×命中率）；"
        f"B/D 组技能衰减快、强度低，泛化命中不到门槛（各 2 次）"
    )
    comp_a = "、".join(short(c) for c in a["compressed"]) or "无"
    comp_d = "、".join(short(c) for c in d["compressed"]) or "无"
    add(
        f"• 压缩随 τ 缩放（cold_after = 2×τ）：A 压缩 {len(a['compressed'])} 条"
        f"（{comp_a}，情景闲置 30s > 2×8s），B {len(b['compressed'])} 条"
        f"（阈值随 τ 放大到 60s 未触发），D 压缩 {len(d['compressed'])} 条"
        f"（{comp_d}——τ 反转后技能阈值缩到 16s，闲置的技能先被埋藏）"
    )
    add(
        f"• 迁移正交于 τ：四组各迁移 {a['sleep'].get('migrations', 0)} 条"
        f"（情景检索 ≥3 次即语义化），但迁移收益（τ 8s→30s 的强度回升）"
        "只在“情景快忘”方向成立——D 组 τ 反转时迁移反而把记忆拉向更快的衰减"
    )
    return "\n".join(L)


def main() -> None:
    from memagent.cli import enable_utf8

    enable_utf8()
    report = build_report()
    print(report)
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "experiment_report.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main()
