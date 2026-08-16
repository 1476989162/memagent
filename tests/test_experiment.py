"""类型行为对照实验测试：时钟注入行为 + 实验可运行 + 组间差异断言。"""

import experiment
from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType


def test_now_fn_controls_decay():
    """注入时钟：快进后强度按 τ 衰减（默认时钟不受影响）。"""
    clock = [100.0]
    agent = MemoryAgent(
        cfg=AgentConfig(tau_by_type={MemType.SEMANTIC: 10.0}),
        now_fn=lambda: clock[0],
    )
    m = agent.remember("北京是中国的首都", importance=0.1)
    s0 = agent._strength(m)
    clock[0] += 10.0  # 快进一个 τ → 时效项衰减到 e^-1
    s1 = agent._strength(m)
    assert s0 == s0  # sanity
    assert s1 < s0
    # 默认时钟不受污染：普通 agent 写入即测，强度与注入时钟的初始强度一致
    # （强度公式 = 时效×freq×importance 归一化，刚写入不是 1 而是固定常数）
    normal = MemoryAgent(cfg=AgentConfig(tau_by_type={MemType.SEMANTIC: 10.0}))
    n = normal.remember("北京是中国的首都", importance=0.1)
    assert abs(normal._strength(n) - s0) < 0.05


def test_now_fn_tracks_last_access_on_retrieve():
    clock = [100.0]
    agent = MemoryAgent(
        cfg=AgentConfig(tau_by_type={MemType.SKILL: 90.0}),
        now_fn=lambda: clock[0],
    )
    m = agent.remember("我会弹钢琴", importance=0.1)
    clock[0] += 50.0
    agent.retrieve("弹钢琴", k=1)
    assert m.last_access == 150.0          # touch 用注入时钟
    assert m.access_count == 1


def test_experiment_runs_and_report_is_well_formed():
    report = experiment.build_report()
    for key in ("对照实验", "对照汇总", "结论", "A 基线", "B 无区分", "C 全冻结", "D 反转",
                "技能最终强度", "情景修订次数", "压缩条数", "迁移条数"):
        assert key in report


def test_identification_consistent_across_groups():
    results = [experiment.run_group(g) for g in experiment.GROUPS]
    for content, _imp in experiment.BATCH:
        types = {r["init_types"][content] for r in results}
        assert len(types) == 1, f"{content} 在各组识别不一致: {types}"
    assert results[0]["init_types"]["我会弹钢琴"] == "skill"
    assert results[0]["init_types"]["北京是中国的首都"] == "semantic"


def test_zero_access_control_group_clean():
    """零检索记忆在四组都保持 access 0（对照组未被泛化命中污染）。"""
    results = [experiment.run_group(g) for g in experiment.GROUPS]
    for r in results:
        assert r["metrics"]["刚才在浇花"]["access"] == 0


def test_group_differences_reflect_parameters():
    results = [experiment.run_group(g) for g in experiment.GROUPS]
    a, b, c, d = results
    sk = lambda r: r["metrics"]["我会弹钢琴"]["strength"]
    ep = lambda r: r["metrics"]["我昨天去吃了火锅"]
    # 类型区分：A（技能慢衰减）技能强度 > B（无区分）
    assert sk(a) > sk(b)
    # 再巩固：A 情景修订 > 0，C（全冻结）修订 0，且 C 重要性不被抬高
    assert ep(a)["revisions"] > 0
    assert ep(c)["revisions"] == 0
    assert ep(a)["importance"] > ep(c)["importance"]
    # 参数方向：A 技能 > D（τ 反转，技能最快遗忘）
    assert sk(a) > sk(d)
    # 压缩随 τ 缩放：A（情景 τ=8s → 阈值 16s）压缩零检索情景，B（τ=30s → 60s）不压缩
    assert len(a["compressed"]) == 1
    assert len(b["compressed"]) == 0
    # 迁移：情景被检索 ≥3 次，四组各迁移 1 条
    assert all(r["sleep"].get("migrations", 0) == 1 for r in results)


def test_sleep_report_contains_compression():
    results = [experiment.run_group(g) for g in experiment.GROUPS]
    assert results[0]["sleep"]["cold_compressed"] == 1
    assert results[0]["compressed"][0] == "刚才在浇花"
