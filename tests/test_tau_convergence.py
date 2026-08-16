"""learn_tau 两路信号收敛轨迹测试：干净段反推 vs 唤醒偏差代理按轮次逼近真实 τ。

设计验证：真实 τ=2 天、模型信念 3 天的双源场景下，学习历史每轮记录两路的
独立估计（clean_est / aw_est）与唤醒中位比值（dev/expected）；收敛轨迹图
展示配置 τ 的 EMA 轨迹、两路 τ_est、真实 τ 参考线与比值趋 1 的下子图——
两条 τ_est 都向真实 τ 收敛 = 两路信号互相印证。
"""

import csv

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType
from memagent.visualize import (
    export_tau_trajectory,
    render_tau_convergence,
    tau_rounds,
)

TRUE_DAYS = 2 * 86400.0
INIT_DAYS = 3 * 86400.0


def _two_source_agent(n_rounds: int = 6, rate: float = 0.3):
    """双源场景：一条只衰减的记忆（干净段源）+ 一条 Cold↔Warm 往返（唤醒源）。"""
    clock = [1000.0]
    cfg = AgentConfig(
        tau_by_type={MemType.EPISODIC: INIT_DAYS},
        true_tau_by_type={MemType.EPISODIC: TRUE_DAYS},
        tau_learning_rate=rate,
    )
    a = MemoryAgent(cfg=cfg, now_fn=lambda: clock[0])
    a.store.add("用户说：我昨天去看了场电影", importance=0.1,
                mtype=MemType.EPISODIC, now=clock[0])
    aw = a.store.add("我昨天去吃了火锅", importance=0.3,
                     mtype=MemType.EPISODIC, now=clock[0])
    aw.access_count = 2
    for _ in range(n_rounds):
        clock[0] += 6 * 3600
        a._observe()                       # 干净衰减段采样
        clock[0] += 1.1 * 3 * 86400
        aw.demote_to_cold("火锅聚餐（已归档）")
        aw = a.recall(aw.id[:6])           # 唤醒观测
        a.learn_tau(force=True)
    return a, aw


def test_learn_history_records_both_sources():
    """学习历史每轮带两路独立估计：干净段 τ_est / 唤醒 τ_est / 唤醒中位比值，
    以及本次更新实际使用的唤醒信号原始值（dev / expected——方向可复盘）。"""
    a, _ = _two_source_agent(n_rounds=6)
    assert a._learn_history, "应产生学习更新"
    any_both = any(len(row) >= 9 and row[6] is not None and row[7] is not None
                   for row in a._learn_history)
    assert any_both, "至少一轮两路源同时活跃（干净段 + 唤醒）"
    for row in a._learn_history:
        assert row[5] is not None and 0.0 <= row[5] <= 1.0   # 置信度
        assert row[8] is None or row[8] > 0                  # 比值 > 0
        # 11 列历史：10/11 列为唤醒中位 dev / expected（本次更新方向依据）
        if len(row) >= 11 and row[9] is not None:
            assert row[10] is not None and row[10] > 0
            assert row[9] > row[10]          # 双源场景真实 τ 更小 → 埋得比信念深
            # dev 与 expected 是中位数（独立取中位），比值由 tau_rounds 单独取中位
            assert row[9] / row[10] > 1.0    # 方向一致：dev/expected > 1
    # 至少一轮把唤醒信号原始值写进历史（可复盘方向）
    assert any(len(row) >= 11 and row[9] is not None for row in a._learn_history)


def test_convergence_both_sources_corroborate():
    """收敛验证：配置 τ 的 EMA 轨迹向真实 τ（2 天）移动，两路估计都在场，
    唤醒比值 > 1（τ 失准方向）且随轮次下降逼近 1。"""
    a, _ = _two_source_agent(n_rounds=10, rate=0.3)
    final = a.cfg.tau_for(MemType.EPISODIC)
    assert final < INIT_DAYS                       # 从 3 天下调
    assert abs(final - TRUE_DAYS) < abs(INIT_DAYS - TRUE_DAYS)  # 更接近真实
    ratios = [row[8] for row in a._learn_history if row[8] is not None]
    assert len(ratios) >= 2
    assert all(r > 1.0 for r in ratios)            # 埋得比预期深 → τ 应下调
    assert ratios[-1] < ratios[0]                  # 比值随校准下降 → 趋 1
    # 两路估计互相印证：任一有更新的轮次里两者都应存在（双源场景）
    both = [row for row in a._learn_history if row[6] is not None and row[7] is not None]
    assert both


def test_tau_rounds_and_csv_export(tmp_path):
    """tau_rounds 提取 + CSV 导出：每行一轮，含两路独立估计与比值列。"""
    a, _ = _two_source_agent(n_rounds=6)
    rs = tau_rounds(a)
    assert rs and rs[0]["mtype"] == "episodic"
    assert any(r["clean_est"] is not None and r["aw_est"] is not None for r in rs)
    out = export_tau_trajectory(a, str(tmp_path / "t.csv"))
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert len(rows) == len(rs)
    assert rows[0]["mtype"] == "episodic"
    assert rows[0]["old_tau_seconds"] and rows[0]["new_tau_seconds"]
    assert rows[0]["clean_est_seconds"] and rows[0]["awakening_est_seconds"]
    assert rows[0]["awakening_ratio"]
    # 唤醒信号原始值列：dev / expected（本次更新方向可复盘）
    assert rows[0]["awakening_dev"] and rows[0]["awakening_expected"]
    assert float(rows[0]["awakening_dev"]) > float(rows[0]["awakening_expected"])


def test_tau_rounds_old_format_backward_compatible(tmp_path):
    """旧 6 列学习历史（无独立源列）→ 源列为 None，渲染不崩溃。"""
    a = MemoryAgent()
    a._learn_history.append([100.0, "episodic", 3.0, 2.5, 2.8, 0.5])
    rs = tau_rounds(a)
    assert rs[0]["clean_est"] is None and rs[0]["aw_est"] is None
    assert rs[0]["ratio"] is None
    out = render_tau_convergence(a, str(tmp_path / "t.svg"))
    svg = open(out, encoding="utf-8").read()
    assert "episodic" in svg and "学习轮次" in svg


def test_render_convergence_svg_elements(tmp_path):
    """SVG 含两路源线、真实 τ 参考线、比值面板与轮次轴。"""
    a, _ = _two_source_agent(n_rounds=6)
    out = render_tau_convergence(a, str(tmp_path / "t.svg"))
    svg = open(out, encoding="utf-8").read()
    for k in ["干净段τ_est", "唤醒τ_est", "真实 τ", "比值=1", "学习轮次",
              "#7048e8", "#f08c00", "#8a8f98", "配置τ(EMA)"]:
        assert k in svg, f"缺少 {k}"


def test_render_no_history_placeholder(tmp_path):
    """无学习轮次 → 占位提示而非空图。"""
    a = MemoryAgent()
    out = render_tau_convergence(a, str(tmp_path / "t.svg"))
    assert "尚无学习轮次" in open(out, encoding="utf-8").read()


def test_plot_tau_convergence_files(tmp_path):
    """agent.plot_tau_convergence() 导出 SVG + CSV 两个文件。"""
    a, _ = _two_source_agent(n_rounds=6)
    files = a.plot_tau_convergence(str(tmp_path / "tc"))
    assert len(files) == 2
    assert files[0].endswith(".svg") and files[1].endswith(".csv")
