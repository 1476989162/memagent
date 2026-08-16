"""τ↔可塑性联合估计器测试：一次唤醒事件同时更新 τ 与 drift（跨轮耦合）。

观测层可塑性调制（_observe_awakening）：
    dev      = base(τ真实) · (1 + g·(p实测 − 1))
    expected = base(τ模型) · (1 + g·(p信念 − 1))
——纯 τ 失准时两刻度相同（p实测==p信念）→ 比值纯 τ（旧行为逐位不变）；纯可塑性
失准时 dev ≠ expected（旧系统该场景 dev==expected 完全无信号）——调制让
"dev 同时编码 τ 失准与可塑性"成立。

归因（单事件两个未知不可完全分离 → 顺序归因 + 跨轮耦合，见 _joint_awakening_estimates）：
- τ 通道先拿比值（埋藏深度主导），剥掉上一轮估计的可塑性因子再反演（纯可塑性
  失准不被误读为 τ 失准）；
- drift 通道拿 τ 解释不了的残余——用**上一轮**事件的精确 τ 反演（按衰减公式对
  埋藏时长重算）校正：纯 τ 失准 + τ 参考收敛 → 残余为零 → drift 不动（消除旧
  双计数——旧代理把整个比值同时判给 τ 与 drift，会把纯 τ 误调成可塑性上调）；
- 两路跨轮状态互为输入 → 两路 EMA 相互加速（彼此的进展清洗对方的信号）。

识别边界（诚实记录）：单事件无法分离 τ 与可塑性——首轮无 τ 参考（drift 读全
比值，瞬态）；纯可塑性失准时 τ 参考的信念刻度会吸收一部分可塑性信号（该场景的
可塑性学习主要由再巩固修订日志承担）；两路同时失准（本估计器的目标场景）时
两路 EMA 在收敛相位互相加速逼近真实值。
"""

import math

from memagent.agent import AgentConfig, MemoryAgent
from memagent.memory import MemType

DAY = 24 * 3600
G = 0.3  # 默认 awakening_plasticity_gain


def _agent(clock, **kw) -> MemoryAgent:
    return MemoryAgent(cfg=AgentConfig(reconsolidate=False, **kw),
                       now_fn=lambda: clock[0])


def _awaken(a: MemoryAgent, clock: list, text: str = "我昨天去吃了火锅"):
    """一次 Cold→Warm 往返，产生一条 6 元组唤醒观测（按当前信念实时计算）。"""
    m = a.store.add(text, importance=0.3, mtype=MemType.EPISODIC)
    m.access_count = 2
    m.last_access = clock[0]
    clock[0] += 1.2 * 3 * DAY       # 闲置至衰减中段（floor 之上）
    m.demote_to_cold(text)
    return a.recall(m.id[:6])


# ---------- 观测层：dev 同时编码 τ 失准与可塑性 ----------

def test_observation_modulation_encodes_plasticity():
    """纯可塑性失准（true drift 3.5 vs 信念 2.5，τ 校准）：dev ≠ expected，
    比值 = 可塑性因子 (1+g(p实测−1))/(1+g(p信念−1)) = 1.75/1.45 ≈ 1.207——
    旧系统该场景 dev==expected 完全无信号，调制让"dev 编码可塑性"成立。"""
    clock = [0.0]
    a = _agent(clock, true_reconsolidation_by_type={
        MemType.EPISODIC: {"drift": 3.5, "importance": 1.5}})
    revived = _awaken(a, clock)
    _, dev, expected, _, dt, n_cold = revived.awakenings[-1]
    assert dev > expected                       # 唤醒比信念预期剧烈（可塑性更高）
    assert abs(dev / expected - 1.75 / 1.45) < 0.05   # 比值 ≈ 可塑性因子
    assert dt > 0 and n_cold == 2               # 6 元组带埋藏时长与检索次数


def test_observation_modulation_preserves_tau_only():
    """纯 τ 失准（true τ 2 天 vs 信念 3 天，无可塑性失配）：两刻度相同（S 抵消）
    → dev/expected 比值与旧观测完全一致（纯 τ 信号不被调制改变）——调制不改
    变 τ-only 场景的观测语义。自洽环境（无 true_*）才 dev == expected。"""
    clock = [0.0]
    a = _agent(clock, true_tau_by_type={MemType.EPISODIC: 2 * DAY})
    revived = _awaken(a, clock)
    _, dev, expected, _, _, _ = revived.awakenings[-1]
    assert dev > expected                      # 真实衰减更快 → 埋得更深
    # 与未调制观测一致：dev/expected = base(τ真实)/base(τ模型)（刻度抵消）
    tau = a.cfg.tau_for(MemType.EPISODIC)
    base_t = a.strength_at_state(revived.mtype, revived.last_access, revived.access_count,
                                 revived.importance, revived.last_access, tau_override=2 * DAY) \
        - a.strength_at_state(revived.mtype, 0.0, revived.access_count - 1,
                              revived.importance, revived.last_access, tau_override=2 * DAY)
    base_m = a.strength_at_state(revived.mtype, revived.last_access, revived.access_count,
                                 revived.importance, revived.last_access, tau_override=tau) \
        - a.strength_at_state(revived.mtype, 0.0, revived.access_count - 1,
                              revived.importance, revived.last_access, tau_override=tau)
    assert abs(dev / expected - base_t / base_m) < 1e-3   # 元组值取整到 4dp
    # 自洽环境（无 true_*）：dev == expected
    clock2 = [0.0]
    b = _agent(clock2)
    revived2 = _awaken(b, clock2)
    _, dev2, expected2, _, _, _ = revived2.awakenings[-1]
    assert dev2 == expected2


# ---------- 联合估计器：一次事件 → τ 与 drift 两路 ----------

def test_joint_tau_estimate_matches_legacy():
    """τ-only 场景：联合估计器的 τ 通道与旧独立路径（_tau_awakening_estimate）
    逐位一致（首轮无上一轮可塑性估计 → 剥因子退化为 1）——τ 行为向后兼容。"""
    clock = [0.0]
    a = _agent(clock, true_tau_by_type={MemType.EPISODIC: 2 * DAY})
    _awaken(a, clock)
    tau_ests, drift_ests, n, ratios = a._joint_awakening_estimates(MemType.EPISODIC)
    legacy_est, legacy_n, legacy_ratio = a._tau_awakening_estimate(MemType.EPISODIC)
    assert n == legacy_n == 1
    assert tau_ests[0] == legacy_est
    assert ratios == [legacy_ratio]
    # 首轮无 τ 参考 → drift 通道读全比值（瞬态，文档记录的识别边界）
    assert len(drift_ests) == 1 and drift_ests[0] > 2.5


def test_joint_tau_only_drift_silent_with_converged_tau_ref():
    """双计数消除（核心判别）：纯 τ 失准时，τ 参考一旦收敛（上一轮精确反演出
    真实 τ），drift 通道按衰减公式重算 → 残余为零 → drift 样本 == 信念（不动）。
    旧独立代理把整个比值同时判给 τ 与 drift——同样场景 drift 会被误调上调。"""
    clock = [0.0]
    a = _agent(clock, true_tau_by_type={MemType.EPISODIC: 2 * DAY})
    _awaken(a, clock)
    # 首轮扫描：τ 参考仍为信念 → 瞬态
    _, drift_r1, _, _ = a._joint_awakening_estimates(MemType.EPISODIC)
    # 第二轮扫描：τ 参考已收敛（第一轮精确反演出 2 天）→ 残余归零
    _, drift_r2, _, _ = a._joint_awakening_estimates(MemType.EPISODIC)
    p_b = a.cfg.reconsolidation_factor(MemType.EPISODIC, "drift")
    assert drift_r1[0] > p_b + 0.3                  # 首轮瞬态（全比值）
    assert abs(drift_r2[0] - p_b) < 0.05            # 收敛后静默（双计数消除）

    # 对照组：旧独立代理同场景误调（保留作回退的旧语义）
    a2 = _agent(clock, true_tau_by_type={MemType.EPISODIC: 2 * DAY})
    _awaken(a2, clock)
    legacy = a2._awakening_drift_estimate(
        MemType.EPISODIC.value, *[float(x) for x in (0.47, 0.3996)])
    assert legacy > p_b + 0.3                        # 旧代理误调上调


def test_joint_plasticity_only_drift_exact_first_round():
    """纯可塑性失准（τ 校准）：首轮（无 τ 参考 → 不校正）调制反演给出**精确**
    实测因子 p_est = 3.5 == p实测。τ 通道的可塑性剥因子让 τ 不被误调：首轮无
    上一轮估计（剥因子=1）τ 短暂误动，第二轮剥因子生效 → τ 回到信念。
    识别边界：后续轮 τ 参考的信念刻度会吸收部分可塑性信号（该场景可塑性学习
    主要由再巩固修订日志承担）。"""
    clock = [0.0]
    a = _agent(clock, true_reconsolidation_by_type={
        MemType.EPISODIC: {"drift": 3.5, "importance": 1.5}})
    _awaken(a, clock)
    tau_ests, drift_ests, _, _ = a._joint_awakening_estimates(MemType.EPISODIC)
    assert abs(drift_ests[0] - 3.5) < 0.05          # 精确反演（首轮全比值）
    tau_m = a.cfg.tau_for(MemType.EPISODIC)
    assert abs(tau_ests[0] - tau_m) / tau_m > 0.05  # 首轮无剥因子 → 短暂误动
    # 第二轮：可塑性估计已更新（p_ref=3.5）→ 剥因子生效 → τ 回到信念
    tau_ests2, drift_ests2, _, _ = a._joint_awakening_estimates(MemType.EPISODIC)
    assert abs(tau_ests2[0] - tau_m) / tau_m < 0.05


def test_joint_both_off_mutual_acceleration():
    """两路同时失准（联合估计器的目标场景）：多轮 (3 次唤醒 + learn_tau +
    learn_plasticity) 后 τ 与 drift 同时逼近真实值（3 天→2 天、2.5→3.5）——
    两路 EMA 相互加速：τ 的收敛（跨轮 τ 参考）清洗 drift 残余，drift 的收敛
    （跨轮可塑性估计）清洗 τ 的比值。"""
    clock = [0.0]
    a = _agent(clock,
               true_tau_by_type={MemType.EPISODIC: 2 * DAY},
               true_reconsolidation_by_type={
                   MemType.EPISODIC: {"drift": 3.5, "importance": 1.5}},
               tau_learning_rate=0.3, plasticity_learning_rate=0.3)
    for rnd in range(8):
        for _ in range(3):
            _awaken(a, clock)
        a.learn_tau(force=True)
        a.learn_plasticity(force=True)
    tau = a.cfg.tau_for(MemType.EPISODIC) / DAY
    drift = a.cfg.reconsolidation_factor(MemType.EPISODIC, "drift")
    assert 1.7 <= tau <= 2.6                # 3 天 → 逼近真实 2 天
    assert 2.7 <= drift <= 3.7              # 2.5 → 逼近真实 3.5
    # 两路都比起点更接近真实值
    assert abs(tau - 2.0) < 1.0 and abs(drift - 3.5) < 0.6


# ---------- 学习历史记录唤醒信号（dev/expected/ratio 可复盘） ----------

def test_learn_history_records_awakening_signal_columns():
    """learn_tau 历史 11 列：末尾两列 = 本次更新实际使用的唤醒中位 dev/expected
    （方向依据：dev > expected = 埋得比信念深 → 下调 τ），与比值列共同构成
    可复盘的信号快照；且随校准 dev/expected 比值单调趋 1。"""
    clock = [0.0]
    a = _agent(clock,
               true_tau_by_type={MemType.EPISODIC: 2 * DAY},
               tau_learning_rate=0.3)
    for _ in range(6):
        _awaken(a, clock)
        a.learn_tau(force=True)
    rows = a._learn_history
    assert rows and all(len(r) == 11 for r in rows)
    for r in rows:
        assert r[9] is not None and r[10] is not None     # dev / expected 已记录
        assert r[9] > r[10]                                # 真实 τ 更小 → 埋得更深
        assert abs(r[9] / r[10] - r[8]) < 0.05             # 比值列与 dev/expected 同向
    # 随 τ 校准（3 天 → 2 天），expected 抬升、比值趋 1：信号衰减可见
    ratios = [r[9] / r[10] for r in rows]
    assert ratios[-1] <= ratios[0] + 1e-9
    assert rows[0][10] < rows[-1][10]                      # 类型预期随信念校准抬升


def test_plasticity_history_records_awakening_signal_columns():
    """learn_plasticity 历史 10 列：drift 通道更新带唤醒信号快照
    （dev / expected / 比值）——可复盘"这次可塑性上调是由多剧烈的唤醒驱动"。"""
    clock = [0.0]
    a = _agent(clock,
               true_tau_by_type={MemType.EPISODIC: 2 * DAY},
               true_reconsolidation_by_type={
                   MemType.EPISODIC: {"drift": 3.5, "importance": 1.5}},
               tau_learning_rate=0.3, plasticity_learning_rate=0.3)
    for _ in range(8):
        for _ in range(3):
            _awaken(a, clock)
        a.learn_tau(force=True)
        a.learn_plasticity(force=True)
    rows = [r for r in a._plasticity_history if r[2] == "drift"]
    assert rows and all(len(r) == 10 for r in rows)
    for r in rows:
        assert r[7] is not None and r[8] is not None and r[9] is not None
        assert r[7] > r[8]                   # 唤醒比类型预期剧烈 → drift 上调的触发信号
        assert r[9] > 1.0
    # 信号随校准衰减：expected 抬升 → 比值趋 1
    ratios = [r[9] for r in rows]
    assert ratios[-1] < ratios[0] + 1e-9


# ---------- 边界与回退 ----------

def test_joint_self_consistent_no_signal():
    """自洽环境（无 true_*）：观测调制刻度相同 → dev == expected，联合估计器
    两通道都静默（τ 估计 == 信念 → 偏差过小跳过；drift 样本 == 信念 → 跳过）。"""
    clock = [0.0]
    a = _agent(clock)
    _awaken(a, clock)
    rep_t = a.learn_tau(force=True)
    rep_p = a.learn_plasticity(force=True)
    assert not [u for u in rep_t["updated"] if u["type"] == "episodic"]
    assert not [u for u in rep_p["updated"]
                if u["type"] == "episodic" and u["channel"] == "drift"]
    assert a.cfg.tau_for(MemType.EPISODIC) == 3 * DAY
    assert a.cfg.reconsolidation_factor(MemType.EPISODIC, "drift") == 2.5


def test_joint_switch_falls_back_to_independent():
    """joint_awakening=False 回退两路独立代理：纯 τ 失准时旧代理把整个比值
    判给可塑性（双计数）——保留给对照实验/回退的旧语义。"""
    clock = [0.0]
    a = _agent(clock, joint_awakening=False,
               true_tau_by_type={MemType.EPISODIC: 2 * DAY})
    _awaken(a, clock)
    samples = a._plasticity_samples()["episodic"]["drift"]
    assert samples and samples[0] > 2.8       # 旧代理误调（纯 τ 信号 → drift 上调）


def test_joint_old_four_tuple_fallback():
    """旧 4 元组（无埋藏时长/检索次数）：精确重算不可用 → 回退线性区近似
    （expected·(τ模型/τ参考)）——不崩溃、方向合理。"""
    clock = [0.0]
    a = _agent(clock, true_reconsolidation_by_type={
        MemType.EPISODIC: {"drift": 3.5, "importance": 1.5}})
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    m.awakenings.append([1.0, 0.4823, 0.3996, "episodic"])  # 旧 4 元组
    _, drift_ests, _, _ = a._joint_awakening_estimates(MemType.EPISODIC)
    assert len(drift_ests) == 1
    assert drift_ests[0] > 2.5                # 方向合理（可塑性更高 → 上调）


def test_joint_channel_gates():
    """通道门控与独立代理一致：tau_from_awakenings=False → τ 通道无样本（观测
    不足）；plasticity_from_awakenings=False → drift 通道无样本。"""
    clock = [0.0]
    a = _agent(clock, tau_from_awakenings=False,
               true_tau_by_type={MemType.EPISODIC: 2 * DAY})
    _awaken(a, clock)
    tau_ests, drift_ests, n, _ = a._joint_awakening_estimates(MemType.EPISODIC)
    assert tau_ests == [] and n == 0          # τ 通道门控
    assert drift_ests                          # drift 通道照常（塑性消费方开着）

    clock2 = [0.0]
    b = _agent(clock2, plasticity_from_awakenings=False,
               true_tau_by_type={MemType.EPISODIC: 2 * DAY})
    _awaken(b, clock2)
    tau_ests, drift_ests, _, _ = b._joint_awakening_estimates(MemType.EPISODIC)
    assert tau_ests                           # τ 通道照常
    assert drift_ests == []                   # drift 通道门控


def test_joint_modulation_math():
    """调制反演的自洽性：给定 dev/expected 比值 = 可塑性因子（无 τ 分量），
    p_est 精确回到实测因子（4 元组线性回退路径验证）。"""
    clock = [0.0]
    a = _agent(clock)
    p_t = 3.5
    # 手工注入一条比值 = P_t/S 的事件（4 元组 → 线性回退路径，τ_prev=τ模型 →
    # 不校正 → 全比值判给可塑性 → 调制反演回 p_t）
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    m.awakenings.append([1.0, 1.75, 1.45, "episodic"])
    _, drift_ests, _, _ = a._joint_awakening_estimates(MemType.EPISODIC)
    assert abs(drift_ests[0] - p_t) < 0.1
