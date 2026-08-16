"""Cold 记忆唤醒链路测试：recall() 唤醒的 Warm 记忆在重排、同义扩展、再巩固下
与 Cold 摘要检索行为一致（唤醒 content = 摘要 → 嵌入/子串/类型因子全链路对齐）。"""

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.embedding import cosine_similarity, embed_text
from memagent.memory import Memory, MemType, Tier


def test_rerank_consistent_between_cold_and_awakened():
    """短查询重排：Cold 经摘要命中、唤醒后经 content 命中——同一摘要文本，
    两者都排最前（压过碰撞噪声），大小写不敏感行为一致。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("用户聊过一次项目背景", importance=0.1)
    cold.demote_to_cold("开发决策：AI 分类链路已跑通（OpenAI 兼容）")
    a.store.add("对照实验靠可注入时钟确定性快进", importance=0.9)

    hits1 = a.retrieve("ai", k=2)
    assert hits1 and hits1[0].memory is cold  # Cold：经摘要（含 AI）命中排最前

    revived = a.recall(cold.id[:6])
    assert revived is not None and revived.tier.value == "warm"
    assert revived.content == cold.summary    # 唤醒 content = 摘要文本
    assert "AI" in revived.content            # 大小写不敏感命中面完整保留

    hits2 = a.retrieve("ai", k=3)
    assert hits2 and hits2[0].memory in (cold, revived)  # 唤醒后同样排最前
    assert revived in [h.memory for h in hits2[:2]]


def test_expansion_consistent_between_cold_and_awakened():
    """同义扩展：Cold（摘要嵌入）与唤醒 Warm（content 嵌入 = 摘要）对同一查询
    的 rel 完全一致，且扩展提升量一致（「用餐」→「吃」对两者同效）。"""
    def _rel(agent, q, mem):
        return next(h.relevance for h in agent.retrieve(q, k=5) if h.memory is mem)

    a_on = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a_on.store.add("我昨天去吃了火锅", importance=0.1)
    cold.demote_to_cold("我昨天去吃了火锅")
    r_on = _rel(a_on, "昨天中午用餐了吗", cold)

    a_off = MemoryAgent(cfg=AgentConfig(reconsolidate=False, query_expansion=False))
    c2 = a_off.store.add("我昨天去吃了火锅", importance=0.1)
    c2.demote_to_cold("我昨天去吃了火锅")
    r_off = _rel(a_off, "昨天中午用餐了吗", c2)
    assert r_on > r_off                       # 扩展对 Cold 摘要检索生效

    revived = a_on.recall(cold.id[:6])
    assert revived.content == cold.summary
    r_w = _rel(a_on, "昨天中午用餐了吗", revived)
    assert abs(r_w - r_on) < 1e-9             # 唤醒 rel 与 Cold 完全一致


def test_reconsolidation_consistent_between_cold_and_awakened():
    """再巩固：唤醒继承 mtype（episodic 仍是 episodic）与类型因子——唤醒记忆的
    漂移量与未唤醒的 Cold 对照在同一轮检索下完全一致（move 语义后原 Cold 已移出，
    用对照验证因子一致性；修复前 revived 丢失类型 → 漂移 2.5× 掉回 1.0×）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=True))
    cold = a.store.add("我昨天去吃了火锅", importance=0.1)   # 昨天 → episodic
    cold.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(cold.id[:6])
    assert revived.mtype is MemType.EPISODIC    # 类型继承
    assert revived.kind == "fact"

    control = a.store.add("我昨天去吃了火锅", importance=0.1)
    control.demote_to_cold("我昨天去吃了火锅")  # 未唤醒的 Cold 对照
    base = embed_text("我昨天去吃了火锅")
    a.retrieve("昨天吃了什么", k=5)             # 同一轮检索 → 两者都再巩固
    d_rev = 1.0 - cosine_similarity(revived.embedding, base)
    d_ctl = 1.0 - cosine_similarity(control.embedding, base)
    assert d_rev > 1e-6                         # 确实发生了漂移
    assert abs(d_rev - d_ctl) < 1e-9            # 漂移量一致（同类型因子）


def test_awaken_inherits_history_trajectory():
    """唤醒继承观测轨迹：history 完整复制（Warm + Cold 阶段全部采样）、
    created_at 保持原出生、access_count = 原值 + 1（唤醒本身 = 一次检索）。"""
    clock = [1000.0]
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False), now_fn=lambda: clock[0])
    m = a.store.add("我昨天去吃了火锅", importance=0.2)
    clock[0] += 100
    a._record_sample(m)
    m.demote_to_cold("我昨天去吃了火锅")
    clock[0] += 100
    a._record_sample(m)          # Cold 阶段的采样也在轨迹里
    old_history = list(m.history)
    revived = a.recall(m.id[:6])
    # recall 自身还会追加一条采样（见 test_recall_sample_appends…），
    # 因此继承的轨迹是前缀——Warm + Cold 阶段全部采样都在
    assert revived.history[:-1] == old_history    # 轨迹完整继承（无断层）
    assert revived.created_at == m.created_at     # 出生时间不变
    assert revived.access_count == m.access_count + 1


def test_recall_sample_appends_to_inherited_history():
    """recall 自身的采样追加在继承轨迹之后——曲线连续，无空洞。"""
    clock = [1000.0]
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False), now_fn=lambda: clock[0])
    m = a.store.add("我昨天去吃了火锅", importance=0.2)
    clock[0] += 50
    a._record_sample(m)
    m.demote_to_cold("我昨天去吃了火锅")
    old_history = list(m.history)
    revived = a.recall(m.id[:6])
    assert revived.history[:-1] == old_history    # 继承 + 追加，无空洞
    assert revived.history[-1][0] == clock[0]     # 追加行时间戳 = 唤醒时刻
    assert revived.history[-1][3] == revived.access_count  # 追加行检索次数=当前


def test_awaken_inherits_usage_history_for_semanticization():
    """语义化评分从 history 推导：继承轨迹后唤醒记忆的使用信号不归零——
    评分与 Cold 一致（修复前空 history 恒为 0，类型迁移信号断层）。"""
    clock = [1000.0]
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False), now_fn=lambda: clock[0])
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    for _ in range(3):
        m.touch(clock[0])
        a._record_sample(m)
        clock[0] += 50
    m.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(m.id[:6])
    assert a._semanticization_score(revived) >= a._semanticization_score(m)  # 不丢失既有使用信号
    assert a._semanticization_score(revived) > 0                              # 唤醒不是从零开始


def test_cold_warm_roundtrip_no_proliferation():
    """循环守护：Cold → recall → Warm → 闲置 → 压回 Cold——往返后记忆数不变、
    仍只有一条 Cold（move 语义：唤醒移除原 Cold，避免增殖）；深藏细节跨往返保留。"""
    clock = [1000.0]
    a = MemoryAgent(
        cfg=AgentConfig(reconsolidate=False, sleep_interval_turns=999),
        now_fn=lambda: clock[0],
    )
    a.store.add("我昨天去吃了火锅", importance=0.1)
    a.store.add("我昨天去吃了火锅，味道很好", importance=0.1)
    for m in a.store.all():
        m.last_access = clock[0] - 200 * 24 * 3600
    a.sleep()
    assert len(a.store.all()) == 1              # 合并压缩 → 1 条 Cold
    n0 = len(a.store.all())
    cold = a.store.by_tier(Tier.COLD)[0]

    revived = a.recall(cold.id[:6])
    assert revived is not None and revived.tier is Tier.WARM
    assert len(a.store.all()) == n0             # 唤醒不增殖（原 Cold 移出）
    assert len(a.store.by_tier(Tier.COLD)) == 0 and len(a.store.by_tier(Tier.WARM)) == 1

    for m in a.store.all():
        m.last_access = clock[0] - 200 * 24 * 3600  # 再次闲置 → 压回 Cold
    a.sleep()
    assert len(a.store.all()) == n0             # 往返后仍 1 条（无增殖）
    assert len(a.store.by_tier(Tier.COLD)) == 1
    assert a.find_memories("味道")             # m2 的深藏词跨往返仍可搜


def test_recall_same_cold_twice_no_duplicate():
    """同一 Cold 重复唤醒：第二次返回 None（原 Cold 已移出），不生成第二条 Warm。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("我昨天去吃了火锅", importance=0.1)
    cold.demote_to_cold("我昨天去吃了火锅")
    r1 = a.recall(cold.id[:6])
    assert r1 is not None and r1.tier is Tier.WARM
    n = len(a.store.all())
    r2 = a.recall(cold.id[:6])
    assert r2 is None                           # 无可唤醒的 Cold
    assert len(a.store.all()) == n


def test_awaken_inherits_revisions_and_count():
    """再巩固修订日志随唤醒延续：revisions 完整复制、revision_count 一致。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=True))
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    for _ in range(3):
        a.retrieve("昨天吃了什么", k=5)          # 3 次回忆事件 → 3 条修订
    assert len(m.revisions) == 3 and m.revision_count == 3
    m.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(m.id[:6])
    assert revived.revisions == m.revisions        # 修订日志完整继承
    assert revived.revision_count == m.revision_count


def test_learn_plasticity_uses_awakened_revisions():
    """唤醒继承的修订日志驱动可塑性学习：真实 drift 因子 3.5 ≠ 配置 2.5 时，
    learn_plasticity 依据唤醒记忆的 3 条继承修订（记录 est 3.5）更新 episodic——
    修复前唤醒后修订归零 → 「事件 0 < 3」跳过。"""
    a = MemoryAgent(cfg=AgentConfig(
        reconsolidate=True,
        true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 3.5, "importance": 1.0}},
    ))
    m = a.store.add("我昨天去吃了火锅", importance=0.1)   # episodic
    for _ in range(3):
        a.retrieve("昨天吃了什么", k=5)          # 修订行记录实际应用因子 3.5
    m.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(m.id[:6])
    assert len(revived.revisions) == 3            # 修订随唤醒延续

    rep = a.learn_plasticity(force=True)
    drift_updates = [u for u in rep["updated"]
                     if u["type"] == "episodic" and u["channel"] == "drift"]
    assert drift_updates                          # 继承修订驱动了更新
    assert drift_updates[0]["old"] == 2.5 and abs(drift_updates[0]["est"] - 3.5) < 1e-9
    reasons = [s["reason"] for s in rep["skipped"]
               if s["type"] == "episodic" and s["channel"] == "drift"]
    assert "事件 0 < 3" not in reasons


def test_awakening_deviation_observed_on_recall():
    """唤醒偏差观测：recall 时记录 [时间戳, 实测偏差, 类型预期偏差, 唤醒时刻类型]。

    实测偏差 = 跳升 − 模型延续预测（测试效应）。无 true_tau 时环境与模型自洽：
    实测 == 类型预期 → 比值 1 → 无学习信号（保守）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("我昨天去吃了火锅", importance=0.1)
    cold.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(cold.id[:6])
    assert revived is not None
    assert len(revived.awakenings) == 1
    ts, dev, expected, mtype, dt, n_cold = revived.awakenings[0]
    assert dev > 0                      # 实测跳升 > 模型延续预测
    assert dev < 0.5                    # 归一化强度域（denom 2.8）内的合理幅值
    assert expected > 0                 # 类型预期偏差存在
    assert dev == expected              # 环境自洽（无 true_tau）→ 比值 1 → 无信号
    assert mtype == "episodic"          # 唤醒时刻类型（迁移后也不误归属）
    assert dt >= 0 and n_cold == 0      # 埋藏时长/检索次数随事件记录（联合估计器用）


def test_learn_plasticity_uses_type_expected_awakenings():
    """唤醒偏差观测驱动可塑性学习（类型专属预期偏差锚，非全局中位数）：
    episodic 实测 > 类型预期 → drift 上调；semantic 实测 < 类型预期 → 下调
    （唤醒越剧烈 → 该类型可塑性越活跃）。单一类型有观测也能校准——
    预期来自模型信念而非跨类型对比，旧全局锚的"保守跳过"限制已移除。"""
    def _agent(rows_by_type):
        a = MemoryAgent(cfg=AgentConfig(reconsolidate=False,
                                        plasticity_learning_rate=1.0))
        for mtype, rows in rows_by_type.items():
            for dev, expected in rows:
                m = a.store.add("我昨天去吃了火锅", importance=0.1)
                m.awakenings.append([1.0, dev, expected, mtype])   # mtype 已是字符串
        return a

    # episodic 剧烈（实测 0.45 > 预期 0.30）vs semantic 弱（实测 0.25 < 预期 0.35）
    a = _agent({"episodic": [(0.45, 0.30)] * 3, "semantic": [(0.25, 0.35)] * 3})
    rep = a.learn_plasticity(force=True)
    upd = {u["type"]: u for u in rep["updated"] if u["channel"] == "drift"}
    assert upd["episodic"]["new"] > 2.5        # 默认 episodic drift 2.5 → 上调
    assert upd["episodic"]["est"] > 2.5
    assert upd["semantic"]["new"] < 1.0        # 默认 semantic drift 1.0 → 下调
    assert upd["semantic"]["est"] < 1.0

    # 单一类型有观测 → 也能校准（预期来自模型信念，不依赖跨类型对比）
    a2 = _agent({"episodic": [(0.45, 0.30)] * 3})
    rep2 = a2.learn_plasticity(force=True)
    upd2 = [u for u in rep2["updated"] if u["channel"] == "drift"]
    assert upd2 and upd2[0]["type"] == "episodic" and upd2[0]["new"] > 2.5


def test_plasticity_samples_skips_old_format_awakenings():
    """旧格式唤醒观测（三元组，无类型预期偏差）无法换算相对信号——跳过。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    m.awakenings.append([1.0, 0.45, "episodic"])        # 旧格式 → 跳过
    m.awakenings.append([1.0, 0.45, 0.30, "episodic"])  # 新格式 → 换算
    drift = a._plasticity_samples()["episodic"]["drift"]
    assert len(drift) == 1


def test_observe_awakening_expected_follows_type_tau():
    """判别场景：true τ ≠ 模型 τ 时，实测偏差偏离类型预期（方向正确）。

    真实衰减更快（true τ < 模型 τ）→ 记忆埋得比信念更深 → 实测 > 类型预期
    （上调信号）；真实衰减更慢 → 实测 < 类型预期（下调信号）。埋深选在
    衰减中段（floor 之上）——深埋时两者都饱和于强度下限，信号自限失效。"""
    DAY = 24 * 3600

    def _run(true_tau: float) -> tuple[float, float]:
        clock = [0.0]
        a = MemoryAgent(
            cfg=AgentConfig(
                reconsolidate=False,
                true_tau_by_type={MemType.EPISODIC: true_tau},
            ),
            now_fn=lambda: clock[0],
        )
        m = a.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
        m.access_count = 2
        m.last_access = clock[0]
        clock[0] += 1.2 * 3 * DAY       # 闲置 1.2×模型τ（3 天）——衰减中段
        m.demote_to_cold("我昨天去吃了火锅")
        revived = a.recall(m.id[:6])
        assert revived is not None
        _, dev, expected, _, _, _ = revived.awakenings[0]
        return dev, expected

    dev_fast, exp_fast = _run(2 * DAY)   # 真实衰减更快 → 埋得更深
    assert dev_fast > exp_fast           # 实测 > 类型预期 → 上调信号
    dev_slow, exp_slow = _run(6 * DAY)   # 真实衰减更慢 → 埋得更浅
    assert dev_slow < exp_slow           # 实测 < 类型预期 → 下调信号


def test_awakenings_inherit_across_roundtrips():
    """唤醒偏差观测随 Cold↔Warm 往返继承（与修订日志同语义）：多次唤醒的信号
    累积进学习事件池（滚 12 条）而非每次唤醒后归零——否则学习器每个记忆永远
    只有 1 条观测，事件数永远达不到门控。"""
    DAY = 24 * 3600
    clock = [0.0]
    a = MemoryAgent(cfg=AgentConfig(
        reconsolidate=False,
        true_tau_by_type={MemType.EPISODIC: 2 * DAY},
    ), now_fn=lambda: clock[0])
    m = a.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
    m.access_count = 2
    m.last_access = clock[0]
    clock[0] += 1.2 * 3 * DAY
    m.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(m.id[:6])
    for _ in range(2):
        clock[0] += 1.2 * 3 * DAY
        revived.demote_to_cold("我昨天去吃了火锅")
        clock[0] += 1.2 * 3 * DAY
        revived = a.recall(revived.id[:6])
    assert len(revived.awakenings) == 3        # 继承 2 + 新观测 1
    devs = [aw[1] for aw in revived.awakenings]
    exps = [aw[2] for aw in revived.awakenings]
    assert all(d > e for d, e in zip(devs, exps))  # 真实衰减更快 → 全为上调信号
    rep = a.learn_plasticity(force=True)
    upd = [u for u in rep["updated"] if u["type"] == "episodic"
           and u["channel"] == "drift"]
    assert upd and upd[0]["new"] > 2.5        # 累积信号驱动真实更新


def _cycle_cold_warm(agent, clock, text="我昨天去吃了火锅"):
    """构建一条经历 3 次 Cold↔Warm 往返的记忆（每次闲置到衰减中段后唤醒，
    激活唤醒信号；唤醒链路全是干扰段 → 无干净段）。"""
    DAY = 24 * 3600
    m = agent.store.add(text, importance=0.3, mtype=MemType.EPISODIC)
    m.access_count = 2
    m.last_access = clock[0]
    clock[0] += 1.2 * 3 * DAY          # 闲置衰减到中段
    m.demote_to_cold(text)
    revived = agent.recall(m.id[:6])   # 立即唤醒（埋深 1.2×τ，信号最清晰）
    assert revived is not None
    for _ in range(2):
        clock[0] += 1.2 * 3 * DAY
        revived.demote_to_cold(text)
        revived = agent.recall(revived.id[:6])
    return revived


def test_learn_tau_awakening_source_drives_update():
    """唤醒偏差作为 τ 第二观测源：真实衰减更快（true τ 2 天 < 模型 3 天）→
    实测跳升深于类型预期 → τ 下调。唤醒链路全是干扰段（clean=0）时，
    仅唤醒观测就驱动更新——旧版此时只能报「观测不足」。"""
    DAY = 24 * 3600
    clock = [0.0]
    a = MemoryAgent(cfg=AgentConfig(
        reconsolidate=False,
        true_tau_by_type={MemType.EPISODIC: 2 * DAY},
    ), now_fn=lambda: clock[0])
    revived = _cycle_cold_warm(a, clock)
    assert len(revived.awakenings) == 3
    assert a.fit_report()["by_type"]["episodic"]["clean"] == 0  # 无干净段
    old = a.cfg.tau_for(MemType.EPISODIC)
    rep = a.learn_tau(force=True)
    upd = [u for u in rep["updated"] if u["type"] == "episodic"]
    assert upd and upd[0]["new_tau"] < old        # 唤醒观测驱动 τ 下调


def test_learn_tau_awakening_reverse_direction():
    """反向信号：真实衰减更慢（true τ 6 天 > 模型 3 天）→ 实测跳升浅于
    类型预期（dev < expected）→ τ 上调。"""
    DAY = 24 * 3600
    clock = [0.0]
    a = MemoryAgent(cfg=AgentConfig(
        reconsolidate=False,
        true_tau_by_type={MemType.EPISODIC: 6 * DAY},
        innate_bounds={},  # 真实 τ 6 天 > 出厂情景上限 3 天——本测试验证学习器
                          # 本身能向上校准，需脱离出厂边界隔离学习数学
    ), now_fn=lambda: clock[0])
    _cycle_cold_warm(a, clock)
    devs = [aw[1] for aw in a.store.all()[0].awakenings]
    exps = [aw[2] for aw in a.store.all()[0].awakenings]
    assert all(d < e for d, e in zip(devs, exps))  # 埋得浅 → 实测 < 预期
    old = a.cfg.tau_for(MemType.EPISODIC)
    rep = a.learn_tau(force=True)
    upd = [u for u in rep["updated"] if u["type"] == "episodic"]
    assert upd and upd[0]["new_tau"] > old        # τ 上调


def test_learn_tau_awakening_self_consistent_no_signal():
    """自洽环境（无 true_tau）：实测 == 类型预期 → 唤醒估计 == τ_model →
    不产生误调（「偏差过小」跳过）。"""
    DAY = 24 * 3600
    clock = [0.0]
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False),
                    now_fn=lambda: clock[0])
    revived = _cycle_cold_warm(a, clock)
    assert len(revived.awakenings) == 3
    assert all(aw[1] == aw[2] for aw in revived.awakenings)  # dev == expected
    old = a.cfg.tau_for(MemType.EPISODIC)
    rep = a.learn_tau(force=True)
    assert not [u for u in rep["updated"] if u["type"] == "episodic"]
    assert a.cfg.tau_for(MemType.EPISODIC) == old  # 未被误调


def test_learn_tau_awakening_combined_with_segments():
    """两路互补：干净段反推 + 唤醒偏差代理同向时，合并估计驱动更新。"""
    DAY = 24 * 3600
    clock = [0.0]
    a = MemoryAgent(cfg=AgentConfig(
        reconsolidate=False,
        true_tau_by_type={MemType.EPISODIC: 2 * DAY},
        tau_min_segments=3,
    ), now_fn=lambda: clock[0])
    # 路 ①：干净段（remember + 4 次采样，无访问干扰）
    mseg = a.remember("我昨天去吃了火锅", importance=0.3)
    for _ in range(4):
        clock[0] += 5000
        a._record_sample(mseg)
    # 路 ②：唤醒观测（3 次 Cold↔Warm 往返）
    _cycle_cold_warm(a, clock, text="我记得那个配方")
    assert a.fit_report()["by_type"]["episodic"]["clean"] >= 3
    old = a.cfg.tau_for(MemType.EPISODIC)
    rep = a.learn_tau(force=True)
    upd = [u for u in rep["updated"] if u["type"] == "episodic"]
    assert upd and upd[0]["new_tau"] < old        # 两路同向下调


def test_learn_tau_awakening_disabled():
    """开关关闭：唤醒观测不参与 τ 估计——无干净段时保持「观测不足」跳过。"""
    DAY = 24 * 3600
    clock = [0.0]
    a = MemoryAgent(cfg=AgentConfig(
        reconsolidate=False,
        true_tau_by_type={MemType.EPISODIC: 2 * DAY},
        tau_from_awakenings=False,
    ), now_fn=lambda: clock[0])
    revived = _cycle_cold_warm(a, clock)
    assert len(revived.awakenings) == 3        # 记录照常（塑性消费方开着）
    rep = a.learn_tau(force=True)
    assert not [u for u in rep["updated"] if u["type"] == "episodic"]
    assert "观测不足" in [s["reason"] for s in rep["skipped"]
                         if s["type"] == "episodic"]


def test_plasticity_from_awakenings_disabled():
    """开关关闭：唤醒观测不进入学习事件池；两路消费方都关时 recall 也不新增
    观测（已累积的观测仍随唤醒继承，只是不被学习器消费）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False,
                                    plasticity_from_awakenings=False,
                                    tau_from_awakenings=False))
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    m.awakenings.append([1.0, 0.45, "episodic"])
    assert a._plasticity_samples()["episodic"]["drift"] == []
    # 两路都关时 recall 不新增观测（继承保留已有观测）
    m.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(m.id[:6])
    assert len(revived.awakenings) == 1       # 继承旧观测，无新观测


def test_awakening_recorded_when_tau_source_only():
    """记录开关解耦：plasticity 关、tau 开 → recall 仍记录观测（τ 学习器消费）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False,
                                    plasticity_from_awakenings=False))
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    m.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(m.id[:6])
    assert len(revived.awakenings) == 1       # tau 消费方开启 → 记录
    assert a._plasticity_samples()["episodic"]["drift"] == []  # 但塑性池仍空


def test_awakenings_persist_roundtrip():
    """唤醒偏差观测随序列化往返保留；旧格式 JSON（无该键）加载兼容。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    m.awakenings.append([100.0, 0.35, 0.30, "episodic"])
    m2 = Memory.from_dict(m.to_dict())
    assert m2.awakenings == [[100.0, 0.35, 0.30, "episodic"]]
    d = m.to_dict()
    del d["awakenings"]
    assert Memory.from_dict(d).awakenings == []


def test_learn_tau_clean_segments_survive_recall():
    """唤醒继承历史后：Cold 阶段的干净衰减段继续参与 τ 学习的 clean 统计——
    clean 4→4 不丢、唤醒本身正确记为干扰段（检索事件）、实测 τ 延续；
    学习器门控通过（非「观测不足」）。"""
    clock = [1000.0]
    a = MemoryAgent(
        cfg=AgentConfig(reconsolidate=False, tau_learning=False),
        now_fn=lambda: clock[0],
    )
    m = a.remember("我昨天去吃了火锅", importance=0.3)
    for _ in range(4):
        clock[0] += 5000
        a._record_sample(m)               # 纯衰减采样（无检索/重要性变化）
    m.demote_to_cold("我昨天去吃了火锅")
    r0 = a.fit_report()
    assert r0["by_type"]["episodic"]["clean"] == 4

    clock[0] += 5000                      # 唤醒前流逝 → 唤醒采样与 Cold 末采样错开
    revived = a.recall(m.id[:6])
    assert revived is not None
    r1 = a.fit_report()
    rev_e = next(x for x in r1["memories"] if x["id"] == revived.id)
    assert rev_e["segments"] == 5
    assert rev_e["clean"] == 4            # Cold 阶段干净段全保留
    assert rev_e["interference"] == 1     # 唤醒 = 检索事件 → 干扰段
    assert r1["by_type"]["episodic"]["clean"] == 4
    assert r1["by_type"]["episodic"]["tau_est"] == r0["by_type"]["episodic"]["tau_est"]
    # 学习器门控：干净段足够 → 参与估算（而非「观测不足」）
    rep = a.learn_tau(force=True)
    reasons = [s["reason"] for s in rep["skipped"] if s["type"] == "episodic"]
    assert "观测不足" not in reasons


def test_learn_tau_updates_from_awakened_segments():
    """唤醒记忆的继承段足以驱动真实更新：真实 τ（5 天）≠ 配置 τ（3 天）时，
    learn_tau 依据唤醒记忆的干净段反推实测 τ（≈5 天）并更新 episodic 配置。"""
    clock = [1000.0]
    a = MemoryAgent(
        cfg=AgentConfig(
            reconsolidate=False,
            true_tau_by_type={MemType.EPISODIC: 5 * 24 * 3600},
            tau_min_segments=3,
            innate_bounds={},  # 实测 τ ≈ 5 天 > 出厂情景上限 3 天，脱离边界测学习器
        ),
        now_fn=lambda: clock[0],
    )
    m = a.remember("我昨天去吃了火锅", importance=0.3)
    for _ in range(4):
        clock[0] += 5000
        a._record_sample(m)
    m.demote_to_cold("我昨天去吃了火锅")
    clock[0] += 5000
    revived = a.recall(m.id[:6])
    assert revived is not None

    rep = a.learn_tau(force=True)
    updated = [u for u in rep["updated"] if u["type"] == "episodic"]
    assert updated                      # 唤醒记忆的继承段驱动了更新
    assert abs(updated[0]["tau_est"] - 5 * 24 * 3600) / (5 * 24 * 3600) < 0.01
    assert updated[0]["old_tau"] == 3 * 24 * 3600


def test_awaken_inherits_mtype_and_confidence():
    """类型继承明细：mtype / mtype_confidence / kind 全部继承（含非默认类型）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("学做饭的步骤是先热油", importance=0.2, mtype=MemType.SKILL,
                       mtype_confidence=0.87)
    cold.demote_to_cold("学做饭的步骤是先热油")
    revived = a.recall(cold.id[:6])
    assert revived.mtype is MemType.SKILL
    assert revived.mtype_confidence == 0.87
    assert revived.kind == "fact"


def test_memories_no_awakened_marker_for_never_awakened(capsys):
    """从未唤醒的记忆保持原展示：修订计数照常，无「唤醒自 Cold」标记。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    a.remember("普通事实：项目用 FastAPI")
    a._print_memories()
    out = capsys.readouterr().out
    assert "修订=0" in out
    assert "唤醒自Cold" not in out


def test_memories_awakened_marker_shows_inherited_counts(capsys):
    """唤醒自 Cold 的记忆显示继承的修订/历史条数（长生命周期可追溯）。

    历史条数 = Cold 继承轨迹 + 唤醒事件自身采样（recall 会追加一条观测）。
    """
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("我昨天去吃了火锅", importance=0.1)
    cold.demote_to_cold("我昨天去吃了火锅")
    # 造出可追溯的过去：观测轨迹 + 再巩固修订日志
    cold.history = [
        [1000.0, 0.8, 1000.0, 1, 0.1], [2000.0, 0.6, 1000.0, 1, 0.1],
        [3000.0, 0.4, 1000.0, 1, 0.1], [4000.0, 0.3, 1000.0, 1, 0.1],
    ]
    cold.revision_count = 3
    cold.revisions = [[1.0, 2.5, 0.2, 0.05], [2.0, 2.5, 0.1, 0.0], [3.0, 2.5, 0.3, 0.02]]

    revived = a.recall(cold.id[:6])
    assert revived is not None and revived.awakened_at is not None
    assert revived.revision_count == 3                      # 修订全继承
    assert len(revived.history) == 4 + 1                    # 继承轨迹 + 唤醒采样

    a._print_memories()
    out = capsys.readouterr().out
    assert f"唤醒自Cold(修订=3 历史={len(revived.history)})" in out
    # 标记替换了裸修订计数，不重复显示
    assert " 修订=3 " not in out.replace("唤醒自Cold(修订=3", "")


def test_awakened_at_serialized_and_survives_demote():
    """唤醒标记随序列化往返保留；再压回 Cold 不清除（复苏史可追溯）；
    旧格式 JSON（无该键）加载不报错且为 None。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("我昨天去吃了火锅", importance=0.1)
    cold.demote_to_cold("我昨天去吃了火锅")
    revived = a.recall(cold.id[:6])
    assert revived.awakened_at is not None

    # 序列化往返保留
    m = Memory.from_dict(revived.to_dict())
    assert m.awakened_at == revived.awakened_at

    # 再压回 Cold：标记不清零（Cold↔Warm 往返后仍可追溯复苏史）
    revived.demote_to_cold("我昨天去吃了火锅")
    assert revived.awakened_at is not None

    # 旧格式持久化（无 awakened_at 键）加载正常
    d = revived.to_dict()
    del d["awakened_at"]
    m2 = Memory.from_dict(d)
    assert m2.awakened_at is None
