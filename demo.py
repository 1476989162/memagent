"""脚本演示：分层、衰减、睡眠巩固、再巩固、按类型遗忘。

用很小的 time constant 模拟"几天后"，让你在几秒内看到
Hot/Warm/Cold 的升降级与压缩过程。
"""

import time

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.cli import enable_utf8
from memagent.embedding import cosine_similarity
from memagent.innate import TIME_SCALE
from memagent.memory import MemType, Tier
from memagent.visualize import floor_verification, forgetting_slope

enable_utf8()

# 时间常数压到秒级：τ=30s，"一天"≈3s，便于演示遗忘与巩固；
# 同时按类型区分：技能慢、语义中、情景快
cfg = AgentConfig(
    tau_seconds=30.0,
    tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
    cold_after_seconds=3.0,
    cold_max_access=2,
    hot_after_access=3,
    k=2,
)

agent = MemoryAgent(cfg=cfg)

print("=== 1) 写入个人信息（重要性高，应长期保留）===")
agent.remember("我叫小林，程序员，喜欢咖啡和爬山，住在杭州。")
agent.remember("我的生日是 1995 年 3 月 14 日。")

print("=== 2) 对话并反复检索同一条记忆 → 应升级到 Hot ===")
for i in range(1, 5):
    reply, hits = agent.respond("你喜欢爬山吗？")
    print(f"第 {i} 轮: {reply}")
    for h in hits:
        print(f"   命中 [{h.memory.tier.value}] {h.memory.content[:30]} 次数={h.memory.access_count}")
print(agent.stats())

print("\n=== 3) 等待超过 cold_after_seconds(3s)，触发睡眠巩固 ===")
time.sleep(3.2)
report = agent.sleep()
print(f"梦境报告: 重放 {report['replayed_count']} 条白天经历（再激活：检索次数+1、强度微调），"
      f"Hot 降级 {len(report['hot_demoted'])} 条, "
      f"压缩 {report['cold_compressed']} 条 → {report['clusters']} 条 Cold 摘要")
agent._print_memories()

# 睡眠中断模拟：只睡 0.5 秒 → 重放预算 0 条 → 白天经历全部未回放、次日模糊
int_cfg = AgentConfig(reconsolidate=False, cold_after_seconds=999999.0)
int_a = MemoryAgent(cfg=int_cfg)
m1 = int_a.remember("我上午改了登录页的样式", importance=0.3)
m2 = int_a.remember("我下午写了检索的测试", importance=0.3)
time.sleep(0.1)
rp = int_a.sleep(duration=0.5)
print(f"睡眠中断模拟（2 条白天经历，只睡 0.5s → 预算 {rp['replayed_count']} 条）: "
      f"未回放 {rp['unreplayed_count']} 条次日模糊 "
      f"（重要度 {m1.importance:.2f} / {m2.importance:.2f}，原 0.30——睡眠不足让记忆更模糊）")

# 心游（默认模式网络）：无查询时按强度加权自发想起——想起即再激活、越想起越牢
print("\n=== 3.5) 心游：无查询时的自发想起（默认模式网络）===")
wm_cfg = AgentConfig(reconsolidate=False, cold_after_seconds=999999.0)
wm = MemoryAgent(cfg=wm_cfg)
strong = wm.remember("我经常去爬山（重要技能）", importance=0.9)
weak = wm.remember("很久以前的琐事", importance=0.1)
strong.access_count = 5
weak.last_access -= 10 * 24 * 3600   # 深衰 → 触底
import random as _r
rng = _r.Random(42)          # 同一个 rng 复用（每轮新种子会重复同一抽签）
picks = {"strong": 0, "weak": 0}
for _ in range(300):
    p = wm.spontaneous_recall(rng=rng)
    picks["strong" if p is strong else "weak"] += 1
print(f"300 次心游采样: 想起重要技能 {picks['strong']} 次 / 琐事 {picks['weak']} 次"
      f"（强度加权——牢的记忆更常被想起）")
acc_before = strong.access_count
wm.spontaneous_recall(rng=_r.Random(1))
print(f"想起一次后: 检索次数 {acc_before} → {strong.access_count}（再激活测试效应），"
      f"成为当晚睡眠回放候选（闭环）")

# 场景重建：把相关记忆片段组合成连贯场景（人脑回忆的是片段组合而非单条事实）
print("\n=== 3.6) 场景重建：把相关片段拼成连贯场景 ===")
from memagent.agent import format_scene
sc_cfg = AgentConfig(reconsolidate=False, cold_after_seconds=999999.0)
sc = MemoryAgent(cfg=sc_cfg)
sc.remember("我在西湖边散步", importance=0.5)
time.sleep(0.05)
sc.remember("西湖边的风很舒服", importance=0.4)
time.sleep(0.05)
sc.remember("我们在西湖边散步后吃了晚饭", importance=0.4)
time.sleep(0.05)
sc.remember("项目后端用 FastAPI 和 SQLite", importance=0.6)  # 无关片段
scene = sc.compose_scene("西湖那天")
print(format_scene(scene) if scene else "（未能重建出场景）")
print("（无关记忆未入场景；场景里被扩展的片段获得再激活测试效应）")

print("\n=== 4) 用摘要索引触发回忆（海马体索引）===")
reply, hits = agent.respond("你是谁？")
print(f"Agent> {reply}")
for h in hits:
    if h.via_summary:
        print(f"   [索引命中] {h.memory.summary}")
        print(f"   → /recall {h.memory.id[:6]} 可唤醒原始内容")

print("\n=== 5) 唤醒一条 Cold 记忆 ===")
cold = agent.store.by_tier(Tier.COLD)
if cold:
    revived = agent.recall(cold[0].id[:6])
    print(f"唤醒成功: {revived.content}")

print("\n=== 6) 记忆再巩固：回忆按重要程度微调原始记忆 ===")
low = agent.remember("用户说：我在楼下便利店买了瓶水", importance=0.1)
high = agent.remember("我的名字是小林", importance=0.95)
low_before = low.embedding[:]
high_before = high.embedding[:]
for _ in range(3):
    agent.retrieve("我在楼下便利店买了瓶水", k=1)
agent.retrieve("你是谁", k=1)
print(f"低重要性记忆（重要={low.importance:.2f}）修订 {low.revision_count} 次，"
      f"向量向回忆情境漂移，余弦变化={cosine_similarity(low_before, low.embedding):.4f}")
print(f"高重要性记忆（重要={high.importance:.2f}）修订 {high.revision_count} 次，"
      f"完全冻结：向量未变化={high.embedding == high_before}")

print("\n=== 6.3) 按类型再巩固：技能稳定、情景易被改写 ===")
skill_m = agent.remember("我在练习做饭", importance=0.1, mtype=MemType.SKILL)
epi_m = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
skill_b, epi_b = skill_m.embedding[:], epi_m.embedding[:]
for _ in range(3):
    agent.retrieve("练习做饭", k=1)
    agent.retrieve("昨天去吃了火锅", k=1)
print(f"技能类（{skill_m.mtype.value}）：修订 {skill_m.revision_count} 次，余弦变化="
      f"{cosine_similarity(skill_b, skill_m.embedding):.4f}（几乎不变）")
print(f"情景类（{epi_m.mtype.value}）：修订 {epi_m.revision_count} 次，余弦变化="
      f"{cosine_similarity(epi_b, epi_m.embedding):.4f}（被情境明显改写）")

print("\n=== 6.4) 按类型分流内容钩子：技能一致性校验、情景情境改写 ===")
from memagent.checkers import consistency_checker

hook_agent = MemoryAgent(
    cfg=AgentConfig(tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0}),
    content_updater=lambda m, q, lab: m.content + f"（回忆情境:{q}）",   # 通用：情境改写
    content_updaters={MemType.SKILL: consistency_checker()},            # 技能：一致性校验
)
sk = hook_agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
ep = hook_agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
sk_before, ep_before = sk.content, ep.content
hook_agent.retrieve("西红柿炒蛋的做法", k=1)      # 一致 → 校验留痕，不改写
hook_agent.retrieve("我不会做西红柿炒蛋", k=1)    # 冲突 → 留痕不改写
hook_agent.retrieve("昨天去吃了火锅", k=1)        # 情景 → 情境改写
print(f"技能类: 内容未变={sk.content == sk_before}，修订 {sk.revision_count} 次，"
      f"校验 {len(sk.checks)} 次: " + "、".join(c[2] for c in sk.checks))
print(f"情景类: 内容被情境改写={ep.content != ep_before} → {ep.content}")

print("\n=== 6.5) 按类型分遗忘曲线：技能慢、情景快（自动识别）===")
skill = agent.remember("我学会了弹钢琴", importance=0.1)
epi = agent.remember("我昨天去吃了火锅", importance=0.1)
sem = agent.remember("北京是中国的首都", importance=0.1)
print(f"自动识别: 「我学会了弹钢琴」→{skill.mtype.value} / 「我昨天去吃了火锅」→{epi.mtype.value} / "
      f"「北京是中国的首都」→{sem.mtype.value}")
for m in (skill, epi, sem):
    m.last_access = time.time() - 12  # 三条记忆同时闲置 12 秒
print(f"闲置 12 秒后强度: 技能={agent._strength(skill):.2f} / 语义={agent._strength(sem):.2f} / "
      f"情景={agent._strength(epi):.2f}  （τ: 技能90s / 语义30s / 情景8s）")

print("\n=== 6.6) 检索同义扩展：问法与记忆措辞不同也能命中 ===")
# 人称互换（您→我）与同义词替换（用餐→吃），让书面/间接问法命中口语记忆
from memagent.synonyms import expand_query

print(f"  查询扩展示例: 「昨天中午用餐了吗」→ {expand_query('昨天中午用餐了吗')}")
ex_off = MemoryAgent(cfg=AgentConfig(query_expansion=False, reconsolidate=False))
ex_on = MemoryAgent(cfg=AgentConfig(query_expansion=True, reconsolidate=False))
for a in (ex_off, ex_on):
    a.remember("我昨天去吃了火锅", importance=0.1)
    a.remember("我叫小林", importance=0.1)
q_off = ex_off.retrieve("昨天中午用餐了吗", k=1)[0]
q_on = ex_on.retrieve("昨天中午用餐了吗", k=1)[0]
print(f"  「昨天中午用餐了吗」→ rel: 关扩展 {q_off.relevance:.2f} / 开扩展 {q_on.relevance:.2f}（同义词替换）")
q2_off = ex_off.retrieve("您叫什么名字", k=1)[0]
q2_on = ex_on.retrieve("您叫什么名字", k=1)[0]
print(f"  「您叫什么名字」→ 命中「{q2_on.memory.content}」rel: {q2_off.relevance:.2f} → {q2_on.relevance:.2f}（人称互换）")

print("\n=== 7) 持续观测：验证预测与真实遗忘的贴合度 ===")
# 模拟"真实"遗忘比配置快 2.5 倍（τ 失准场景）：观测用真τ，预测用配置τ
val_cfg = AgentConfig(
    tau_by_type={MemType.EPISODIC: 30.0},
    true_tau_by_type={MemType.EPISODIC: 12.0},
)
val = MemoryAgent(cfg=val_cfg)
val.remember("用户说：我昨天去看了一场电影", importance=0.1)  # 自动识别为 episodic
for _ in range(4):
    val._observe()  # 每轮对话后自动观测（demo 里手动触发）
    time.sleep(0.5)
val.retrieve("昨天看的电影怎么样", k=1)  # 一次检索 → 制造干扰段
val._observe()
print(val.format_fit())
print("（配置τ=30秒 但真实遗忘τ=12秒 → 贴合度低，提示应调小该类型 τ）")

print("\n最终统计:", agent.stats())

print("\n=== 8) 分类器与回复生成：LLM（OpenAI 兼容）或离线回退 ===")
llm_on = bool(getattr(agent.classifier, "available", False))
print(f"LLM 分类：{'已启用（' + agent.classifier.model + '）' if llm_on else '未启用（未设置 OPENAI_API_KEY，使用关键词回退）'}")
resp_on = bool(agent.responder and agent.responder.available)
if resp_on:
    print(f"LLM 回复生成：已启用（{agent.responder.model}）")
else:
    print("LLM 回复生成：未启用（未设置 OPENAI_API_KEY，使用模板回复）")
for s in ["我学会了弹钢琴", "我昨天去吃了火锅", "北京是中国的首都"]:
    mt, conf, src = agent.classify_text(s)
    print(f"  「{s}」→ {mt.value}（置信 {conf:.2f}，来源 {src}）")

print("\n=== 9) 参数自适应：遗忘曲线学习器自动调 τ ===")
# 配置 τ=6s，但“真实”遗忘 τ=2s → 学习器应从观测中逼近真实值
learn_cfg = AgentConfig(
    tau_by_type={MemType.EPISODIC: 6.0},
    true_tau_by_type={MemType.EPISODIC: 2.0},
)
lea = MemoryAgent(cfg=learn_cfg)
lea.remember("用户说：我昨天去看了场电影", importance=0.1)  # 自动识别 episodic
traj = []
for _round in range(2):
    for _ in range(8):  # 观测 8 轮（积累衰减数据）
        lea._observe()
        time.sleep(0.3)
    for _ in range(6):  # 学习 6 次（EMA 逼近真实 τ）
        r = lea.learn_tau()
        if r["updated"]:
            traj.append(r["updated"][0]["new_tau"])
print(f"学习轨迹（配置 6s → 真实 2s）: {[round(t, 1) for t in traj]}")
d = lea.fit_report()["by_type"]["episodic"]
print(f"学习后: 配置τ={lea.cfg.tau_for(MemType.EPISODIC):.1f}s 实测τ≈{d['tau_est']:.1f}s 贴合度={d['fit'] * 100:.0f}%")
# 第二观测源：唤醒偏差——实测跳升深于类型预期 → 该类型衰减比信念快 → τ 下调。
# 唤醒链路全是干扰段（无干净段），仅唤醒观测就驱动更新（旧版此时报"观测不足"）。
aw_cfg = AgentConfig(
    tau_by_type={MemType.EPISODIC: 3 * 24 * 3600},
    true_tau_by_type={MemType.EPISODIC: 2 * 24 * 3600},
)
aw_clock = [0.0]
aw = MemoryAgent(cfg=aw_cfg, now_fn=lambda: aw_clock[0])
awm = aw.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
awm.access_count = 2
awm.last_access = aw_clock[0]
for _ in range(3):  # 3 次 Cold↔Warm 往返 → 3 条唤醒观测
    aw_clock[0] += 1.2 * 3 * 24 * 3600
    awm.demote_to_cold("我昨天去吃了火锅")
    awm = aw.recall(awm.id[:6])
r = aw.learn_tau(force=True)
au = [x for x in r["updated"] if x["type"] == "episodic"][0]
ratio = awm.awakenings[0][1] / awm.awakenings[0][2]
print(f"唤醒偏差第二观测源: 实测/预期 ≈{ratio:.2f}（埋得比信念深）→ τ_est="
      f"{au['tau_est'] / (24 * 3600):.2f} 天 → episodic τ {au['old_tau'] / (24 * 3600):.1f} → "
      f"{au['new_tau'] / (24 * 3600):.2f} 天（真实 2 天，无干净段）")

# 两路信号同场景收敛：干净段 + 唤醒偏差同时在场，画收敛轨迹图
print("\n=== 9.5) 两路信号收敛轨迹：干净段 vs 唤醒偏差互相印证 ===")
conv_clock = [0.0]
conv_cfg = AgentConfig(
    tau_by_type={MemType.EPISODIC: 3 * 24 * 3600},
    true_tau_by_type={MemType.EPISODIC: 2 * 24 * 3600},
    tau_learning_rate=0.3,
)
conv = MemoryAgent(cfg=conv_cfg, now_fn=lambda: conv_clock[0])
conv.store.add("用户说：我昨天去看了场电影", importance=0.1, mtype=MemType.EPISODIC)
conv_aw = conv.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
conv_aw.access_count = 2
for _ in range(10):
    conv_clock[0] += 6 * 3600
    conv._observe()                    # 干净衰减段源
    conv_clock[0] += 1.1 * 3 * 24 * 3600
    conv_aw.demote_to_cold("火锅聚餐（已归档）")
    conv_aw = conv.recall(conv_aw.id[:6])   # 唤醒偏差源
    conv.learn_tau(force=True)
tau_traj = [(row[4] / TIME_SCALE / 86400, row[8]) for row in conv._learn_history]
print(f"EMA 轨迹（天）: {[round(t, 2) for t, _ in tau_traj]}")
print(f"唤醒比值（→1）: {[round(r, 3) for _, r in tau_traj]}")
def _days_or_na(v):
    """agent-秒 → 人类-天（对任意 TIME_SCALE 正确）"""
    return f"{v / TIME_SCALE / 86400:.2f}" if v is not None else "N/A"

print(f"两路互相印证: 干净段 τ_est={_days_or_na(conv._learn_history[-1][6])} 天 / "
      f"唤醒 τ_est={_days_or_na(conv._learn_history[-1][7])} 天 → 配置 τ "
      f"{_days_or_na(conv.cfg.tau_for(MemType.EPISODIC))} 天（真实 2 天）")
print(f"唤醒信号复盘: 每轮记录实际使用的 dev/expected（方向可追）——"
      f"首轮 dev={conv._learn_history[0][9]:.3f} vs 预期 {conv._learn_history[0][10]:.3f}（> → 下调）"
      f" · 末轮 dev={conv._learn_history[-1][9]:.3f} vs 预期 {conv._learn_history[-1][10]:.3f}"
      f"（趋 1 = 已校准）")
conv.plot_tau_convergence("tau_convergence")
print("收敛轨迹图已导出: tau_convergence.svg（两路 τ_est 线 + 比值子图逼近 1；"
      "tau_convergence.csv 现含 dev/expected 信号列）")

print("\n=== 9.7) τ↔可塑性联合估计：一次唤醒事件同时更新 τ 与 drift ===")
# dev 同时编码 τ 失准与可塑性：真 τ=2 天、真 drift=3.5（配置信念 3 天 / 1.0）。
# 一次唤醒 → 联合估计器拆出两路信号，两路 EMA 跨轮互相修正、加速收敛。
jnt_clock = [0.0]
jnt_cfg = AgentConfig(
    tau_by_type={MemType.EPISODIC: 3 * 24 * 3600},
    true_tau_by_type={MemType.EPISODIC: 2 * 24 * 3600},
    reconsolidation_by_type={MemType.EPISODIC: {"drift": 1.0, "importance": 1.0}},
    true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 3.5, "importance": 1.0}},
    tau_learning_rate=0.2,
    joint_awakening=True,
    awakening_plasticity_gain=0.5,
)
jnt = MemoryAgent(cfg=jnt_cfg, now_fn=lambda: jnt_clock[0])
jnt_m = jnt.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
jnt_m.access_count = 2
jnt_m.last_access = jnt_clock[0]
for _ in range(8):  # 8 轮：每轮 3 次 Cold↔Warm 往返 = 3 条唤醒观测
    for _ in range(3):
        jnt_clock[0] += 1.1 * 3 * 24 * 3600
        jnt_m.demote_to_cold("火锅聚餐（已归档）")
        jnt_m = jnt.recall(jnt_m.id[:6])
    jnt.learn_tau(force=True)
    jnt.learn_plasticity(force=True)  # 两路 EMA 同轮更新 → 互相加速
j_tau = [r[4] / TIME_SCALE / 86400 for r in jnt._learn_history]
j_ratio = [r[8] for r in jnt._learn_history]
j_drift = [r[5] for r in jnt._plasticity_history if r[2] == "drift"]
j_d = jnt.cfg.reconsolidation_by_type[MemType.EPISODIC]
print(f"τ EMA 轨迹（天，信念 3 → 真实 2）: {[round(t, 2) for t in j_tau]}")
print(f"唤醒比值（→1）: {[round(r, 3) for r in j_ratio]}")
print(f"drift EMA 轨迹（信念 1.0 → 真实 3.5）: {[round(x, 2) for x in j_drift]}")
print(f"联合收敛: τ={jnt.cfg.tau_for(MemType.EPISODIC) / TIME_SCALE / 86400:.2f} 天 / "
      f"drift={j_d['drift']:.2f}（真实 2 天 / 3.5）——两路 EMA 互相加速")

print("\n=== 10) 类型迁移：情景记忆语义化 ===")
# 独立 Agent：检索事件新鲜度压到 5 秒，几秒内演示完整的双向迁移
mig_cfg = AgentConfig(
    tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
    cold_after_seconds=3.0,
    semanticization_tau_seconds=5.0,   # 检索事件 5 秒内算"近期"
    semanticize_threshold=3.0,         # 近期检索 ≥3 次 → 固化为语义
    desemanticize_threshold=0.8,       # 评分 <0.8 → 淡化为情景
    replay=False,                      # 聚焦迁移演示：回放会在睡眠时重新激活记忆，干扰"停止检索→淡化"流程
)
mig = MemoryAgent(cfg=mig_cfg)
m = mig.remember("我昨天去看了场电影", importance=0.1)  # 自动识别为 episodic
print(f"初始类型: {m.mtype.value}（τ={mig_cfg.tau_for(m.mtype):.0f}s）")
for _ in range(4):  # 连续检索 4 次 → 测试效应累积语义化评分
    mig.retrieve("昨天去看了场电影", k=1)
    time.sleep(0.3)
print(f"检索 4 次后语义化评分 = {mig._semanticization_score(m):.2f}")
rep = mig.sleep()
print(f"睡眠后类型: {m.mtype.value}（迁移 {rep['migrations']} 条）→ τ 由 8s 变 "
      f"{mig_cfg.tau_for(m.mtype):.0f}s，强度回升到 {mig._strength(m):.2f}（语义化让记忆更持久）")
time.sleep(8)  # 停止检索，评分衰减
rep2 = mig.sleep()
print(f"停止检索 8 秒后评分 = {mig._semanticization_score(m):.2f}，再次睡眠 → 类型 {m.mtype.value}"
      f"（反向淡化 {rep2['migrations']} 条）")

print("\n=== 11) 再巩固因子自适应：从观测估计实际可塑性 ===")
# 隐藏真实可塑性：情景类实际 drift 2.5 / importance 1.5，但配置（信念）是 1.0/1.0
pla_cfg = AgentConfig(
    tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
    cold_after_seconds=999999.0,  # 聚焦因子学习，不触发压缩
    reconsolidation_by_type={MemType.EPISODIC: {"drift": 1.0, "importance": 1.0}},
    true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 2.5, "importance": 1.5}},
)
pla = MemoryAgent(cfg=pla_cfg)
m = pla.remember("我昨天去看了场电影", importance=0.1)
for _ in range(6):  # 6 次回忆 → 6 个修订事件（每个都记录实际应用的因子）
    pla.retrieve("昨天去看了场电影", k=1)
traj = []
for _ in range(4):  # 学习 4 次（EMA 逼近真实可塑性）
    r = pla.learn_plasticity()
    for u in r["updated"]:
        if u["channel"] == "drift":
            traj.append(u["new"])
d = pla.cfg.reconsolidation_by_type[MemType.EPISODIC]
print(f"drift 因子轨迹（配置 1.0 → 真实 2.5）: {[round(x, 2) for x in traj]}")
print(f"学习后: drift={d['drift']:.2f} importance={d['importance']:.2f}（真实 2.5 / 1.5）")

print("\n=== 12) 记忆类型画像：每类型完整的遗忘 / 可塑性 / 压缩配置 ===")
print(agent.profile_table())

print("\n=== 13) 触底验证：实测触底时刻 vs 遗忘斜率预测 ===")
# 模拟时钟：模型"信念" τ=30s，真实环境 τ=4s——观测按真实衰减采样，预测按模型
vc_cfg = AgentConfig(
    tau_seconds=30.0,
    tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 30.0},
    true_tau_by_type={MemType.EPISODIC: 4.0},
)
clock = {"t": 1000.0}
vc = MemoryAgent(cfg=vc_cfg, now_fn=lambda: clock["t"])
vm = vc.remember("我昨天去了趟图书馆", importance=0.1, mtype=MemType.EPISODIC)
for _ in range(60):  # 每秒采样一次，直到观测到强度触底 0.2
    clock["t"] += 1.0
    vc._observe()
    if vm.history[-1][1] <= 0.2 + 1e-6:
        break
fc = floor_verification(vc, vm, clock["t"])
sl = forgetting_slope(vc, vm, clock["t"])
print(f"模型预测: 从最后访问起 {fc['predicted_dt']:.1f} 秒触底（遗忘斜率预测 {sl['label']}）")
print(f"实测: {fc['actual_dt']:.1f} 秒后强度到达 0.2 下限")
print(f"验证结论: {fc['label']}")

print("\n=== 14) 唤醒事件标注：fit 图展示 learn_tau 校准的信号衰减 ===")
# 配置 τ=3 天、真实 τ=2 天：多次 Cold↔Warm 往返累积唤醒观测，learn_tau 校准
# 过程中每条唤醒事件的比值（dev/expected）应趋 1——fit 图把全部事件标注出来：
# 菱形 + 红/青双条 + 信号徽章（颜色随比值：红 >1 应下调 → 灰/青 已校准）。
aw_clk = [0.0]
aw_agent = MemoryAgent(cfg=AgentConfig(
    tau_by_type={MemType.EPISODIC: 3 * 24 * 3600},
    true_tau_by_type={MemType.EPISODIC: 2 * 24 * 3600},
    tau_learning_rate=0.3,
    joint_awakening=True,
), now_fn=lambda: aw_clk[0])
awm = aw_agent.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
awm.access_count = 2
awm.last_access = aw_clk[0]
for _ in range(3):  # 3 轮 × 3 次唤醒 → 9 条唤醒观测
    for _ in range(3):
        aw_clk[0] += 1.1 * 3 * 24 * 3600
        awm.demote_to_cold("火锅聚餐（已归档）")
        awm = aw_agent.recall(awm.id[:6])
    aw_agent.learn_tau(force=True)
from memagent.visualize import _awakening_events
aw_evs = _awakening_events(awm)
print(f"唤醒事件 {len(aw_evs)} 条：比值 {[round(e['ratio'], 2) for e in aw_evs[:5]]}… → "
      f"τ {aw_agent.cfg.tau_for(MemType.EPISODIC) / TIME_SCALE / 86400:.2f} 天（真实 2）")
aw_files = aw_agent.plot_curves("memories_curves_awakening")
print("已导出：" + "，".join(aw_files))
print("打开 memories_curves_awakening.svg：每条唤醒事件一个 ◇ + 红/青双条 + 信号徽章")

print("\n=== 15) 导出记忆强度曲线图 ===")
files = agent.plot_curves("memories_curves")
print("已导出：" + "，".join(files))
print("用浏览器打开 memories_curves.svg 查看：实线=遗忘曲线预测，圆点=实际采样")
p = agent.plot_interactive()
print(f"多视图仪表盘：{p}（曲线/记忆地图/分布/Top列表四面板联动）")
print("\n提示：设置 OPENAI_API_KEY（及可选 OPENAI_BASE_URL/OPENAI_MODEL）可同时启用 LLM 类型分类\n      与 LLM 回复生成——配置 responder 后，无相关记忆时 Agent 也能直接聊天而不只是“不了解”。\n      CLI 里用 /classify <文本> 测试分类；python llm_classify_demo.py 验证完整链路。")
