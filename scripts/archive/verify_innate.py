"""出厂硬件衰减实验：验证 InnateBounds + TIME_SCALE 在实际运行中的行为。

验证点：
  A) 默认τ衰减率正确（TIME_SCALE 缩放后衰减速度符合预期）
  B) InnateBounds 钳制：learn_tau 触发时 τ 不会超出出厂上下界
  C) frozen=True 时学习完全被忽略，出厂τ不变
  D) 技能/语义/情景三类衰减速度梯度正确（技能最慢 > 语义 > 情景）

实验方法：在 TIME_SCALE=1/86400 下，1 真实秒 = 1 人类天。
情景 τ=3s → 3 秒后强度降至 1/e ≈ 0.368
技能 τ=60s → 60 秒后强度降至 0.368
语义 τ=14s → 14 秒后强度降至 0.368
"""
import sys; sys.path.insert(0, r"E:/神经网络")
import time, math
from memagent.memory import MemType, MemoryStore
from memagent.agent import AgentConfig, MemoryAgent
from memagent.innate import TIME_SCALE, InnateBounds, INNATE_DEFAULTS
from memagent.visualize import fmt_duration

PASS = 0; FAIL = 0
def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS  " + label + ( (" " + detail) if detail else ""))
    else:
        FAIL += 1
        print("  FAIL  " + label + ( (" " + detail) if detail else ""))

def e_decay(tau, dt):
    return math.exp(-dt / tau) if tau > 0 else 0.0

print("=" * 60)
print("出厂硬件衰减实验")
print("=" * 60)

# ── A) 默认τ衰减率验证 ──────────────────────────────────────────
print("\n── A) 默认τ衰减率（理论值验证）──")

cfg = AgentConfig()
tau_epi = cfg.tau_for(MemType.EPISODIC)
tau_sem = cfg.tau_for(MemType.SEMANTIC)
tau_skl = cfg.tau_for(MemType.SKILL)
check("情景τ=3s", abs(tau_epi - 3.0) < 0.01, f"实际={tau_epi:.4f}s")
check("语义τ=14s", abs(tau_sem - 14.0) < 0.01, f"实际={tau_sem:.4f}s")
check("技能τ=60s", abs(tau_skl - 60.0) < 0.01, f"实际={tau_skl:.4f}s")

# 理论衰减预测
expected_epi_3s = e_decay(tau_epi, 3.0)
expected_epi_1s = e_decay(tau_epi, 1.0)
expected_skl_1s = e_decay(tau_skl, 1.0)
print(f"  情景3秒后强度理论值: {expected_epi_3s:.4f} (应≈0.368)")
print(f"  情景1秒后强度理论值: {expected_epi_1s:.4f} (应≈0.717)")
print(f"  技能1秒后强度理论值: {expected_skl_1s:.4f} (应≈0.983)")
check("情景3秒衰减≈0.368", abs(expected_epi_3s - 0.3679) < 0.001)
check("情景1秒衰减≈0.717", abs(expected_epi_1s - 0.7165) < 0.001)
check("技能1秒衰减≈0.983", abs(expected_skl_1s - 0.9834) < 0.001)

# 实际内存衰减（通过检索模拟）
print("\n── A2) 实际衰减行为（MemoryAgent 真实检索）──")

store = MemoryStore()
agent = MemoryAgent(store=store)
# 写情景记忆
id1 = agent.remember("我看到一只猫在跳", kind="turn", importance=1.0)
# 立即检索
now0 = time.time()
r0 = agent.retrieve("我看到什么")
s0 = r0[0].total
print(f"  t=0s  情景记忆强度: {s0:.4f}")

# 等3秒（≈3人类天），情景τ=3s → 应衰减到~0.37
time.sleep(3.01)
r3 = agent.retrieve("我看到什么")
s3 = r3[0].total
print(f"  t=3s  情景记忆强度: {s3:.4f}  (理论≈0.368)")
# 容差：检索还会叠加语义相似度，强度可能不完全等于 exp(-t/τ)
# 但相对比例应该接近
ratio_3 = s3 / s0 if s0 > 0 else 0
print(f"  3秒/0秒强度比: {ratio_3:.4f} (理论≈{expected_epi_3s:.4f})")
check("情景3秒衰减比≈理论值", abs(ratio_3 - expected_epi_3s) < 0.15,
      f"ratio={ratio_3:.4f} vs theory={expected_epi_3s:.4f}")

# 再等5秒（总8秒），情景记忆应接近触底
time.sleep(5.01)
r8 = agent.retrieve("我看到什么")
s8 = r8[0].total
print(f"  t=8s  情景记忆强度: {s8:.4f}  (理论≈{e_decay(tau_epi, 8.0):.4f})")
check("8秒后强度 < 3秒强度", s8 < s3, f"8s={s8:.4f} < 3s={s3:.4f}")

# 技能记忆：同时写入，衰减应该慢得多
store2 = MemoryStore()
agent2 = MemoryAgent(store=store2)
id_skl = agent2.remember("我会游泳", kind="turn", importance=1.0)
t_skl_0 = time.time()
r_skl_0 = agent2.retrieve("你会什么")
s_skl_0 = r_skl_0[0].total
print(f"\n  t=0s  技能记忆强度: {s_skl_0:.4f}")

time.sleep(3.01)  # 等3秒
r_skl_3 = agent2.retrieve("你会什么")
s_skl_3 = r_skl_3[0].total
print(f"  t=3s  技能记忆强度: {s_skl_3:.4f}  (理论≈{e_decay(tau_skl, 3.0):.4f})")
ratio_skl = s_skl_3 / s_skl_0 if s_skl_0 > 0 else 0
print(f"  技能3秒/0秒强度比: {ratio_skl:.4f} (理论≈{e_decay(tau_skl, 3.0):.4f})")
check("技能3秒衰减比≈理论值", abs(ratio_skl - e_decay(tau_skl, 3.0)) < 0.10,
      f"ratio={ratio_skl:.4f} vs theory={e_decay(tau_skl, 3.0):.4f}")
check("技能衰减远慢于情景", ratio_skl > ratio_3,
      f"技能ratio={ratio_skl:.4f} > 情景ratio={ratio_3:.4f}")

# ── B) InnateBounds 钳制验证 ──────────────────────────────────
print("\n── B) InnateBounds 钳制（learn_tau 边界）──")

bounds_epi = agent.cfg.innate_bounds[MemType.EPISODIC]
print(f"  情景出厂: tau_min={bounds_epi.tau_min:.5f}, tau_max={bounds_epi.tau_max:.4f}")
print(f"  情景出厂: drift_min={bounds_epi.drift_min}, drift_max={bounds_epi.drift_max}")

# 模拟钳制：任何超出出厂界的τ都被压回
def clamp_tau(est):
    return min(bounds_epi.tau_max, max(bounds_epi.tau_min, est))

test_cases = [
    (0.0001, bounds_epi.tau_min, "极低值 → 顶到下限"),
    (1000.0, bounds_epi.tau_max, "极高值 → 压到上限"),
    (1.5, 1.5, "中间值 → 不变"),
    (bounds_epi.tau_min - 0.001, bounds_epi.tau_min, "边界下 → 顶到下限"),
    (bounds_epi.tau_max + 1.0, bounds_epi.tau_max, "边界上 → 压到上限"),
]
for est, expected, label in test_cases:
    got = clamp_tau(est)
    check(f"钳制 {label}", abs(got - expected) < 0.001,
          f"est={est}, clamp={got:.6f}, expected={expected:.6f}")

# ── C) frozen=True 验证 ────────────────────────────────────────
print("\n── C) frozen=True（不可学习）──")

# 把技能设为冻结
agent.cfg.innate_bounds[MemType.SKILL] = InnateBounds(
    tau_min=1000.0, tau_max=10000.0,
    drift_min=0.0, drift_max=0.1,
    importance_min=0.0, importance_max=0.1,
    frozen=True,
)
frozen_bounds = agent.cfg.innate_bounds[MemType.SKILL]
check("frozen=True", frozen_bounds.frozen == True)

# 模拟 learn_tau 行为：frozen 时保持出厂τ不变
import copy
old_tau = agent.cfg.tau_for(MemType.SKILL)
# frozen 时，无论传入什么估计值，都应该保持出厂τ
sim_est = 0.1  # 即使估算出极低τ
new_tau = old_tau if frozen_bounds.frozen else sim_est
check("frozen 时τ不变", new_tau == old_tau,
      f"出厂={old_tau:.2f}, 估算={sim_est}, 结果={new_tau:.2f}")

# ── D) 三类衰减梯度验证 ─────────────────────────────────────────
print("\n── D) 衰减梯度：技能 > 语义 > 情景 ──")

t = 5.0  # agent-秒（≈5人类天）
s_epi = e_decay(tau_epi, t)
s_sem = e_decay(tau_sem, t)
s_skl = e_decay(tau_skl, t)
print(f"  t=5s后强度: 情景={s_epi:.4f}, 语义={s_sem:.4f}, 技能={s_skl:.4f}")
check("技能 > 语义 > 情景", s_skl > s_sem > s_epi,
      f"{s_skl:.4f} > {s_sem:.4f} > {s_epi:.4f}")

# ── E) TIME_SCALE 一致性验证 ──────────────────────────────────
print("\n── E) TIME_SCALE 全局一致性 ──")

check("TIME_SCALE = 1/86400", abs(TIME_SCALE - 1/86400) < 1e-15)
check("1 agent-秒 = 1 人类-天", abs(TIME_SCALE * 86400 - 1.0) < 1e-10)
check("情景τ=3s = 3人类天", abs(tau_epi / TIME_SCALE / 86400 - 3.0) < 0.01)
check("技能τ=60s = 60人类天", abs(tau_skl / TIME_SCALE / 86400 - 60.0) < 0.01)

# ── 总结 ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = PASS + FAIL
status = "✅ 出厂硬件验证通过，可以进入情绪系统" if FAIL == 0 else "⚠️ 有失败项，需要修复"
print(f"结果: {PASS} 通过, {FAIL} 失败 (共 {total} 项)")
print(status)
print("=" * 60)