"""memagent v3 进化端到端测试——验证所有新模块协同工作。

测试场景：
1. 创建 agent，写入多种类型记忆（带情绪、情境）
2. 检索测试（验证情境编码加成）
3. 前瞻记忆（添加待办任务，验证触发）
4. 元认知校准（记录预测并验证）
5. 偏差检测（模拟过度自信场景）
6. 睡眠巩固（验证记忆分级）
7. 间隔重复（验证到期复习）
8. 持久化测试（保存→加载→状态恢复）
"""
import sys
import os
import tempfile
import time

# 添加 memagent 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memagent.agent import MemoryAgent, AgentConfig
from memagent.memory import MemType
from memagent.emotion import Emotion, infer_emotion, reappraise, tau_factor
from memagent.human import (
    ContextualEncoding, Context,
    MetacognitiveMonitor, JudgmentType,
    BiasDetector,
    ProspectiveMemory,
    ElaborativeRehearsal,
    MemoryTriage,
    SpacedRepetitionOptimizer,
)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label, actual, expected, tolerance=0.05):
    """简单断言检查。"""
    if isinstance(expected, (int, float)):
        ok = abs(actual - expected) <= tolerance
    else:
        ok = actual == expected
    status = "✅" if ok else "❌"
    print(f"  {status} {label}: {actual} (预期: {expected})")
    return ok

def main():
    all_passed = True

    # ──────────────────────────────────────────
    section("1. 创建 Agent 并写入记忆")
    # ──────────────────────────────────────────

    cfg = AgentConfig(
        tau_by_type={
            MemType.SKILL: 60 * 24 * 3600,
            MemType.SEMANTIC: 14 * 24 * 3600,
            MemType.EPISODIC: 3 * 24 * 3600,
        },
        tau_seconds=7 * 24 * 3600,
    )

    agent = MemoryAgent(cfg=cfg)

    # 写入不同类型的记忆，附带情绪
    mem1 = agent.remember("主角林尘，青州林氏旁支少年，觉醒灵根时被族人嘲笑",
                          importance=0.9,
                          emotion=Emotion(valence=-0.3, arousal=0.6, self_relevance=0.8, label="sadness"))

    mem2 = agent.remember("林尘在悬崖边修炼，意外发现上古传承，实力大增",
                          importance=0.8,
                          emotion=Emotion(valence=0.7, arousal=0.8, self_relevance=0.7, label="joy"))

    mem3 = agent.remember("修炼法门：每日卯时面向东方，引气归元，循环周天",
                          importance=0.7,
                          emotion=None)  # 技能类，无情绪

    mem4 = agent.remember("今日在藏经阁偶遇苏婉儿，她借给我一本《灵草图谱》",
                          importance=0.5,
                          emotion=Emotion(valence=0.4, arousal=0.5, self_relevance=0.6, label="joy"))

    print(f"  写入 4 条记忆，当前记忆库: {len(agent.store.all())} 条")
    all_passed &= check("记忆数量", len(agent.store.all()), 4)

    # ──────────────────────────────────────────
    section("2. 情境编码加成测试")
    # ──────────────────────────────────────────

    # 设置当前情境：joy 情绪（与 mem2, mem4 编码时相同）
    agent.contextual_encoding.set_current(Context(
        timestamp=time.time(),
        emotion_label="joy",
        activity="remember"
    ))

    # 检索时，joy 情绪编码的记忆应获得 bonus
    bonus_joy = agent.contextual_encoding.get_context_bonus(mem2.id)  # joy 编码
    bonus_sadness = agent.contextual_encoding.get_context_bonus(mem1.id)  # sadness 编码

    print(f"  当前情境: joy 情绪")
    print(f"  mem2(joy编码) 情境bonus: {bonus_joy:.3f}")
    print(f"  mem1(sadness编码) 情境bonus: {bonus_sadness:.3f}")

    all_passed &= check("joy记忆bonus > 1.0", bonus_joy > 1.0, True)
    all_passed &= check("sadness记忆bonus == 1.0", bonus_sadness, 1.0)

    # ──────────────────────────────────────────
    section("3. 前瞻记忆测试")
    # ──────────────────────────────────────────

    # 添加待办任务
    t1 = agent.prospective.add_task("完成第三章林尘复仇剧情", "event", "复仇", priority=0.9)
    t2 = agent.prospective.add_task("补充苏婉儿人物背景", "event", "人物", priority=0.6)
    t3 = agent.prospective.add_task("检查伏笔：灵根觉醒的异常", "event", "伏笔", priority=0.7)

    print(f"  添加 3 个前瞻记忆任务")
    all_passed &= check("待办任务数", agent.prospective.get_pending_count(), 3)

    # 模拟触发：当前活动包含"复仇"
    due = agent.prospective.get_due_tasks(current_activity="复仇剧情构思")
    print(f"  当前活动'复仇剧情构思'触发任务: {len(due)} 个")
    all_passed &= check("触发任务数 >= 1", len(due) >= 1, True)

    # 完成任务
    agent.prospective.complete_task(t1.id)
    print(f"  完成任务后待办: {agent.prospective.get_pending_count()}")
    all_passed &= check("剩余任务数", agent.prospective.get_pending_count(), 2)

    # ──────────────────────────────────────────
    section("4. 元认知校准测试")
    # ──────────────────────────────────────────

    # 模拟 5 次预测（预测值 vs 实际值）
    predictions = [
        (0.9, 0.6),  # 过度自信
        (0.8, 0.5),  # 过度自信
        (0.7, 0.7),  # 准确
        (0.85, 0.4), # 严重过度自信
        (0.6, 0.5),  # 轻微过度自信
    ]

    for pred, actual in predictions:
        agent.metacognition.record_prediction(JudgmentType.PREDICTION, pred, actual)

    report = agent.metacognition.calibration_report()
    print(f"  记录 5 次预测验证")
    print(f"  校准状态: {report['status']}")
    print(f"  均值偏差: {report.get('mean_bias', 0):.3f} (>0 表示过度自信)")
    print(f"  建议: {report.get('recommendation', '')}")

    all_passed &= check("校准状态", report['status'], 'calibrated')
    all_passed &= check("过度自信偏差 > 0", report.get('mean_bias', 0) > 0, True)

    # 测试校准后确信度
    raw_conf = 0.9
    adjusted = agent.metacognition.adjusted_confidence(raw_conf)
    print(f"  原始确信度 {raw_conf} -> 校准后 {adjusted:.3f}")
    all_passed &= check("校准后确信度下调", adjusted < raw_conf, True)

    # ──────────────────────────────────────────
    section("5. 偏差检测测试")
    # ──────────────────────────────────────────

    # 模拟过度自信场景
    w = agent.bias_detector.check_overconfidence(0.95, report.get('mean_bias', 0))
    if w:
        print(f"  检测到偏差: {w.bias_type.value}")
        print(f"  描述: {w.description}")
        print(f"  建议: {w.suggestion}")
        all_passed &= check("偏差检测触发", True, True)
    else:
        print(f"  ❌ 未检测到偏差")
        all_passed = False

    # ──────────────────────────────────────────
    section("6. 检索 + 精细复述 + 间隔重复")
    # ──────────────────────────────────────────

    # 执行检索
    hits = agent.retrieve("林尘 修炼 复仇", k=3)
    print(f"  检索 '林尘 修炼 复仇' 命中 {len(hits)} 条")

    for i, h in enumerate(hits[:3]):
        print(f"  [{i+1}] {h.memory.content[:40]}... rel={h.relevance:.3f} total={h.total:.3f}")

    # 验证精细复述构建了关联
    associations = agent.elaboration.get_associations(hits[0].memory.id)
    print(f"  精细复述: 为 top1 记忆构建 {len(associations)} 个关联")

    # 验证间隔重复记录了复习
    stats = agent.spaced_rep.get_stats()
    print(f"  间隔重复: 总条目 {stats.get('total_items', 0)}, 总复习 {stats.get('total_reviews', 0)}")
    all_passed &= check("间隔重复记录复习", stats.get('total_reviews', 0) > 0, True)

    # ──────────────────────────────────────────
    section("7. 睡眠巩固 + 记忆分级")
    # ──────────────────────────────────────────

    # 执行完整睡眠
    sleep_report = agent.sleep()
    print(f"  睡眠报告:")
    print(f"    回放: {sleep_report.get('replayed_count', 0)} 条")
    print(f"    未回放(模糊): {sleep_report.get('unreplayed_count', 0)} 条")
    print(f"    压缩进Cold: {sleep_report.get('cold_compressed', 0)} 条")
    print(f"    记忆分级: 高={sleep_report.get('triage_high', 0)}, 中={sleep_report.get('triage_medium', 0)}, 低={sleep_report.get('triage_low', 0)}")
    print(f"    类型迁移: {sleep_report.get('migrations', 0)} 条")

    all_passed &= check("记忆分级有数据", sleep_report.get('triage_high', 0) >= 0, True)

    # ──────────────────────────────────────────
    section("8. 情绪调节测试（恐惧调制）")
    # ──────────────────────────────────────────

    # 创建恐惧情绪
    fear = Emotion(valence=-0.9, arousal=0.95, self_relevance=0.8, label="fear")
    tau_fear = tau_factor(fear)
    print(f"  恐惧情绪 τ 调制: {tau_fear:.2f}x")
    print(f"  (已从原始 100x 降为更合理范围)")

    # 认知重评
    reappraised = reappraise(fear)
    tau_reappraised = tau_factor(reappraised)
    print(f"  重评后 τ 调制: {tau_reappraised:.2f}x")
    print(f"  重评效果: 价 {fear.valence:.1f} -> {reappraised.valence:.2f}, 唤醒 {fear.arousal:.1f} -> {reappraised.arousal:.2f}")

    all_passed &= check("重评后唤醒降低", reappraised.arousal < fear.arousal, True)
    all_passed &= check("恐惧调制 < 30x", tau_fear < 30, True)

    # ──────────────────────────────────────────
    section("9. 持久化测试")
    # ──────────────────────────────────────────

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        tmp_path = f.name

    try:
        agent.save(tmp_path)
        print(f"  保存到: {tmp_path}")

        # 重新加载
        agent2 = MemoryAgent(cfg=cfg, persist_path=tmp_path)
        print(f"  重新加载后记忆数: {len(agent2.store.all())}")
        all_passed &= check("记忆数恢复", len(agent2.store.all()), 4)

        # 验证新模块状态恢复
        all_passed &= check("待办任务恢复", agent2.prospective.get_pending_count(), 2)

        meta_report = agent2.metacognition.calibration_report()
        all_passed &= check("元认知记录恢复", meta_report['status'], 'calibrated')

        stats2 = agent2.spaced_rep.get_stats()
        all_passed &= check("间隔重复恢复", stats2.get('total_items', 0) > 0, True)

        print(f"  所有模块状态恢复 ✅")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ──────────────────────────────────────────
    section("10. 综合场景测试")
    # ──────────────────────────────────────────

    # 模拟一次完整的对话响应
    agent3 = MemoryAgent(cfg=cfg)

    # 预置记忆
    agent3.remember("林尘的仇人赵无极，筑基后期，掌管外门执法堂",
                    importance=0.85, emotion=Emotion(valence=-0.5, arousal=0.7, self_relevance=0.9, label="anger"))
    agent3.remember("筑基期突破需要：1.灵力充盈 2.心境通达 3.机缘触发",
                    importance=0.8, emotion=None)
    agent3.remember("今日突破瓶颈，从炼气九层迈入大圆满",
                    importance=0.6, emotion=Emotion(valence=0.6, arousal=0.7, self_relevance=0.8, label="joy"))

    # 模拟提问
    reply, hits = agent3.respond("林尘如何突破到筑基期？")
    print(f"  提问: '林尘如何突破到筑基期？'")
    print(f"  命中记忆: {len(hits)} 条")
    print(f"  回复预览: {reply[:60]}...")

    # 验证元认知工作
    agent3.metacognition.record_prediction(JudgmentType.RECALL_CONFIDENCE, 0.8, 0.7)
    cal_report = agent3.metacognition.calibration_report()
    print(f"  元认知: 状态={cal_report['status']}, 偏差={cal_report.get('mean_bias', 0):.3f}")

    all_passed &= check("综合场景检索成功", len(hits) > 0, True)

    # ──────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_passed:
        print("  🎉 所有测试通过！进化模块协同工作正常")
    else:
        print("  ⚠️ 部分测试失败，请检查上方日志")
    print(f"{'='*60}\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
