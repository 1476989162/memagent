"""memagent 交互式对话 CLI：直接与 agent 聊天，所有记忆/情绪/生长机制在后台运行。

用法：
    python E:/神经网络/chat.py [--persona novelist]          # 交互对话
    python E:/神经网络/chat.py --auto 10 --persona novelist  # 无头自主演化

选项：
    --persona <文本|别名>   人设（novelist/小说家 映射内置小说家人设，或任意自定义
                            文本；默认读 OPENAI_PERSONA 环境变量）。设置后 agent
                            自动接 LLM 回复生成器，且每次睡眠自动自主演化（反思记忆
                            + 联网查资料 → 沉淀新设定）。
    --auto N               无头自主演化模式：连续跑 N 轮 {自主演化(含联网搜索) + 睡眠
                            巩固}，每轮打印演化报告。睡觉前挂上即可让 agent 自己成长。

命令：
    [输入你的话]     → 与 agent 对话
    /sleep           → 触发睡眠巩固
    /memory [关键词] → 查看记忆检索结果
    /persona         → 查看人设与演化档案
    /evolve          → 手动自主演化一轮（反思记忆 + 联网 → 新设定入库）
    /write [字数]    → 基于当前人设档案续写下一章（落盘 works/<书名>/chapters/）
    /web <查询>      → 联网搜索（Bing → DuckDuckGo 备用）
    /models          → 查看 LLM 模型池状态（429 自动切换）
    /interest        → 查看兴趣排名
    /skill           → 查看技能进展
    /growth          → 查看生长摘要
    /boundary        → 查看认知边界
    /stats           → 系统总览
    /help            → 显示帮助
    /quit /exit      → 退出
"""
import argparse
import sys, time, os
sys.path.insert(0, r"E:/神经网络")

from memagent.agent import AgentConfig, MemoryAgent
from memagent.memory import MemoryStore, MemType, Tier

STORE_PATH = r"E:/神经网络/agent_memory.json"


def banner():
    print("=" * 50)
    print("  memagent 交互式对话")
    print("  输入文字与 agent 对话")
    print("  /help 查看所有命令")
    print("=" * 50)


def fmt_ts(ts: float) -> str:
    """将 agent 秒转换为可读时间。"""
    scale = 1.0 / 86400
    days = ts / scale / 86400
    if days >= 1:
        return f"{days:.1f}天"
    hours = ts / scale / 3600
    if hours >= 1:
        return f"{hours:.1f}小时"
    return f"{ts / scale:.0f}秒"


def show_stats(agent: MemoryAgent, store: MemoryStore):
    print(f"\n--- 系统总览 ---")
    hot = sum(1 for m in store.all() if m.tier == Tier.HOT)
    warm = sum(1 for m in store.all() if m.tier == Tier.WARM)
    cold = sum(1 for m in store.all() if m.tier == Tier.COLD)
    print(f"  记忆: Hot={hot}  Warm={warm}  Cold={cold}  (共 {hot+warm+cold})")
    print(f"  兴趣排名: {agent.interest.top(10)}")
    print(f"  生长步数: {agent.growth.growth_step_count}")
    print(f"  预测数: {len(agent.growth.predictions)}")
    print(f"  模式数: {len(agent.growth.patterns)}")
    print(f"  图谱节点: {len(agent.graph.nodes)}")
    print(f"  图谱边: {len(agent.graph.edges)}")
    print(f"  技能: {agent.cognition.self_summary()['skill_count']}")
    print(f"  目标: {agent.cognition.self_summary()['goal_count']}")
    # 记忆强度
    if store.all():
        imports = [m.importance for m in store.all()]
        avg_i = sum(imports) / len(imports)
        print(f"  记忆强度: 平均={avg_i:.4f}  最低={min(imports):.4f}  最高={max(imports):.4f}")


def cmd_memory(agent: MemoryAgent, kw: str):
    if kw:
        hits = agent.retrieve(kw, k=5)
        if hits:
            print(f"\n--- 检索: {kw} ---")
            for r in hits:
                print(f"  {r.memory.content}  (重要性={r.memory.importance:.3f}, 类型={r.memory.mtype.value})")
        else:
            print(f"  未找到相关内容")
    else:
        print("  用法: /memory [关键词]")


def run_auto(agent: MemoryAgent, cycles: int):
    """无头自主演化 + 创作：每轮 {演化(联网) + 写一章 + 睡眠巩固}，打印报告。"""
    print(f"\n=== 自主演化 + 创作模式：{cycles} 轮（每轮 = 反思 + 联网研究 + 写一章 + 睡眠巩固）===\n")
    for i in range(1, cycles + 1):
        print(f"--- 第 {i}/{cycles} 轮 ---")
        ev = agent.evolve(with_web=True)
        if not ev.get("ok"):
            print(f"  演化未执行：{ev.get('reason')}（配 OPENAI_API_KEY 后可用）")
        else:
            print(f"  研究主题: {ev['query']} | 联网资料: {ev['web_n']} 条 | 新增设定: {len(ev['added'])} 条")
            for s in ev["added"]:
                print(f"    • {s}")
        w = agent.write_chapter()
        if w.get("ok"):
            print(f"  写作: 《{w['title']}》第 {w['chapter']} 章（{w['words']} 字）→ {w['path']}")
        else:
            print(f"  写作未完成：{w.get('reason')}")
        sr = agent.sleep()
        print(f"  睡眠: 回放 {sr.get('replayed_count', 0)} 条, 冷压缩 {sr.get('cold_compressed', 0)}"
              f", 演化入库 {len(sr.get('evolved', []))} 条\n")
    print("=== 自主演化 + 创作结束 ===")
    sheet = agent.persona_sheet() or ""
    print(f"最终设定档案（{len(sheet.splitlines())} 条）:")
    for line in sheet.splitlines():
        print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(description="memagent 交互式对话")
    parser.add_argument("--persona", default=os.environ.get("OPENAI_PERSONA"),
                        help="人设（novelist/小说家 或自定义文本）")
    parser.add_argument("--auto", type=int, default=0,
                        help="无头自主演化轮数（0=交互模式）；每轮 = 演化(联网) + 写一章 + 睡眠巩固")
    args = parser.parse_args()

    store = MemoryStore(path=STORE_PATH) if os.path.exists(STORE_PATH) else MemoryStore()
    store.path = STORE_PATH  # 固定落盘路径（退出时保存演化成果）
    # 配了人设 → 睡眠时自动自主演化（交互与无头模式都生效）
    agent = MemoryAgent(store=store, persona=args.persona,
                        cfg=AgentConfig(evolve_on_sleep=bool(args.persona)))

    if args.auto > 0:
        try:
            run_auto(agent, args.auto)
        finally:
            agent.save()
            print(f"\n记忆已保存 → {STORE_PATH}")
        return

    banner()
    if args.persona:
        print("\n[agent] 你好，我是 memagent。我已载入人设：")
        print(f"       {args.persona}")
        print("       我还会记住你告诉我的作品设定，让创作人设随设定自主演化。")
        print("       输入 /persona 查看当前档案，输入 /help 查看全部命令。\n")
    else:
        # 首次启动欢迎语
        print("\n[agent] 你好，我是 memagent。你可以直接跟我聊天，")
        print("       我会记住你说的话，并且根据我们的对话学习成长。")
        print("       输入 /help 查看可用命令。\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd = line[1:].strip().lower()
            if cmd in ("quit", "exit"):
                agent.save()
                print("记忆已保存，再见。")
                break
            elif cmd == "help":
                print("\n命令列表:")
                print("  /memory [关键词]  → 检索记忆")
                print("  /sleep            → 触发睡眠巩固（配人设时自动自主演化）")
                print("  /persona          → 人设与演化档案")
                print("  /evolve           → 手动自主演化一轮（反思+联网→新设定）")
                print("  /write [字数]     → 续写下一章（落盘 works/<书名>/chapters/）")
                print("  /web <查询>       → 联网搜索")
                print("  /models           → LLM 模型池状态（429 自动切换）")
                print("  /interest         → 兴趣排名")
                print("  /skill            → 技能进展")
                print("  /growth           → 生长摘要")
                print("  /boundary         → 认知边界")
                print("  /stats            → 系统总览")
                print("  /quit /exit       → 退出")
            elif cmd.startswith("memory"):
                cmd_memory(agent, cmd[6:].strip())
            elif cmd == "sleep":
                sr = agent.sleep()
                print(f"\n--- 睡眠报告 ---")
                print(f"  回放: {sr.get('replayed_count',0)} 条")
                if sr.get("replayed"):
                    for r in sr["replayed"]:
                        print(f"    • {r[:50]}")
                print(f"  冷压缩: {sr.get('cold_compressed',0)}")
            elif cmd == "interest":
                print(f"\n--- 兴趣排名 ---")
                for topic, val in agent.interest.top(10):
                    print(f"  {topic}: {val:.4f}")
            elif cmd == "skill":
                ss = agent.cognition.self_summary()
                print(f"\n--- 技能 ---")
                for sk in ss.get("skills", []):
                    print(f"  {sk['name']}: mastery={sk['mastery']:.4f}  练习={sk['practices']}  成功率={sk['success_rate']:.2%}")
                if not ss.get("skills"):
                    print("  (尚未注册任何技能)")
            elif cmd == "growth":
                gs = agent.growth.growth_summary()
                print(f"\n--- 生长摘要 ---")
                print(f"  总步数: {gs['total_steps']}")
                print(f"  预测数: {gs['predictions']}")
                print(f"  模式数: {gs['patterns']}")
                print(f"  概念数: {gs['concept_count']}")
                for p in gs.get("top_predictions", [])[:3]:
                    print(f"    预测: 如果{p.get('trigger','')}则{p.get('expected','')} (置信{p.get('confidence',0):.2f})")
                for p in gs.get("top_patterns", [])[:3]:
                    print(f"    模式: {p.get('antecedent','')}→{p.get('consequent','')} (置信{p.get('confidence',0):.2f})")
            elif cmd == "boundary":
                bd = agent.cognition.knowledge_boundary(interest_getter=agent.interest.get)
                print(f"\n--- 认知边界 ---")
                for b in bd:
                    print(f"  {b['topic']} (兴趣{b['interest']:.2f}, 技能{b['skill_level']:.2f})")
                    for k in b.get("known", []):
                        print(f"    ✓ {k}")
                    for u in b.get("unknown", []):
                        print(f"    ? {u}")
                if not bd:
                    print("  (当前没有可报告的知识边界)")
            elif cmd.startswith("write"):
                parts = cmd.split()
                try:
                    words = int(parts[1]) if len(parts) > 1 else None
                except ValueError:
                    words = None
                print("写作中（基于当前人设档案续写下一章）…")
                w = agent.write_chapter(target_words=words)
                if not w.get("ok"):
                    print(f"  未完成：{w.get('reason')}")
                else:
                    print(f"  《{w['title']}》第 {w['chapter']} 章已写完（{w['words']} 字）")
                    print(f"  存档: {w['path']}")
                    print(f"  开头: {w['preview']}…")
            elif cmd == "evolve":
                print("自主演化中（反思记忆 + 联网查资料 → 沉淀新设定）…")
                ev = agent.evolve()
                if not ev.get("ok"):
                    print(f"  未执行：{ev.get('reason')}（配 OPENAI_API_KEY 后可用）")
                else:
                    print(f"  研究主题: {ev['query']}（联网资料 {ev['web_n']} 条）")
                    if ev["added"]:
                        print(f"  新增设定 {len(ev['added'])} 条:")
                        for s in ev["added"]:
                            print(f"    • {s}")
                    else:
                        print("  没有可吸收的新设定")
                    sheet = agent.persona_sheet()
                    print(f"  当前档案 {len(sheet.splitlines()) if sheet else 0} 条设定")
            elif cmd.startswith("web"):
                q = cmd[3:].strip()
                if not q:
                    print("  用法: /web <查询>")
                else:
                    from memagent.websearch import search_web

                    results = search_web(q, n=5)
                    if not results:
                        print("  （联网搜索无结果或不可用）")
                    for r in results[:5]:
                        print(f"  • {r['title']}")
                        print(f"    {r['url']}")
                        if r.get("snippet"):
                            print(f"    {r['snippet'][:140]}")
            elif cmd == "persona":
                if agent.persona:
                    print(f"\n--- 人设（{agent.persona[:24]}{'…' if len(agent.persona) > 24 else ''}）---")
                else:
                    print("\n--- 人设 ---（未设置）")
                sheet = agent.persona_sheet()
                if sheet:
                    print("你的身份档案（随设定记忆自主演化）:")
                    for line in sheet.splitlines():
                        print(f"  {line}")
                else:
                    print("  （还没有设定记忆——对话里说「记住设定：…」并配合 remember_setting 入库）")
            elif cmd == "models":
                print("\n--- LLM 模型池（429 自动切换）---")
                if agent.responder is not None and agent.responder.available:
                    st = agent.responder.pool_status()
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
            elif cmd == "stats":
                show_stats(agent, store)
            else:
                print(f"  未知命令: {cmd}")
        else:
            # 直接对话
            reply, hits = agent.respond(line)
            if hits:
                print(f"  (回忆到 {len(hits)} 条相关记忆)")
            print(f"  [agent] {reply}")


if __name__ == "__main__":
    main()
