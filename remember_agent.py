"""把 memagent 作为长期记忆层，接到真实 LLM agent 的演示。

数据流（记忆层 → 编排层 → LLM 层）：

    1. 用户输入 → 记忆层 `MemoryAgent.retrieve()` 检索相关记忆
       （遗忘曲线评分 × 语义相似度 × 查询同义扩展）；
    2. 编排层把命中记忆注入 system prompt（带类型与强度）；
    3. LLM（OpenAI 兼容，LLMResponder）基于记忆回答；无相关记忆时直接回答；
    4. 对话写入记忆层（持久化到 memories_agent.json）——下次提问自动注入。

记忆是"活的"：被反复检索的记忆越来越强（测试效应）、长期不用会衰减；
`--remember` 写入的事实（姓名、技术栈、偏好）跨对话长期保留。

用法：
    python remember_agent.py                          # 交互式对话（每轮注入相关记忆）
    python remember_agent.py --remember "我叫小林，用 Python"   # 手动写入事实记忆（可多次）
    python remember_agent.py --show-memories          # 查看记忆层内容
    python remember_agent.py --reset                  # 清空记忆，重新开始
    python remember_agent.py --demo                   # 自动演示：事实→跨轮注入→基于记忆回答

LLM 配置：OPENAI_API_KEY（可选 OPENAI_BASE_URL / OPENAI_MODEL），
兼容 OpenAI / DeepSeek / Moonshot / Ollama。未配置 key 时降级为
"记忆层演示"——仍检索并显示注入的记忆，只是不生成 LLM 回复。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

from memagent import MemoryAgent
from memagent.responder import LLMResponder
from memagent.synonyms import is_short_query


def build_responder(post=None) -> LLMResponder:
    """LLM 客户端：读 OPENAI_* 环境变量；post 可注入（测试用）。"""
    return LLMResponder(post=post)


def injected_from(hits, k: int = 3) -> list[tuple[str, str, float]]:
    """把检索命中转成注入格式 [(内容, 类型, 强度)]，弱相关（total≤RELEVANT_TOTAL）剔除。

    Cold 命中（via_summary）注入**摘要文本**而非深藏 content——命中词在摘要里，
    content 可能不含（与核心 _template_reply / _generate_reply 行为一致）。
    """
    from memagent.agent import RELEVANT_TOTAL

    return [
        (h.memory.summary or h.memory.content, h.memory.mtype.value, h.strength)
        for h in hits[:k]
        if h.total > RELEVANT_TOTAL
    ]


class RememberAgent:
    """memagent 记忆层 + OpenAI 兼容 LLM 的编排。"""

    def __init__(self, persist_path: str = "memories_agent.json", responder=None):
        self.memory = MemoryAgent(persist_path=persist_path)
        self.responder = responder if responder is not None else build_responder()

    def remember(self, text: str):
        """写入一条事实记忆（--remember 接口）。"""
        return self.memory.remember(text)

    def chat(self, text: str) -> tuple[str | None, list[tuple[str, str, float]]]:
        """一轮对话：检索相关记忆 → 注入 prompt → LLM 回复 → 对话写入记忆。

        返回 (回复或 None, 注入的记忆列表)。responder 不可用（未配 key）时
        返回 reply=None，调用方降级展示注入内容。
        """
        hits = self.memory.retrieve(text, k=3)
        injected = injected_from(hits)
        # 短查询重排已由核心 retrieve() 单点完成（含 rel×强度 组内排序）——
        # 此处只按同一判据打印加长提示，不再重排（避免与核心排序不一致）
        if self.memory.cfg.rerank_short_query and is_short_query(
            text, self.memory.cfg.rerank_short_len,
        ):
            print(f"（提示：查询「{text}」仅 {len(text.strip())} 字，已做子串优先重排；"
                  f"建议加长以获得更精确检索）")
        reply = None
        if self.responder.available:
            try:
                reply = self.responder.respond(text, memories=injected or None)
            except Exception as exc:
                # Provider errors must not discard this turn or stop memory upkeep.
                print(f"（LLM 暂时不可用：{exc}；本轮仅使用记忆层）")
        self.memory.remember(f"用户说：{text}", kind="turn")  # 对话流水也进记忆
        return reply, injected

    def show_memories(self) -> None:
        self.memory._print_memories()

    def save(self) -> None:
        self.memory.save()

    def sleep(self) -> dict:
        report = self.memory.sleep()
        self.save()
        return report

    def spontaneous_recall(self):
        return self.memory.spontaneous_recall()

    def observe(self) -> int:
        return self.memory._observe()

    def learn(self) -> tuple[dict, dict]:
        reports = self.memory.learn_tau(), self.memory.learn_plasticity()
        self.save()
        return reports

    def stats(self) -> str:
        return self.memory.stats()

    def export_agents_md(
        self,
        path: str = "AGENTS.md",
        min_importance: float = 0.25,
        min_access: int = 2,
        max_items: int = 30,
        dual: bool = False,
    ) -> str | tuple[str, str]:
        """把记忆层"固化知识"导出成 AGENTS.md 风格指令文件（第三种嵌入模式）。

        固化知识 = 重要（importance ≥ min_importance）或被反复检索确认
        （access_count ≥ min_access）的事实/决策/偏好；对话流水（turn）排除。
        按类型分组（semantic=事实偏好 / skill=经验方法 / episodic=事件），
        每条带重要性与检索次数——被反复确认的排前面。

        文件名可改（如 CLAUDE.md / project.md），内容为 markdown 指令，
        适配习惯全量加载指令文件的 agent（Codex / Claude Code 等）。

        dual=True 时用同一份内容同时写入 AGENTS.md 与 CLAUDE.md（逐字节一致，
        保持同步——Codex 加载前者、Claude Code 加载后者），返回两个路径；
        否则只写 path 并返回该路径。
        """
        mems = [m for m in self.memory.store.all() if m.kind != "turn"]
        kept = sorted(
            (m for m in mems if m.importance >= min_importance or m.access_count >= min_access),
            key=lambda m: (-m.importance, -m.access_count),
        )[:max_items]
        groups: dict[str, list] = {"semantic": [], "skill": [], "episodic": []}
        for m in kept:
            groups.setdefault(m.mtype.value, []).append(m)
        titles = {
            "semantic": "项目事实与用户偏好（重要度与检索次数标注，越靠前越可靠）",
            "skill": "经验与方法（技能类：被反复验证的做法）",
            "episodic": "历史事件（情景类：随遗忘可能过时）",
        }
        lines = [
            "# 项目记忆（由 memagent 自动生成）",
            "",
            f"生成时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}；",
            f"来源：{self.memory.store.path or 'memories_agent.json'}（共 {len(kept)} 条固化知识）。",
        "本文件供支持 AGENTS.md / CLAUDE.md 风格的 agent 在会话开始时加载。",
        "重新运行 `python remember_agent.py --export-agents-md` 可同时刷新",
        "AGENTS.md 与 CLAUDE.md（内容同步）。",
        "",
    ]
        for mtype in ("semantic", "skill", "episodic"):
            items = groups.get(mtype, [])
            if not items:
                continue
            lines.append(f"## {mtype} — {titles[mtype]}")
            lines.append("")
            for m in items:
                lines.append(
                    f"- {m.content}（重要 {m.importance:.2f} · 检索 {m.access_count} 次）"
                )
            lines.append("")
        lines.append("<!-- 由 memagent 记忆层自动生成，重新运行 --export-agents-md 更新 -->")
        text = "\n".join(lines)
        if dual:
            targets = ("AGENTS.md", "CLAUDE.md")
            for t in targets:
                with open(t, "w", encoding="utf-8") as f:
                    f.write(text)
            return targets
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


# ---------- 交互与演示 ----------

HELP = """可用命令：
  /memories             查看记忆层内容
  /remember <文本>      写入一条事实记忆（跨对话保留）
  /show-injection       查看本轮实际注入的记忆
  /stats /observe       查看状态 / 记录一轮观测
  /sleep /mind /learn   睡眠巩固 / 自发回忆 / 学习参数
  /save /quit           保存 / 退出
  其他输入              作为对话内容（每轮自动注入相关记忆）"""


def interactive(agent: RememberAgent) -> None:
    from memagent.cli import enable_utf8

    enable_utf8()
    llm_state = "已启用" if agent.responder.available else "未启用（仅演示记忆层注入）"
    print(f"记忆增强 Agent 已启动（memagent 记忆层 · LLM {llm_state}）。输入 /help 查看命令。")
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not text:
            continue
        if text == "/quit":
            agent.save()
            print("记忆已保存，再见！")
            break
        if text == "/help":
            print(HELP)
            continue
        if text == "/memories":
            agent.show_memories()
            continue
        if text == "/stats":
            print(agent.stats())
            continue
        if text == "/observe":
            print(f"已观测一轮（新采样 {agent.observe()} 条）")
            continue
        if text == "/sleep":
            report = agent.sleep()
            print(
                f"睡眠巩固完成：回放 {report['replayed_count']} 条，"
                f"压缩 {report['cold_compressed']} 条，类型迁移 {report['migrations']} 条"
            )
            continue
        if text == "/mind":
            memory = agent.spontaneous_recall()
            if memory is None:
                print("（记忆库为空，暂时没有自发回忆）")
            else:
                print(f"心游：突然想起「{memory.content}」")
            agent.save()
            continue
        if text == "/learn":
            tau_report, plasticity_report = agent.learn()
            print(f"τ 学习：更新 {len(tau_report['updated'])} 项，跳过 {len(tau_report['skipped'])} 项")
            print(f"可塑性学习：更新 {len(plasticity_report['updated'])} 项，跳过 {len(plasticity_report['skipped'])} 项")
            continue
        if text == "/show-injection":
            print("（注入发生在每轮对话中，直接对话即可看到）")
            continue
        if text.startswith("/remember"):
            parts = text.split(maxsplit=1)
            if len(parts) == 2:
                m = agent.remember(parts[1])
                print(f"已写入记忆 [{m.mtype.value}] {m.content}")
            else:
                print("用法：/remember <文本>")
            continue
        if text == "/save":
            agent.save()
            print("已保存。")
            continue
        reply, injected = agent.chat(text)
        for c, mt, s in injected:
            print(f"  注入[{mt}] {c}（强度 {s:.2f}）")
        if reply is not None:
            print(f"Agent> {reply}")
        else:
            if agent.responder.available:
                print("（本轮没有生成 LLM 回复；记忆已经保存，可继续对话）")
            else:
                print("（未配置 OPENAI_API_KEY，仅演示记忆层注入——设置后可生成回复）")


def demo(agent: RememberAgent) -> None:
    """自动演示：写入事实 → 跨轮注入 → 基于记忆回答 → 无相关记忆直接回答。"""
    from memagent.cli import enable_utf8

    enable_utf8()
    steps = [
        ("fact", "我叫小林，是一名 Python 程序员，项目用 FastAPI"),
        ("fact", "用户偏好：代码注释用中文"),
        ("ask", "我的技术栈是什么？"),
        ("ask", "代码注释应该用什么语言写？"),
        ("ask", "今天天气怎么样？"),
    ]
    for tag, text in steps:
        if tag == "fact":
            m = agent.remember(text)
            print(f"[写入] {text} → {m.mtype.value}（置信 {m.mtype_confidence:.2f}）")
        else:
            reply, injected = agent.chat(text)
            print(f"\n[提问] {text}")
            if injected:
                for c, mt, s in injected:
                    print(f"  注入[{mt}] {c}（强度 {s:.2f}）")
            else:
                print("  注入：无相关记忆")
            if reply is not None:
                print(f"  Agent> {reply}")
            else:
                if agent.responder.available:
                    print("  （本轮未生成 LLM 回复——可能是限流或服务暂时不可用，记忆层照常工作）")
                else:
                    print("  （未配置 OPENAI_API_KEY——记忆层照常工作，仅缺 LLM 回复）")
    print("\n跨轮记忆演示完成：事实一次写入、每轮自动注入、频繁检索的记忆强度在上升。")


def main(argv: list[str] | None = None) -> int:
    from memagent.cli import enable_utf8

    enable_utf8()
    parser = argparse.ArgumentParser(
        description="memagent 记忆层 + 真实 LLM agent（检索→注入→回复→写入）",
    )
    parser.add_argument("--remember", action="append", default=[], metavar="文本",
                        help="写入一条事实记忆（可多次指定）")
    parser.add_argument("--show-memories", action="store_true", help="查看记忆层内容后退出")
    parser.add_argument("--reset", action="store_true", help="清空记忆后退出")
    parser.add_argument("--demo", action="store_true", help="自动演示跨轮记忆注入")
    parser.add_argument("--persist", default="memories_agent.json", help="记忆持久化文件")
    parser.add_argument("--export-agents-md", nargs="?", const="__BOTH__", metavar="文件",
                        help="把固化知识导出成 AGENTS.md/CLAUDE.md 风格指令文件；不带文件名时同时生成两者（内容同步）")
    args = parser.parse_args(argv)

    if args.reset:
        for p in (args.persist,):
            if os.path.exists(p):
                os.remove(p)
        print(f"已清空记忆（{args.persist}）")
        return 0

    agent = RememberAgent(persist_path=args.persist)
    if args.export_agents_md is not None:
        kwargs = dict(
            min_importance=float(getattr(args, "min_importance", 0.25) or 0.25),
            min_access=int(getattr(args, "min_access", 2) or 2),
        )
        if args.export_agents_md == "__BOTH__":
            exports = agent.export_agents_md(dual=True, **kwargs)
            print(f"已导出固化知识 → {exports[0]} + {exports[1]}（内容同步，供 Codex / Claude Code 等加载）")
        else:
            p = agent.export_agents_md(path=args.export_agents_md, **kwargs)
            print(f"已导出固化知识 → {p}（供 Codex / Claude Code 等加载；重新运行可更新）")
    if args.remember:
        for text in args.remember:
            m = agent.remember(text)
            print(f"已写入记忆 [{m.mtype.value}] {m.content}")
        agent.save()
    if args.show_memories:
        agent.show_memories()
    if args.demo:
        demo(agent)
        agent.save()
        return 0
    if args.remember or args.show_memories:
        return 0
    interactive(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
