"""Installed ``memagent`` and ``python -m memagent`` entry point."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from . import __version__
from .agent import AgentConfig, MemoryAgent
from .cli import enable_utf8
from .memory import MemoryStore, StoreCorruptionError


def _check(path: str) -> int:
    target = Path(path).resolve()
    try:
        store = MemoryStore(path=str(target))
    except StoreCorruptionError as exc:
        print(f"[FAIL] persistence: {exc}")
        return 1
    print(f"[OK] memagent {__version__} / Python {platform.python_version()}")
    print(f"[OK] persistence: {target} ({len(store)} memories)")
    print(f"[OK] writable parent: {os.access(target.parent, os.W_OK)}")
    backup = target.with_suffix(target.suffix + ".bak")
    print(f"[INFO] backup: {backup if backup.exists() else 'created after first update'}")
    return 0


def _one_shot_agent(args) -> MemoryAgent:
    return MemoryAgent(persist_path=args.persist,
                       cfg=AgentConfig(evolve_on_sleep=False))


def _inject(args) -> int:
    from .instructions import build_injection_md

    agent = _one_shot_agent(args)
    block = build_injection_md(agent, topic=args.inject or None, k=args.k)
    print(block)
    return 0


def _sleep_once(args) -> int:
    import json as _json

    agent = _one_shot_agent(args)
    report = agent.sleep()
    agent.save()
    keep = ("replayed_count", "unreplayed_count", "cold_compressed",
            "migrations", "triage_high")
    print(_json.dumps({k: report.get(k) for k in keep}, ensure_ascii=False))
    return 0


_DISTILL_PROMPT = (
    "你是记忆沉淀助手。从下面的开发会话记录中提炼值得长期记住的关键结论："
    "技术决策、踩坑教训、用户偏好、项目约定。忽略闲聊与过程性内容。\n"
    "只输出 JSON 数组，每项 {\"content\": \"一句话结论(≤80字)\", "
    "\"importance\": 0.4~0.9}，最多 8 条；没有值得记的就输出 []。\n\n会话记录：\n"
)


def _distill_session(args) -> int:
    """把会话记录交给 LLM 提炼决策并入库；无 key 或失败时静默跳过。"""
    import json as _json
    import re as _re
    import urllib.request

    turns = _json.loads(Path(args.distill_session).read_text(encoding="utf-8"))
    if not isinstance(turns, list) or not turns:
        return 0
    transcript = "\n".join(
        f"{'用户' if t.get('role') == 'user' else '助手'}：{str(t.get('content', ''))[:400]}"
        for t in turns[-40:]
    )
    base = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY") or ""
    model = os.environ.get("OPENAI_MODEL") or ""
    if not (base and key and model):
        print("[]")  # 无 LLM 配置：静默跳过（婴儿原则，不报错打扰）
        return 0
    payload = _json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": _DISTILL_PROMPT + transcript}],
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            reply = _json.loads(resp.read().decode("utf-8"))
        text = reply["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[] # distill skipped: {type(e).__name__}")
        return 0
    m = _re.search(r"\[.*\]", text, _re.S)
    if not m:
        print("[]")
        return 0
    agent = _one_shot_agent(args)
    added = 0
    try:
        for item in _json.loads(m.group(0))[:8]:
            content = str(item.get("content", "")).strip()
            if len(content) < 6:
                continue
            imp = min(0.9, max(0.4, float(item.get("importance", 0.5))))
            mem = agent.remember(content, kind="fact", importance=imp)
            added += 1
            print(f"+ {mem.id[:8]} [{imp:.2f}] {content[:50]}")
    except (ValueError, TypeError):
        pass
    if added:
        agent.save()
    print(f"# distilled {added}")
    return 0


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="Local-first layered memory agent")
    parser.add_argument("--persist", default="memories.json", help="memory JSON path")
    parser.add_argument("--persona", default=os.environ.get("OPENAI_PERSONA") or None)
    parser.add_argument("--check", action="store_true", help="validate runtime and persistence")
    parser.add_argument("--migrate-work", nargs=2, metavar=("OLD", "NEW"),
                        help="move pre-title work safely without overwriting chapters")
    parser.add_argument("--works-dir", default="works", help="work root for --migrate-work")
    # --- 一次性命令（opencode 插件 / 脚本调用，不进交互 REPL） ---
    parser.add_argument("--inject", nargs="?", const="", default=None, metavar="TOPIC",
                        help="打印开工注入块（可选主题），配合 --k")
    parser.add_argument("--k", type=int, default=5, help="注入条数（默认 5）")
    parser.add_argument("--sleep-once", action="store_true",
                        help="执行一次睡眠巩固并落盘")
    parser.add_argument("--distill-session", metavar="JSON_FILE",
                        help="会话记录 JSON 提炼决策入库（需 OPENAI_* 环境变量）")
    parser.add_argument("--version", action="version", version=f"memagent {__version__}")
    args = parser.parse_args(argv)
    if args.inject is not None:
        return _inject(args)
    if args.sleep_once:
        return _sleep_once(args)
    if args.distill_session:
        return _distill_session(args)
    if args.check:
        return _check(args.persist)
    if args.migrate_work:
        from .architecture import migrate_legacy_work

        report = migrate_legacy_work(
            Path(args.works_dir).resolve(), args.migrate_work[0], args.migrate_work[1]
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0
    MemoryAgent(
        persist_path=args.persist,
        persona=args.persona,
        cfg=AgentConfig(evolve_on_sleep=bool(args.persona)),
    ).cli_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
