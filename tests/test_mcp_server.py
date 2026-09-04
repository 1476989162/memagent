"""MCP 服务器协议级测试：直接调 lowlevel Server 的 handler 验证九个工具。"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp.types as types

from memagent.mcp_server import build_server

EXPECTED_TOOLS = [
    "memagent_export", "memagent_find", "memagent_forget",
    "memagent_recall", "memagent_remember", "memagent_retrieve",
    "memagent_sleep", "memagent_start", "memagent_stats",
]


async def main() -> int:
    persist = Path(__file__).parent / "_mcp_test_memories.json"
    if persist.exists():
        persist.unlink()
    export_path = Path(__file__).parent / "_mcp_test_agents.md"
    if export_path.exists():
        export_path.unlink()

    server = build_server(str(persist), offline=True)
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
        if not cond:
            failures.append(name)

    # tools/list
    entry = server.get_request_handler("tools/list")
    lt = await entry.handler(None, types.ListToolsRequest())
    names = sorted(t.name for t in lt.tools)
    check("tools/list", names == EXPECTED_TOOLS, f"→ {names}")

    async def call(name: str, args: dict) -> dict:
        call_entry = server.get_request_handler("tools/call")
        req = types.CallToolRequest(
            params=types.CallToolRequestParams(name=name, arguments=args))
        res = await call_entry.handler(None, req)
        if getattr(res, "is_error", False):
            raise ValueError(res.content[0].text)
        return json.loads(res.content[0].text)

    # remember
    r = await call("memagent_remember", {"content": "用户偏好简洁回复", "importance": 0.9})
    check("remember", "简洁" in r["content"] and r["mtype"] == "semantic", f"→ {r}")

    # retrieve：高置信命中
    r = await call("memagent_retrieve", {"query": "简洁回复 不要废话", "k": 2})
    check("retrieve", len(r["hits"]) >= 1 and "简洁" in r["hits"][0]["content"],
          f"→ top={r['hits'][0] if r['hits'] else None}")
    check("retrieve matched flag", r["matched"] >= 1, f"→ matched={r.get('matched')}")

    # retrieve：无关查询
    r = await call("memagent_retrieve", {"query": "火锅", "k": 2})
    check("retrieve low-confidence notice",
          r.get("matched") == 0 and "notice" in r,
          f"→ matched={r.get('matched')}, top_rel={r['hits'][0]['relevance'] if r['hits'] else None}")

    # find
    r = await call("memagent_find", {"keywords": "简洁"})
    check("find", r["count"] >= 1 and r["memories"][0]["id"], f"→ {r}")
    mem_id = r["memories"][0]["id"]

    # start
    r = await call("memagent_start", {"k": 3})
    check("start", "注入" in r["injection"] and "简洁" in r["injection"],
          f"→ topic={r['topic']}, block={r['injection'][:40]}...")
    r = await call("memagent_start", {"topic": "用户偏好", "k": 3})
    check("start topic", "用户偏好" in r["injection"], f"→ {r['topic']}")

    # export
    r = await call("memagent_export", {"path": str(export_path)})
    check("export", export_path.exists() and "简洁" in export_path.read_text(encoding="utf-8"),
          f"→ {r}")

    # recall
    agent = server._agent
    m = agent.store.add("我昨天去吃了火锅", importance=0.1)
    m.demote_to_cold("火锅聚餐（已归档）")
    r = await call("memagent_recall", {"id_prefix": m.id[:6]})
    check("recall", r["awakened"] and any("火锅" in o for o in r["originals"]),
          f"→ {r}")
    r = await call("memagent_recall", {"id_prefix": "noexist"})
    check("recall miss", r["awakened"] is False, f"→ {r}")

    # forget
    r = await call("memagent_forget", {"id": mem_id})
    check("forget", r["forgotten"] is True, f"→ {r}")
    r = await call("memagent_find", {"keywords": "简洁"})
    check("forget verified", r["count"] == 0, f"→ {r}")

    # sleep
    r = await call("memagent_sleep", {})
    check("sleep", "triage" in r, f"→ {r}")

    # stats
    r = await call("memagent_stats", {})
    check("stats", r["total"] >= 1 and "warm" in r["tiers"], f"→ {r}")

    # persist
    check("persisted", persist.exists(),
          f"→ {persist.stat().st_size if persist.exists() else 0}B")

    # unknown tool → is_error=True
    call_entry = server.get_request_handler("tools/call")
    res = await call_entry.handler(None, types.CallToolRequest(
        params=types.CallToolRequestParams(name="nope", arguments={})))
    check("unknown tool error result",
          getattr(res, "is_error", False) and "unknown tool" in res.content[0].text,
          f"→ {res.content[0].text}")

    for p in (persist, export_path):
        if p.exists():
            p.unlink()

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL MCP TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))