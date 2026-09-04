"""REST API 端到端测试：真实起服务器 + 真实 HTTP 调用（stdlib urllib）。"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memagent.server import serve


def _post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port: int, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def main() -> int:
    port = 8399
    persist = Path(__file__).parent / "_server_test_memories.json"
    if persist.exists():
        persist.unlink()

    # 后台线程起服务器（离线模式，不碰网络 LLM）
    t = threading.Thread(
        target=serve, args=("127.0.0.1", port, str(persist), True), daemon=True
    )
    t.start()
    time.sleep(0.5)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
        if not cond:
            failures.append(name)

    # 1) health
    code, body = _get(port, "/health")
    check("GET /health", code == 200 and body["ok"] is True and body["memories"] == 0,
          f"→ {body}")

    # 2) remember
    code, body = _post(port, "/remember", {"content": "用户偏好简洁回复", "importance": 0.9})
    check("POST /remember", code == 200 and "简洁" in body["content"], f"→ {body}")

    code, body = _post(port, "/remember", {"content": "项目截止日是周五", "importance": 0.6})
    check("POST /remember #2", code == 200 and body["id"], f"→ {body}")

    # 3) remember 缺字段 → 400
    code, body = _post(port, "/remember", {})
    check("POST /remember missing field → 400", code == 400, f"→ {body}")

    # 4) retrieve
    code, body = _post(port, "/retrieve", {"query": "简洁回复 不要废话", "k": 2})
    hits = body.get("hits", [])
    check("POST /retrieve", code == 200 and len(hits) >= 1
          and "简洁" in hits[0]["content"], f"→ top={hits[0] if hits else None}")

    # 5) sleep
    code, body = _post(port, "/sleep", {})
    check("POST /sleep", code == 200 and "triage" in body, f"→ {body}")

    # 6) 持久化验证：重启后记忆还在（新 agent 读同一文件）
    from memagent import MemoryAgent
    from memagent.llm import LLMClassifier
    agent2 = MemoryAgent(persist_path=str(persist),
                         classifier=LLMClassifier(api_key=""))
    check("persistence round-trip", len(agent2.store.all()) >= 2,
          f"→ {len(agent2.store.all())} memories")

    # 7) 404
    code, body = _post(port, "/nope", {})
    check("POST /nope → 404", code == 404, f"→ {body}")

    if persist.exists():
        persist.unlink()

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL REST API TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
