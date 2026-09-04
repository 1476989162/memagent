"""memagent REST API —— 纯 stdlib http.server，零新依赖。

给不用 Python 的调用方（Node/Go/浏览器/其他进程）暴露三个核心操作：

    POST /remember  {"content": "...", "importance": 0.8, "kind": "fact"}
    POST /retrieve  {"query": "...", "k": 3}
    POST /sleep     {"duration": null}
    GET  /health    → {"ok": true, "version": "0.3.2", "memories": N}

启动：
    python -m memagent.server --port 8399 --persist memories.json
    memagent-server --port 8399          # 安装后可用

设计取舍（ponytail: 单进程单 agent，够覆盖 SDK 场景；多租户/鉴权等
真实需求出现时再换 ASGI + token）。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .agent import AgentConfig, MemoryAgent
from .llm import LLMClassifier


def _make_agent(persist_path: str | None, offline: bool) -> MemoryAgent:
    kwargs: dict = {"persist_path": persist_path} if persist_path else {}
    classifier = LLMClassifier(api_key="") if offline else None
    return MemoryAgent(classifier=classifier, **kwargs)


def _save_if_persistent(agent: MemoryAgent) -> None:
    """配置了持久化路径时，在变更型请求（remember/sleep）后自动落盘。"""
    if agent.store.path:
        agent.save()


def make_handler(agent: MemoryAgent) -> type:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server 约定)
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "version": __version__,
                                 "memories": len(agent.store.all())})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 1_000_000:
                    self._send(413, {"error": "body too large"})
                    return
                req = json.loads(self.rfile.read(length) or b"{}")
                if self.path.rstrip("/") == "/remember":
                    mem = agent.remember(
                        str(req["content"]),
                        importance=float(req.get("importance", 0.5)),
                        kind=str(req.get("kind", "fact")),
                    )
                    _save_if_persistent(agent)
                    self._send(200, {"id": mem.id, "content": mem.content,
                                     "importance": mem.importance,
                                     "mtype": mem.mtype.value})
                elif self.path.rstrip("/") == "/retrieve":
                    hits = agent.retrieve(str(req["query"]), k=int(req.get("k", 3)))
                    self._send(200, {"hits": [
                        {"id": h.memory.id, "content": h.memory.content,
                         "score": round(h.total, 4), "mtype": h.memory.mtype.value}
                        for h in hits
                    ]})
                elif self.path.rstrip("/") == "/sleep":
                    report = agent.sleep(req.get("duration"))
                    _save_if_persistent(agent)
                    self._send(200, {"replayed": report.get("replayed_count", 0),
                                     "fogged": report.get("unreplayed_count", 0),
                                     "cold_compressed": report.get("cold_compressed", 0),
                                     "triage": {"high": report.get("triage_high", 0),
                                                "medium": report.get("triage_medium", 0),
                                                "low": report.get("triage_low", 0)}})
                else:
                    self._send(404, {"error": "not found"})
            except KeyError as e:
                self._send(400, {"error": f"missing field: {e}"})
            except (ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # 防单请求异常拖垮服务线程
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt: str, *args) -> None:  # 静默默认访问日志
            pass

    return Handler


def serve(host: str, port: int, persist_path: str | None, offline: bool) -> None:
    agent = _make_agent(persist_path, offline)
    httpd = ThreadingHTTPServer((host, port), make_handler(agent))
    print(f"memagent {__version__} serving on http://{host}:{port} "
          f"(persist={persist_path or 'memory-only'}, offline={offline})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="memagent REST API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument("--persist", default=None, help="memory JSON path (omit = memory-only)")
    parser.add_argument("--offline", action="store_true",
                        help="force keyword classification, no LLM network calls")
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.persist, args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
