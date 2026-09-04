"""memagent MCP 服务器 —— 把记忆系统暴露为 MCP 工具，供 Hermes 等 Agent 原生调用。

用官方 mcp SDK 的 lowlevel Server（mcp 2.x: add_request_handler 注册
tools/list 和 tools/call）；memagent 本体保持零依赖不变——
`pip install memagent-local[mcp]` 才需要 mcp SDK。

注册（Hermes）：
    hermes mcp add memagent --command <python> --args -m memagent.mcp_server --args --persist <path>

工具（与 CLI/交互模式对齐，不再是只露一角的薄面）：
    memagent_remember(content, importance?, kind?)   写入记忆
    memagent_retrieve(query, k?)                     检索记忆（带置信标注：低置信命中明确说"查不到"）
    memagent_forget(id)                              彻底删除一条记忆（CLI /forget）
    memagent_recall(id_prefix)                       唤醒 Cold 摘要 → 完整记忆（CLI /recall）
    memagent_find(keywords)                          关键词定位记忆（获取 id 供 forget/recall 用）
    memagent_start(topic?, k?)                       开工注入：按主题取相关决策组成上下文块
    memagent_export(path?, dual?)                    导出 AGENTS.md（dual=True 同步 CLAUDE.md）
    memagent_sleep(duration?)                        睡眠巩固
    memagent_stats()                                 记忆库统计
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server

# 检索置信阈值：relevance 低于该值视为"无高置信命中"。符号哈希嵌入下
# 无关文本的相似度回到 0 附近，该阈值能把"真查不到"诚实暴露给调用方，
# 而不是返回一堆碰撞噪声让 agent 硬编故事。
REL_CONFIDENT = 0.25


def _text(payload: dict) -> types.CallToolResult:
    return types.CallToolResult(content=[
        types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))
    ])


def build_server(persist_path: str | None, offline: bool,
                 embedder=None) -> Server:
    from memagent import MemoryAgent, Tier, __version__
    from memagent.embedding import set_embedder
    from memagent.instructions import (build_injection_md, build_tips,
                                       clip_content, export_agents_md)
    from memagent.llm import LLMClassifier
    from memagent.memory import ConcurrentWriteError

    if embedder is not None:
        set_embedder(embedder)

    kwargs: dict = {"persist_path": persist_path} if persist_path else {}
    agent = MemoryAgent(
        classifier=LLMClassifier(api_key="") if offline else None,
        **kwargs,
    )

    # 脱敏诊断：只记调用/耗时/异常类型，零记忆内容（见 diagnostics.py）
    from .diagnostics import Diagnostics

    diag = Diagnostics(agent.store.meta)
    try:
        from .embedding import embedding_dim

        _dim = embedding_dim()
    except Exception:
        _dim = 0
    diag.record_env(embedder_name=("semantic" if embedder is not None else "hash"),
                    embed_dim=_dim, version=__version__)

    last_save = [0.0]
    SAVE_THROTTLE_S = 5.0  # retrieve 的测试效应/再巩固节流落盘，避免每次全量写

    def _save(force: bool = False) -> None:
        if not agent.store.path:
            return
        now = time.time()
        if not force and now - last_save[0] < SAVE_THROTTLE_S:
            return
        try:
            agent.save()
        except ConcurrentWriteError as e:
            # 共享记忆库场景（autonomous_coder / 另一个 opencode 会话）：
            # 外部进程写过磁盘后签名失配，不处理会让本次会话的持久化
            # 静默死亡——重载磁盘状态恢复保存链。代价：自上次成功保存
            # 以来的内存增量（几秒内的检索强化/最新一条 remember）被
            # 磁盘版本覆盖，冲突罕见，可接受。
            print("[memagent-mcp] 记忆库被外部进程写入，已重载磁盘状态",
                  file=sys.stderr)
            diag.note("save_conflicts")
            agent.store.load()
        last_save[0] = now

    # --- tool definitions ---
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="memagent_remember",
                description=("写入一条记忆。importance: 0~1，≥0.8 冻结为核心记忆"
                             "（永不遗忘、不再巩固改写）。铁律/偏好用 0.9+，普通事实 0.5，闲聊 0.3。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "记忆内容"},
                        "importance": {"type": "number", "default": 0.5},
                        "kind": {"type": "string", "default": "fact",
                                 "description": "fact|setting|skill|turn"},
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="memagent_retrieve",
                description=("检索记忆：语义相似度×遗忘曲线强度×情境加成排序；"
                             "命中会强化该记忆（测试效应）。回答用户问题前先查这里。"
                             "注意：短查询请尽量补全成完整句子；返回的 relevance 低于 "
                             f"{REL_CONFIDENT} 视为无高置信命中，matched=false 时"
                             "不要把该条当作已知事实引用。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="memagent_forget",
                description=("彻底删除一条记忆（id 用 memagent_find 查）。"
                             "用于错误/过时/被要求遗忘的内容。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "记忆 id（支持前缀）"},
                    },
                    "required": ["id"],
                },
            ),
            types.Tool(
                name="memagent_recall",
                description=("唤醒一条 Cold 记忆：摘要索引 → 重建完整记忆（含深藏细节）。"
                             "id 前缀即可（memagent_retrieve 命中 via_summary=true 的条目"
                             "可用它唤醒细节）。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id_prefix": {"type": "string", "description": "Cold 记忆 id 前缀"},
                    },
                    "required": ["id_prefix"],
                },
            ),
            types.Tool(
                name="memagent_find",
                description=("按关键词定位记忆（内容/摘要/原始内容需包含全部关键词，"
                             "不区分大小写）。用于拿到精确 id 后做 forget / recall。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "string",
                                     "description": "空格分隔多词 = 同时包含"},
                    },
                    "required": ["keywords"],
                },
            ),
            types.Tool(
                name="memagent_start",
                description=("开工注入：按主题检索相关决策（省略主题 = 当前强度最高的决策），"
                             "组成可直接放进上下文的决策记忆块。会话/任务开始时调用一次。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "nullable": True,
                                  "description": "主题关键词；省略取最强决策"},
                        "k": {"type": "integer", "default": 5},
                    },
                },
            ),
            types.Tool(
                name="memagent_export",
                description=("把全部决策记忆导出成 AGENTS.md 风格文档（按语义/技能/情景"
                             "分组，供支持指令文件的 agent 全量加载）。dual=true 时同步写"
                             " CLAUDE.md（内容一致）。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "AGENTS.md"},
                        "dual": {"type": "boolean", "default": False,
                                 "description": "同时写 AGENTS.md 与 CLAUDE.md"},
                    },
                },
            ),
            types.Tool(
                name="memagent_sleep",
                description=("睡眠巩固：回放近期记忆（再激活）、按价值分级、低频旧记忆压缩成摘要。"
                             "建议每 ~10 轮对话调用一次。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "duration": {"type": "number", "nullable": True,
                                     "description": "秒；省略=完整睡眠"},
                    },
                },
            ),
            types.Tool(
                name="memagent_stats",
                description="记忆库统计：Hot/Warm/Cold 各层数量。",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    async def call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        args = arguments or {}
        try:
            if name == "memagent_remember":
                mem = agent.remember(str(args["content"]),
                                     importance=float(args.get("importance", 0.5)),
                                     kind=str(args.get("kind", "fact")))
                _save(force=True)
                return _text({"id": mem.id, "content": mem.content,
                              "importance": mem.importance, "mtype": mem.mtype.value})
            if name == "memagent_retrieve":
                hits = agent.retrieve(str(args["query"]),
                                      k=max(1, min(int(args.get("k", 3)), 10)))
                _save()
                out = []
                matched = 0
                for h in hits:
                    q = str(args["query"]).strip().lower()
                    hay = (h.memory.content + (h.memory.summary or "")).lower()
                    is_match = h.relevance >= REL_CONFIDENT or (q in hay and len(q) >= 2)
                    if is_match:
                        matched += 1
                    out.append({
                        "id": h.memory.id,
                        "content": clip_content(h.memory.content, h.memory.id),
                        "summary": h.memory.summary,
                        "relevance": round(h.relevance, 4),
                        "score": round(h.total, 4),
                        "mtype": h.memory.mtype.value,
                        "via_summary": h.via_summary,
                        "matched": is_match,
                        "recall_hint": (f"Cold 摘要命中：细节已深藏，"
                                        f"可 memagent_recall {h.memory.id} 唤醒")
                        if h.via_summary else None,
                    })
                payload = {"hits": out, "matched": matched}
                # 舌尖现象层：低置信带（0.12≤rel<0.25）转成模糊线索，
                # 引导模型用 memagent_recall / memagent_find 顺势深挖
                tips = build_tips(str(args["query"]), hits)
                if tips:
                    payload["tips"] = [
                        {**t, "next": f"memagent_find {t['id_prefix']}"} for t in tips]
                if out and matched == 0:
                    payload["notice"] = (
                        "无高置信命中（top relevance="
                        f"{out[0]['relevance']} < {REL_CONFIDENT}）：结果置信偏低，"
                        "请勿当作已确认事实；可补全查询、用 memagent_recall 唤醒"
                        "Cold 摘要细节，或 memagent_start 按主题取决策后再试。")
                elif not out:
                    payload["notice"] = "记忆库为空或无候选。"
                return _text(payload)
            if name == "memagent_forget":
                mem_id = str(args["id"]).strip()
                target = None
                for m in agent.store.all():
                    if m.id == mem_id or m.id.startswith(mem_id):
                        target = m.id
                        break
                ok = agent.store.remove(target) if target else False
                _save(force=True)
                return _text({"forgotten": ok, "id": target or mem_id})
            if name == "memagent_recall":
                revived = agent.recall(str(args["id_prefix"]).strip())
                _save(force=True)
                if revived is None:
                    return _text({"awakened": False,
                                  "reason": "未找到该 Cold 记忆（用 memagent_find 查 id）"})
                return _text({"awakened": True, "id": revived.id,
                              "content": revived.content,
                              "originals": list(revived.originals.values()),
                              "mtype": revived.mtype.value,
                              "importance": revived.importance})
            if name == "memagent_find":
                mems = agent.find_memories(str(args["keywords"]))[:10]
                return _text({"count": len(mems), "memories": [
                    {"id": m.id, "tier": m.tier.value, "mtype": m.mtype.value,
                     "importance": round(m.importance, 2),
                     "content": clip_content(m.summary or m.content, m.id)}
                    for m in mems]})
            if name == "memagent_start":
                topic = args.get("topic") or None
                k = max(1, min(int(args.get("k", 5)), 20))
                block = build_injection_md(
                    agent, topic, k,
                    refresh_hint="重新调用 memagent_start 工具刷新本区块。",
                )
                return _text({"topic": topic, "injection": block})
            if name == "memagent_export":
                path = str(args.get("path", "AGENTS.md"))
                dual = bool(args.get("dual", False))
                result = export_agents_md(agent, path, dual=dual)
                paths = list(result) if isinstance(result, tuple) else [result]
                return _text({"paths": [str(Path(p).resolve()) for p in paths]})
            if name == "memagent_sleep":
                report = agent.sleep(args.get("duration"))
                _save(force=True)
                return _text({"replayed": report.get("replayed_count", 0),
                              "fogged": report.get("unreplayed_count", 0),
                              "cold_compressed": report.get("cold_compressed", 0),
                              "triage": {"high": report.get("triage_high", 0),
                                         "medium": report.get("triage_medium", 0),
                                         "low": report.get("triage_low", 0)}})
            if name == "memagent_stats":
                tiers = {t.value: len(agent.store.by_tier(t)) for t in Tier}
                return _text({"version": __version__, "tiers": tiers,
                              "total": len(agent.store.all())})
            raise ValueError(f"unknown tool: {name}")
        except KeyError as e:
            raise ValueError(f"missing argument: {e}") from e

    # --- mcp 2.x request handlers ---
    async def _list_tools(ctx, params):
        return types.ListToolsResult(tools=await list_tools())

    async def _call_tool(ctx, params):
        name = params.params.name
        arguments = params.params.arguments
        t0 = time.perf_counter()
        err_type = err_msg = None
        try:
            return await call_tool(name, arguments)
        except Exception as e:
            err_type, err_msg = type(e).__name__, str(e)[:200]
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(e))],
                is_error=True,
            )
        finally:
            diag.record_call(name, (time.perf_counter() - t0) * 1000.0,
                             err_type=err_type, err_msg=err_msg)

    server = Server("memagent", version=__version__)
    server.add_request_handler("tools/list", types.ListToolsRequest, _list_tools)
    server.add_request_handler("tools/call", types.CallToolRequest, _call_tool)
    server._agent = agent  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="memagent MCP server (stdio)")
    parser.add_argument("--persist", default=None, help="memory JSON path")
    parser.add_argument("--offline", action="store_true",
                        help="keyword classification only, no LLM calls")
    parser.add_argument("--embed-base-url", default=None,
                        help="OpenAI 兼容 /embeddings 端点")
    parser.add_argument("--embed-model", default="text-embedding-3-small",
                        help="远程嵌入模型名")
    parser.add_argument("--embed-api-key", default=None,
                        help="嵌入 API key（默认 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--embed-local", default=None,
                        help="本地 sentence-transformers 模型名")
    parser.add_argument("--embed-fastembed", default=None,
                        help="本地 ONNX 模型名（fastembed）")
    args = parser.parse_args(argv)

    import asyncio
    import os

    from memagent.embedders import FastEmbedder, LocalEmbedder, RemoteEmbedder

    embedder = None
    if args.embed_fastembed:
        embedder = FastEmbedder(args.embed_fastembed)
    elif args.embed_local:
        embedder = LocalEmbedder(args.embed_local)
    elif args.embed_base_url:
        embedder = RemoteEmbedder(
            base_url=args.embed_base_url,
            api_key=args.embed_api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=args.embed_model,
        )

    server = build_server(args.persist, args.offline, embedder=embedder)

    async def run() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream,
                             server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()