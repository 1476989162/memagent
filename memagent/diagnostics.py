# -*- coding: utf-8 -*-
"""脱敏诊断聚合：只收集引擎信号，永不收集记忆内容。

「出厂婴儿 + 本地优先」原则的延伸——别人使用本产品产生的进化燃料
（故障分布、性能特征、配置组合效果）需要一条**隐私安全**的回流通道：

- 收集：工具调用计数/耗时/异常类型与截断消息、环境指纹、库规模、学习器状态
- 不收集：记忆内容、查询原文、嵌入向量、文件路径、用户身份
- 去向：仅存本地 store.meta，由用户执行 ``--diagnostics`` 主动导出并自愿分享

用法：
    python -m memagent --diagnostics --persist <路径>   # 打印脱敏诊断报告
"""
from __future__ import annotations

import platform
import time

MAX_MSG = 160          # 异常消息截断上限（防内容经消息外泄）


class Diagnostics:
    """聚合器：挂在 store.meta["diagnostics"] 上随正常保存持久化，重启续记。"""

    VERSION = 1

    def __init__(self, meta: dict):
        slot = meta.setdefault("diagnostics", {})
        slot.setdefault("version", self.VERSION)
        slot.setdefault("tools", {})
        slot.setdefault("notes", {})
        self.slot = slot

    def record_env(self, *, embedder_name: str, embed_dim: int,
                   version: str) -> None:
        self.slot["env"] = {
            "os": platform.platform()[:60],
            "python": platform.python_version(),
            "memagent": version,
            "embedder": embedder_name,
            "embed_dim": embed_dim,
            "recorded_at": round(time.time(), 1),
        }

    def record_call(self, tool: str, ms: float, err_type: str | None = None,
                    err_msg: str | None = None) -> None:
        tools = self.slot["tools"]
        t = tools.setdefault(tool, {"calls": 0, "errors": 0,
                                    "ms_total": 0.0, "ms_max": 0.0})
        t["calls"] += 1
        ms = round(ms, 1)
        t["ms_total"] = round(t["ms_total"] + ms, 1)
        t["ms_max"] = round(max(t["ms_max"], ms), 1)
        if err_type:
            t["errors"] += 1
            # 只留最后一笔且截断——异常消息可能携带用户文本片段
            t["last_error"] = {"type": err_type,
                               "msg": (err_msg or "")[:MAX_MSG],
                               "at": round(time.time(), 1)}

    def note(self, key: str) -> None:
        """非调用类事件计数（如落盘冲突）。"""
        self.slot["notes"][key] = self.slot["notes"].get(key, 0) + 1


def build_report(store, *, version: str) -> dict:
    """构建脱敏诊断报告：引擎信号 + 库规模 + 学习器状态；零记忆内容。"""
    from .diagnostics import Diagnostics  # 自引用保持单一来源

    diag = Diagnostics(store.meta)
    tiers: dict[str, int] = {}
    for m in store.all():
        tiers[m.tier.value] = tiers.get(m.tier.value, 0) + 1
    state = store.meta.get("agent_state") or {}
    rem_edges = sum(len(v) for v in (state.get("rem_links") or {}).values())
    return {
        "report_version": 1,
        "generated_at": round(time.time(), 1),
        "engine": {
            "memagent": version,
            "python": platform.python_version(),
            "os": platform.platform()[:60],
        },
        "store": {"total": len(store), "tiers": tiers},
        "learner": {
            "tau": store.meta.get("learned_tau") or {},
            "plasticity": store.meta.get("learned_plasticity") or {},
        },
        "rem_links": rem_edges,
        "env": diag.slot.get("env", {}),
        "usage": diag.slot.get("tools", {}),
        "notes": diag.slot.get("notes", {}),
    }
