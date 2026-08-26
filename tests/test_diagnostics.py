# -*- coding: utf-8 -*-
"""脱敏诊断测试：收集引擎信号、永不外泄记忆内容。

核心不变式：把含「独特秘密串」的记忆写入库后，诊断报告的完整序列化
文本中不得出现该秘密——这是别人愿意分享诊断包的前提。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMAGENT_TEST", "1")

from memagent.diagnostics import Diagnostics, build_report  # noqa: E402
from memagent import MemoryAgent  # noqa: E402
from memagent.agent import AgentConfig  # noqa: E402

SECRET = "绝密项目代号夜莺749局"


def test_record_call_shape_and_truncation():
    meta: dict = {}
    d = Diagnostics(meta)
    d.record_env(embedder_name="hash", embed_dim=256, version="9.9.9")
    d.record_call("memagent_retrieve", 12.3)
    long_msg = "前缀" + SECRET * 30
    d.record_call("memagent_remember", 5.0, err_type="ValueError",
                  err_msg=long_msg)
    tools = meta["diagnostics"]["tools"]
    assert tools["memagent_retrieve"]["calls"] == 1
    assert tools["memagent_retrieve"]["errors"] == 0
    t = tools["memagent_remember"]
    assert t["calls"] == 1 and t["errors"] == 1
    assert len(t["last_error"]["msg"]) <= 160
    assert meta["diagnostics"]["env"]["embedder"] == "hash"


def test_note_counter():
    meta: dict = {}
    d = Diagnostics(meta)
    d.note("save_conflicts")
    d.note("save_conflicts")
    assert meta["diagnostics"]["notes"]["save_conflicts"] == 2


def test_report_never_leaks_memory_content(tmp_path):
    """隐私不变式：秘密记忆写入后，诊断报告序列化文本零泄漏。"""
    path = tmp_path / "m.json"
    agent = MemoryAgent(persist_path=str(path),
                        cfg=AgentConfig(tau_seconds=30.0))
    agent.remember(f"我叫小林，参与{SECRET}计划，邮箱 lin@example.com",
                   importance=0.8)
    agent.save()
    # 重启式加载（模拟真实使用）
    from memagent.memory import MemoryStore

    store = MemoryStore(path=str(path))
    report = build_report(store, version="test-0.0")
    text = json.dumps(report, ensure_ascii=False)
    assert "total" in report["store"]
    assert SECRET not in text, "记忆内容泄漏进诊断报告！"
    assert "lin@example.com" not in text
    assert "我叫小林" not in text


def test_meta_roundtrip_continues_counters(tmp_path):
    """诊断数据随正常保存持久化，重启后计数续记而非清零。"""
    path = tmp_path / "m.json"
    meta_holder: dict = {}
    d1 = Diagnostics(meta_holder)
    d1.record_call("memagent_stats", 3.0)
    # 模拟挂到 store.meta 后保存再加载
    agent = MemoryAgent(persist_path=str(path),
                        cfg=AgentConfig(tau_seconds=30.0))
    agent.store.meta["diagnostics"] = meta_holder["diagnostics"]
    agent.save()
    from memagent.memory import MemoryStore

    store2 = MemoryStore(path=str(path))
    d2 = Diagnostics(store2.meta)
    d2.record_call("memagent_stats", 4.0)
    assert store2.meta["diagnostics"]["tools"]["memagent_stats"]["calls"] == 2
