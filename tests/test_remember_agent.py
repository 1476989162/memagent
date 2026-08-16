"""memagent 记忆层 + LLM agent 编排测试：记忆注入、跨轮持久化、降级、CLI。"""

import json
import subprocess
import sys
from pathlib import Path

import remember_agent

SCRIPT = Path(__file__).resolve().parents[1] / "remember_agent.py"
from memagent.responder import LLMResponder
from memagent.synonyms import substring_priority_order


class FakePost:
    def __init__(self, reply="（LLM 回复）好的。"):
        self.reply = reply
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        return 200, json.dumps({"choices": [{"message": {"content": self.reply}}]})


def _mk(reply="（LLM 回复）好的。"):
    """persist_path=None → 内存 store：测试不依赖 CWD 的 memories_agent.json
    （项目根跑过 demo 会落盘该文件，默认路径加载会污染测试）。"""
    post = FakePost(reply)
    responder = LLMResponder(api_key="k", base_url="https://example.com/v1", post=post)
    return remember_agent.RememberAgent(persist_path=None, responder=responder), post


def test_fact_injected_into_llm_prompt():
    agent, post = _mk()
    agent.remember("我叫小林，是一名 Python 程序员")
    reply, injected = agent.chat("我的技术栈是什么？")
    assert reply == "（LLM 回复）好的。"
    assert any("我叫小林" in c for c, _mt, _s in injected)  # 检索命中并注入
    user = post.calls[-1]["messages"][-1]["content"]
    assert "我叫小林" in user          # 记忆真的进了发给 LLM 的 prompt


def test_conversation_turns_also_remembered():
    agent, post = _mk()
    agent.chat("我最近在学 Go")
    agent.chat("我最近在学 Go")  # 同句去重强化
    hits = agent.memory.retrieve("我在学什么", k=3)
    assert any("学 Go" in h.memory.content for h in hits)


def test_degrade_without_key_returns_none():
    """conftest 清了 OPENAI_API_KEY → 降级：注入照常、回复为 None。"""
    agent = remember_agent.RememberAgent(persist_path=None)  # 无 key responder，内存 store
    assert not agent.responder.available
    agent.remember("我叫小林")
    reply, injected = agent.chat("我叫什么名字")
    assert reply is None
    assert any("我叫小林" in c for c, _mt, _s in injected)


def test_chat_rate_limit_does_not_crash_or_drop_turn(capsys):
    def rate_limited(*args, **kwargs):
        raise RuntimeError("LLM 回复生成 HTTP 429（请求过于频繁）")

    responder = LLMResponder(
        api_key="k", base_url="https://example.com/v1",
        post=rate_limited, max_retries=0,
    )
    agent = remember_agent.RememberAgent(persist_path=None, responder=responder)
    reply, _ = agent.chat("你好")
    assert reply is None
    assert agent.memory.store.all()
    assert "429" in capsys.readouterr().out


def test_emotion_roundtrip_is_restored_as_emotion(tmp_path):
    from memagent.emotion import Emotion

    p = str(tmp_path / "mem.json")
    agent = remember_agent.RememberAgent(persist_path=p)
    agent.memory.remember("我今天很开心")
    agent.save()
    restored = remember_agent.RememberAgent(persist_path=p)
    restored.chat("你能做什么")
    assert all(isinstance(m.emotion, (Emotion, type(None)))
               for m in restored.memory.store.all())


def test_short_query_rerank_in_chat(capsys):
    """短查询（<3 字）chat 注入做子串优先重排 + 打印提示（与 session_memory 一致）。"""
    agent, post = _mk()
    agent.remember("遗忘斜率对比用触底时间而非斜率比")
    agent.remember("对照实验靠可注入时钟确定性快进")
    reply, injected = agent.chat("触底")
    assert "触底" in injected[0][0]  # 含查询词的排最前
    out = capsys.readouterr().out
    assert "子串优先重排" in out and "建议加长" in out


def test_rerank_flag_off_no_hint_in_chat(capsys):
    """AgentConfig.rerank_short_query=False：对话注入不重排也不打印提示。"""
    agent, _post = _mk()
    agent.memory.cfg.rerank_short_query = False
    agent.remember("遗忘斜率对比用触底时间而非斜率比")
    agent.remember("对照实验靠可注入时钟确定性快进")
    _reply, injected = agent.chat("触底")
    assert "子串优先重排" not in capsys.readouterr().out
    assert injected  # 注入照常（只是不重排）


def test_normal_query_no_rerank_in_chat(capsys):
    """长查询（≥3 字）不重排不提示。"""
    agent, post = _mk()
    agent.remember("语义化双阈值滞回避免振荡")
    agent.remember("循环导入用函数级导入解决")
    reply, injected = agent.chat("语义化")
    assert "语义化" in injected[0][0]
    assert "子串优先重排" not in capsys.readouterr().out


def test_rerank_consistent_across_entries(capsys):
    """两个入口（session_memory / remember_agent）对相同数据产生一致的短查询重排。"""
    import session_memory
    from memagent import MemoryAgent

    # session_memory 入口
    a1 = MemoryAgent()
    session_memory.record(a1, [], ["遗忘斜率对比用触底时间", "对照实验靠可注入时钟"])
    p1 = session_memory.pick_decisions(a1, topic="触底", k=5)
    # remember_agent 入口（同一批数据）
    a2, _post = _mk()
    a2.remember("遗忘斜率对比用触底时间")
    a2.remember("对照实验靠可注入时钟")
    _reply, inj = a2.chat("触底")
    assert "触底" in p1[0][0].content   # 两者含查询词的条目都排最前
    assert "触底" in inj[0][0]
    assert substring_priority_order.__module__ == "memagent.synonyms"  # 单点实现
    capsys.readouterr()


def test_injected_from_cold_hit_uses_summary():
    """Cold 命中（via_summary）注入摘要文本而非深藏 content（与核心一致）——
    命中词「ai」只在摘要里，content 不含。"""
    agent, _post = _mk()
    m = agent.memory.store.add("用户聊过一次项目背景", importance=0.1)
    m.demote_to_cold("开发决策：AI 分类")
    hits = agent.memory.retrieve("ai", k=3)
    injected = remember_agent.injected_from(hits, k=3)
    assert injected
    assert any("开发决策：AI 分类" in c for c, _mt, _s in injected)   # 注入摘要
    assert all("用户聊过一次项目背景" not in c for c, _mt, _s in injected)


def test_weak_relevant_filtered_from_injection():
    """弱相关（total≤0.05）不注入（与模板回复的 relevant 阈值一致）。"""
    from memagent.agent import Retrieved
    from memagent.memory import MemoryStore

    mem = MemoryStore().add("测试记忆")
    hits = [
        Retrieved(memory=mem, relevance=0.9, strength=0.5, total=0.45, via_summary=False),
        Retrieved(memory=mem, relevance=0.1, strength=0.2, total=0.02, via_summary=False),
    ]
    injected = remember_agent.injected_from(hits, k=3)
    assert len(injected) == 1  # 只有强相关注入


def test_persistence_across_instances(tmp_path):
    p = str(tmp_path / "mem.json")
    a = remember_agent.RememberAgent(persist_path=p)
    a.remember("我的生日是 3 月 14 日")
    a.save()
    b = remember_agent.RememberAgent(persist_path=p)
    hits = b.memory.retrieve("我的生日", k=1)
    assert hits and "3 月 14 日" in hits[0].memory.content


def test_export_agents_md_content_and_groups(tmp_path):
    """导出文件含高价值记忆、按类型分组、排除对话流水。"""
    agent, _ = _mk()
    m1 = agent.remember("我叫小林，是一名 Python 程序员")  # semantic
    m1.access_count = 3  # 被反复确认
    m2 = agent.remember("用户偏好：代码注释用中文")
    m2.access_count = 5
    agent.chat("今天聊了架构")  # turn 流水（不应导出）
    path = agent.export_agents_md(str(tmp_path / "AGENTS.md"))
    text = open(path, encoding="utf-8").read()
    assert "我叫小林" in text
    assert "代码注释用中文" in text
    assert "## semantic" in text       # 分组标题
    assert "检索 5 次" in text         # 频率标注
    assert "用户说：" not in text      # turn 流水排除


def test_export_agents_md_filters_low_value(tmp_path):
    """低重要且从未检索的记忆不导出（非固化知识）。"""
    agent, _ = _mk()
    m = agent.remember("昨天聊了个无关话题")
    m.importance = 0.1  # 低重要性 + 零检索 → 非固化知识
    path = agent.export_agents_md(str(tmp_path / "AGENTS.md"))
    text = open(path, encoding="utf-8").read()
    assert "无关话题" not in text
    assert "共 0 条固化知识" in text


def test_export_agents_md_custom_threshold_and_path(tmp_path):
    agent, _ = _mk()
    agent.remember("重要事实：项目用 FastAPI").importance = 0.6
    path = agent.export_agents_md(
        str(tmp_path / "CLAUDE.md"), min_importance=0.5, min_access=0,
    )
    text = open(path, encoding="utf-8").read()
    assert "FastAPI" in text
    assert "共 1 条固化知识" in text


def test_export_agents_md_dual_writes_both_in_sync(tmp_path, monkeypatch):
    """双格式：同一份内容写入 AGENTS.md + CLAUDE.md（逐字节一致，保持同步）。"""
    monkeypatch.chdir(tmp_path)
    agent, _ = _mk()
    agent.remember("我叫小林，是一名 Python 程序员").access_count = 3
    exports = agent.export_agents_md(dual=True)
    assert set(exports) == {"AGENTS.md", "CLAUDE.md"}
    a = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    c = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert a == c
    assert "我叫小林" in a
    assert "同时刷新" in a and "CLAUDE.md" in a


def test_cli_export_agents_md_dual_no_arg(tmp_path):
    """CLI 不带文件名 → 同时生成 AGENTS.md + CLAUDE.md（内容同步），不污染项目根。"""
    persist = str(tmp_path / "mem.json")
    subprocess.run(
        [sys.executable, "remember_agent.py", "--remember", "我叫小林，是一名 Python 程序员",
         "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    e = subprocess.run(
        [sys.executable, str(SCRIPT), "--export-agents-md", "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=tmp_path,
    )
    assert e.returncode == 0, e.stdout + e.stderr
    assert "AGENTS.md + CLAUDE.md" in e.stdout
    a = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    c = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert a == c
    assert "AGENTS.md 与 CLAUDE.md" in a  # 头部注明双格式同步


def test_cli_demo_runs(tmp_path):
    """--demo 在无 key 环境降级运行：写入事实、注入记忆、退出码 0。"""
    proc = subprocess.run(
        [sys.executable, "remember_agent.py", "--demo", "--persist", str(tmp_path / "demo_mem.json")],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[写入]" in proc.stdout
    assert "注入" in proc.stdout          # 记忆层注入工作
    assert "跨轮记忆演示完成" in proc.stdout
