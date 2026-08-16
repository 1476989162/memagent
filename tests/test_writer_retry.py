"""写章管线重试机制（合并自 write_chapter_55.py 的一次性实现）的单元测试。

覆盖：
1. call_with_retry —— 空回复/异常自动重试，成功才返回；
2. 短回复门槛 —— min_len=1 时短回复不被误判为空（ch54 空标题根因）；
3. min_len 门槛 —— 低于门槛的回复重试到失败才抛错；
4. 写章级重试 —— llm-incomplete/llm-empty 整轮重跑，章号不占号，最终成功。
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402


class FakeResponder:
    """可用假 responder：按预设序列返回/抛异常。"""

    available = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def respond(self, query, **kw):
        self.calls.append(query)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_agent(replies):
    tmp = tempfile.mkdtemp()
    store = MemoryStore(path=os.path.join(tmp, "test_memory.json"))
    agent = MemoryAgent(store=store, persona="novelist",
                        cfg=AgentConfig(evolve_on_sleep=False))
    agent.responder = FakeResponder(replies)
    return agent


def test_call_with_retry_empty_then_success():
    """前 2 次抛空回复异常，第 3 次成功 → 返回文本，共 3 次调用。"""
    agent = _make_agent([
        RuntimeError("LLM 回复为空（可能只有 reasoning，没有最终 content）"),
        RuntimeError("LLM 回复为空（可能只有 reasoning，没有最终 content）"),
        "这是正文内容，足够长。",
    ])
    agent.cfg.llm_retry_delay = 0.01
    reply = agent.call_with_retry("prompt", min_len=5)
    assert reply == "这是正文内容，足够长。"
    assert len(agent.responder.calls) == 3


def test_call_with_retry_short_reply_passes_with_min_len_1():
    """短回复门槛：min_len=1 时 6-12 字标题不被误判为空（ch54 空标题根因）。"""
    agent = _make_agent(["霜线攀骨"])
    reply = agent.call_with_retry("拟标题", min_len=1)
    assert reply == "霜线攀骨"
    assert len(agent.responder.calls) == 1


def test_call_with_retry_below_min_len_retries_then_raises():
    """低于 min_len 门槛的回复触发重试，全失败抛最后一个错误。"""
    agent = _make_agent(["短", "短", "短"])
    agent.cfg.llm_retry_delay = 0.01
    with pytest.raises(RuntimeError):
        agent.call_with_retry("prompt", min_len=50)
    assert len(agent.responder.calls) == 3


def test_write_chapter_retries_incomplete_rounds():
    """写章级重试：llm-incomplete → llm-empty → 成功，整轮重跑 3 次不占章号。"""
    agent = _make_agent([])  # responder 只为过 available 检查
    agent.cfg.chapter_retry_delay = 0.01
    tmp = tempfile.mkdtemp()

    outcomes = [
        {"ok": False, "reason": "llm-incomplete"},
        {"ok": False, "reason": "llm-empty"},
        {"ok": True, "title": "错季锁星", "chapter": 1, "words": 100,
         "path": str(Path(tmp) / "chapters" / "第1章.md")},
    ]

    def fake_write(*a, **kw):
        return outcomes.pop(0)

    agent._write_chapter_locked = fake_write
    result = agent.write_chapter(save_dir=str(Path(tmp) / "chapters"))
    assert result.get("ok") is True
    assert result["chapter"] == 1
    assert len(outcomes) == 0  # 三轮全部消费


def test_write_chapter_non_retryable_reason_breaks():
    """非可重试失败（如 writer 冲突）不重试，直接返回。"""
    agent = _make_agent([])
    agent.cfg.chapter_retry_delay = 0.01
    tmp = tempfile.mkdtemp()

    def fake_write(*a, **kw):
        return {"ok": False, "reason": "chapter-conflict"}

    agent._write_chapter_locked = fake_write
    result = agent.write_chapter(save_dir=str(Path(tmp) / "chapters"))
    assert result.get("ok") is False
    assert result["reason"] == "chapter-conflict"
