# -*- coding: utf-8 -*-
"""展示边界截断测试：存储全文永不截断，注入/导出/回复按预算裁剪 + 唤醒指针。

原则（与 Cold 层"摘要索引+originals 深藏"同构）：
- 检索向量、测试效应、再巩固全部基于原文——智力不受损
- 只有展示给 LLM/用户的文本被预算封顶，细节可 memagent_recall 按需取回
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMAGENT_TEST", "1")

from memagent.instructions import (INJECTION_MAX_CHARS, build_injection_md,
                                   clip_content, export_agents_md_text)
from memagent.agent import AgentConfig, MemoryAgent


def test_short_text_untouched():
    assert clip_content("短记忆", "abc123") == "短记忆"


def test_long_text_clipped_with_recall_pointer():
    text = "长" * 500
    out = clip_content(text, "abcdef123456")
    assert out.startswith("长" * INJECTION_MAX_CHARS)
    assert "memagent_recall abcdef" in out
    assert len(out) < len(text)                      # 确实变短
    assert "500" not in out[:INJECTION_MAX_CHARS]    # 原文尾部被裁掉


def test_zero_disables_clipping():
    text = "长" * 500
    assert clip_content(text, "x", max_chars=0) == text


def test_storage_never_truncated():
    """核心不变式：remember 后 store 里仍是全文，截断只发生在展示函数。"""
    agent = MemoryAgent(cfg=AgentConfig(tau_seconds=30.0))
    raw = "伏笔细节，" * 80                            # ~400 字
    m = agent.remember(raw)
    assert len(m.content) == len(raw)                # 存储层全文
    shown = clip_content(m.content, m.id)            # 展示层才裁
    assert len(shown) < len(m.content)


def test_build_injection_md_bounds_payload():
    long_text = "线" * 800
    agent = MemoryAgent(cfg=AgentConfig(tau_seconds=30.0))
    agent.remember(long_text, importance=0.9)
    block = build_injection_md(agent)
    assert "memagent_recall" in block
    # 单条贡献被钳在预算内（含标注尾巴也远小于原文）
    assert ("线" * 300) not in block


def test_export_agents_md_also_clipped():
    agent = MemoryAgent(cfg=AgentConfig(tau_seconds=30.0))
    agent.remember("细" * 600, importance=0.9)
    doc = export_agents_md_text(agent)
    assert "memagent_recall" in doc
    assert ("细" * 300) not in doc


def test_reply_template_clips_long_memory():
    agent = MemoryAgent(cfg=AgentConfig(tau_seconds=30.0, reconsolidate=False))
    agent.remember("回" * 500, importance=0.5)
    hits = agent.retrieve("回" * 3, k=1)
    reply = agent._template_reply(hits)
    assert "memagent_recall" in reply
