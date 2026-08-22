"""写章守卫 ensure_title_in_sheet 的单元测试。

背景：伏笔/新设定记忆以高重要性积累，会把作品名记忆挤出人设档案前 8，
导致 _work_title() 误判为"未命名作品"、章节写进错误目录。
守卫应在每次写章前/后自动把书名记忆重要性提到第 8 名之上。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402


def _make_agent():
    tmp = tempfile.mkdtemp()
    store = MemoryStore(path=os.path.join(tmp, "test_memory.json"))
    agent = MemoryAgent(store=store, persona="novelist",
                        cfg=AgentConfig(evolve_on_sleep=False))
    return store, agent


def _title_mem(agent):
    return next(m for m in agent.store.all()
                if m.kind == "setting" and "错季锁星" in m.content)


def test_title_pushed_out_is_auto_boosted():
    """无进度标记的书名被多条高重要性伏笔挤出前 8 → 守卫提升后 _work_title 恢复正确。

    （带“已连载 N 章”进度条的书名记忆在 persona_sheet 中恒排最前，不会被
    挤出——那是排序口径修复；本测试覆盖的是守卫对普通书名记忆的提升路径。）
    """
    store, agent = _make_agent()
    # 书名用低 importance，且 kind="setting" 走 remember 时会经情绪调制被略抬，
    # 但远低于伏笔——伏笔用 0.98 大量写入，足以把书名挤到第 8 名之后。
    agent.remember_setting("《错季锁星》是本玄幻小说的作品名", importance=0.3)
    for i in range(20):
        agent.remember_setting(f"伏笔第{i}章某物伏笔正在积累", importance=0.98)
    # 复现 bug：书名被挤出前 8
    assert agent._work_title() == "未命名作品"
    before = _title_mem(agent).importance
    assert agent.ensure_title_in_sheet() == "错季锁星"
    after = _title_mem(agent).importance
    assert agent._work_title() == "错季锁星"
    assert after > before


def test_guard_is_idempotent():
    """已提升过且仍在前 8 → 再次调用不再改动 importance。"""
    _, agent = _make_agent()
    agent.remember_setting("《错季锁星》：已连载 55 章", importance=0.3)
    for i in range(20):
        agent.remember_setting(f"伏笔第{i}章某物伏笔正在积累", importance=0.98)
    agent.ensure_title_in_sheet()
    first = _title_mem(agent).importance
    agent.ensure_title_in_sheet()
    assert _title_mem(agent).importance == first


def test_unnamed_work_memory_is_filtered():
    """高重要性的《未命名作品》假记忆不被当作作品名。"""
    _, agent = _make_agent()
    agent.remember_setting("《未命名作品》：已连载 2 章", importance=5.0)
    agent.remember_setting("《错季锁星》：已连载 55 章", importance=0.95)
    assert agent.ensure_title_in_sheet() == "错季锁星"


def test_already_in_sheet_is_noop():
    """书名已在前 8 → 守卫不改 importance。"""
    store, agent = _make_agent()
    agent.remember_setting("《错季锁星》：已连载 55 章", importance=3.0)
    agent.remember_setting("普通设定一条", importance=0.5)
    before = _title_mem(agent).importance
    assert agent.ensure_title_in_sheet() == "错季锁星"
    assert _title_mem(agent).importance == before


def test_no_title_memory_returns_none():
    """没有任何《…》设定记忆 → 返回 None，调用方走"未命名作品"兜底。"""
    _, agent = _make_agent()
    agent.remember_setting("一些无关设定", importance=1.0)
    assert agent.ensure_title_in_sheet() is None
