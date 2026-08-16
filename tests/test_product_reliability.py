"""Product reliability: state recovery, locking, and cross-module contracts."""

import pytest

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.emotion import BASIC_EMOTIONS
from memagent.graph import KnowledgeGraph
from memagent.io_utils import FileLock
from memagent.memory import ConcurrentWriteError, MemoryStore
from memagent.architecture import migrate_legacy_work, next_chapter_goal


class FakeResponder:
    available = True

    def __init__(self, reply="足够长的正文" * 30):
        self.reply = reply

    def respond(self, query, memories=None, persona_extras=None):
        return self.reply


def test_curiosity_pattern_fallback_uses_growth_list():
    agent = MemoryAgent()
    agent.set_growth_direction("AI", 0.8, ["AI"])
    agent.growth.predict("AI", "AI模型", "准确", 0.3)
    report = agent.curiosity.explore_loop()
    assert report["unanswered"] == 1


def test_graph_both_neighbors_excludes_unrelated_edges():
    graph = KnowledgeGraph()
    graph.add_edge("X", "Y", "related", 1.0)
    assert graph.neighbors("A", direction="both") == []
    assert graph.neighbors("X", direction="both") == [("Y", "related", 1.0)]


def test_agent_growth_state_roundtrip(tmp_path):
    path = tmp_path / "memory.json"
    agent = MemoryAgent(persist_path=str(path))
    agent.set_growth_direction("Python", 0.9, ["python", "pytest"])
    agent.graph.add_node("pytest", language="Python")
    agent.graph.label_node("pytest", "Python")
    agent.growth.predict("Python", "运行测试", "发现回归", 0.7)
    agent.growth.form_concept("回归测试", "Python", ["可重复"], 3)
    agent.cognition.register_skill("测试", "Python")
    agent.cognition.practice("测试", True, 123.0)
    agent.cognition.set_goal("稳定发布", "保持全绿", "Python", "测试", 0.8)
    agent.curiosity.unanswered.append("如何降低回归率？")
    agent.analogy.register_analogy("写作", "复盘", "编程", "代码审查")
    agent.social.social_history.append({"type": "share_memory", "peer": "peer-1"})
    agent.set_current_emotion(BASIC_EMOTIONS["joy"])
    agent.save()

    restored = MemoryAgent(persist_path=str(path))
    assert restored.interest.get("Python") == pytest.approx(0.9)
    assert restored.graph.get_node("pytest")["properties"]["language"] == "Python"
    assert restored.growth.predictions[0].expected == "发现回归"
    assert restored.growth.concepts[0].name == "回归测试"
    assert restored.cognition.skills["测试"].practice_count == 1
    assert restored.cognition.goals["稳定发布"].target_skill == "测试"
    assert restored.curiosity.unanswered == ["如何降低回归率？"]
    assert restored.analogy.transfer_summary()["registered_analogies"]
    assert restored.social.social_history[0]["peer"] == "peer-1"
    assert restored.current_emotion.label == "joy"


def test_memory_store_detects_stale_writer(tmp_path):
    path = tmp_path / "memory.json"
    initial = MemoryStore(path=str(path))
    initial.add("initial")
    initial.save()
    first = MemoryStore(path=str(path))
    stale = MemoryStore(path=str(path))
    first.add("first writer")
    first.save()
    stale.add("stale writer")
    with pytest.raises(ConcurrentWriteError):
        stale.save()


def test_memory_store_recovers_from_backup(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path=str(path))
    store.add("version one")
    store.save()
    store.add("version two")
    store.save()
    path.write_text("{broken", encoding="utf-8")
    recovered = MemoryStore(path=str(path))
    assert any(m.content == "version one" for m in recovered.all())
    recovered.save()
    assert len(MemoryStore(path=str(path))) == 1
    backup = path.with_suffix(path.suffix + ".bak")
    assert len(MemoryStore(path=str(backup))) == 1


def test_writer_lock_returns_busy_without_generating(tmp_path):
    cfg = AgentConfig(chapter_save_dir=str(tmp_path), chapter_words=80)
    agent = MemoryAgent(responder=FakeResponder(), persona="novelist", cfg=cfg)
    work_root = tmp_path / "未命名作品"
    with FileLock(work_root / ".writer.lock"):
        report = agent.write_chapter()
    assert report == {"ok": False, "reason": "writer-busy", "title": "未命名作品"}


def test_growth_summary_has_count_and_items():
    agent = MemoryAgent()
    agent.growth.form_concept("概念", "测试", ["稳定"], 2)
    summary = agent.growth.growth_summary()
    assert summary["concept_count"] == 1
    assert summary["concepts"][0]["name"] == "概念"


def test_architecture_reads_existing_chapters_directory(tmp_path):
    class CaptureResponder(FakeResponder):
        def __init__(self):
            super().__init__("剧情目标：继续追查线索。")
            self.query = ""

        def respond(self, query, memories=None, persona_extras=None):
            self.query = query
            return self.reply

    responder = CaptureResponder()
    agent = MemoryAgent(responder=responder, persona="novelist")
    chapters = tmp_path / "作品" / "chapters"
    chapters.mkdir(parents=True)
    (chapters / "第1章.md").write_text("开头\n独一无二的上一章结尾", encoding="utf-8")
    goal = next_chapter_goal(agent, "作品", 2, chapters)
    assert goal == "继续追查线索。"
    assert "独一无二的上一章结尾" in responder.query


def test_work_title_migration_archives_when_target_has_chapters(tmp_path):
    old = tmp_path / "未命名作品" / "chapters"
    new = tmp_path / "新书名" / "chapters"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "第1章.md").write_text("旧开篇", encoding="utf-8")
    (new / "第1章.md").write_text("新开篇", encoding="utf-8")
    report = migrate_legacy_work(tmp_path, "未命名作品", "新书名")
    assert report["migrated"] is True and report["archived"] is True
    assert (tmp_path / "新书名" / "legacy" / "未命名作品" / "chapters" / "第1章.md").is_file()
    assert (new / "第1章.md").read_text(encoding="utf-8") == "新开篇"
