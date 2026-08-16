"""每日创作产出测试：write_chapter 落盘、章节号递增、进度回写、连续性注入。"""

from pathlib import Path

from memagent import MemoryAgent
from memagent.agent import AgentConfig

CHAPTER1 = (
    "夜色渐深，林尘在丹房外停住脚步，指尖还残留着药炉的余温。"
    "他抬头望向林氏主家的方向，握紧了残剑听雪的剑柄。"
)


class FakeResponder:
    def __init__(self, reply: str):
        self.reply = reply
        self.queries: list[str] = []

    @property
    def available(self):
        return True

    def set_persona(self, persona):
        pass

    def respond(self, query, memories=None, persona_extras=None):
        self.queries.append(query)
        return self.reply


def _agent(tmp_path, reply=CHAPTER1 + "……这一夜注定无眠，远处的钟声仍在提醒他风暴尚未过去。"):
    r = FakeResponder(reply)
    a = MemoryAgent(responder=r, persona="novelist",
                    cfg=AgentConfig(chapter_save_dir=str(tmp_path), chapter_words=80))
    a.remember_setting("作品：《青州问剑录》，长篇玄幻，已连载 42 章", importance=0.95)
    a.remember_setting("主角：林尘，青州林氏旁支少年，身负残剑‘听雪’", importance=0.9)
    return a, r


def test_write_chapter_no_llm_noop():
    a = MemoryAgent()
    w = a.write_chapter()
    assert w["ok"] is False and w["reason"] == "no-llm"


def test_write_chapter_writes_file_and_progress(tmp_path):
    a, r = _agent(tmp_path)
    w = a.write_chapter()
    assert w["ok"] is True
    assert w["title"] == "青州问剑录"
    assert w["chapter"] == 1
    assert w["words"] > 50
    f = Path(w["path"])
    assert f.is_file()
    assert f.read_text(encoding="utf-8").startswith("# 《青州问剑录》第1章")
    # 进度回写进设定记忆
    progress = [m.content for m in a.store.all()
                if m.kind == "setting" and "已连载" in m.content]
    assert progress == ["《青州问剑录》：已连载 1 章"]
    # 设定档案注入写作提示
    assert "残剑" in r.queries[0] and "第一章" in r.queries[0] or "第 1 章" in r.queries[0]


def test_write_chapter_increments_and_replaces_progress(tmp_path):
    a, r = _agent(tmp_path)
    a.write_chapter()
    w2 = a.write_chapter()
    assert w2["chapter"] == 2
    assert (tmp_path / "青州问剑录" / "chapters" / "第2章.md").is_file()
    progress = [m.content for m in a.store.all()
                if m.kind == "setting" and "已连载" in m.content]
    assert progress == ["《青州问剑录》：已连载 2 章"]   # 替换而非新增


def test_write_chapter_continuity_injects_previous_tail(tmp_path):
    a, r = _agent(tmp_path)
    a.write_chapter()
    a.write_chapter()
    # 第二章的提示里应包含第一章结尾（无缝衔接）
    assert CHAPTER1[-30:] in r.queries[-1]


def test_write_chapter_empty_llm_rejected(tmp_path):
    a, r = _agent(tmp_path, reply="好")
    a.cfg.chapter_retry_delay = 0.01  # 新重试特性：失败轮整轮重跑，测试用快速间隔
    w = a.write_chapter()
    assert w["ok"] is False and w["reason"] == "llm-empty"
    assert not list((tmp_path / "青州问剑录" / "chapters").glob("第*.md"))


def test_write_chapter_truncated_llm_rejected(tmp_path):
    a, _ = _agent(tmp_path, reply="截断正文" * 20)
    a.cfg.chapter_words = 1000
    a.cfg.chapter_retry_delay = 0.01  # 新重试特性：失败轮整轮重跑，测试用快速间隔
    w = a.write_chapter()
    assert w["ok"] is False and w["reason"] == "llm-truncated"
    assert w["minimum_words"] == 900


def test_write_chapter_incomplete_ending_rejected(tmp_path):
    a, _ = _agent(tmp_path, reply="正文仍在继续，" * 150)
    a.cfg.chapter_words = 1000
    a.cfg.chapter_retry_delay = 0.01  # 新重试特性：失败轮整轮重跑，测试用快速间隔
    w = a.write_chapter()
    assert w["ok"] is False and w["reason"] == "llm-incomplete"
    assert not list((tmp_path / "青州问剑录" / "chapters").glob("第*.md"))


def test_write_chapter_default_work_name(tmp_path):
    r = FakeResponder("这是一个足够长的正文内容用来通过长度门控……" * 5)
    a = MemoryAgent(responder=r, persona="novelist",
                    cfg=AgentConfig(chapter_save_dir=str(tmp_path), chapter_words=80))
    w = a.write_chapter()
    assert w["title"] == "未命名作品"
    assert w["ok"] is True
