"""自主演化测试：evolve() 吸收设定入库、去重、sleep 钩子、人设边界文本。"""

import json

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.responder import _NOVELIST_PERSONA


class FakeResponder:
    """最小假 responder：available=True，respond 返回预设文本。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0
        self.persona = None

    @property
    def available(self) -> bool:
        return True

    def set_persona(self, persona: str) -> None:
        self.persona = persona

    def respond(self, query, memories=None, persona_extras=None):
        self.calls += 1
        return self.reply


def _agent(reply: str, **kw):
    r = FakeResponder(reply)
    return MemoryAgent(responder=r, persona="novelist", **kw), r


def test_evolve_no_llm_noop():
    a = MemoryAgent()  # 无 responder
    ev = a.evolve()
    assert ev["ok"] is False and ev["reason"] == "no-llm"
    assert ev["added"] == []


def test_evolve_absorbs_settings_into_sheet():
    reply = "设定：新人物——林蝉衣，粗通封印术\n设定：残剑听雪为雌雄双剑之一\n无关行应被跳过"
    a, r = _agent(reply, cfg=AgentConfig(evolve_search=False))
    ev = a.evolve(with_web=False)
    assert ev["ok"] is True
    assert len(ev["added"]) == 2
    assert all(m.kind == "setting" for m in a.store.all())
    sheet = a.persona_sheet() or ""
    assert "林蝉衣" in sheet and "雌雄双剑" in sheet


def test_evolve_dedup_skips_existing_settings():
    a, r = _agent("设定：林尘身负残剑听雪", cfg=AgentConfig(evolve_search=False))
    a.remember_setting("林尘身负残剑听雪", importance=0.9)
    ev = a.evolve(with_web=False)
    assert ev["added"] == []            # 与已有设定高度相似 → 去重跳过


def test_evolve_llm_error_is_graceful():
    class BoomResponder:
        @property
        def available(self):
            return True

        def respond(self, query, memories=None, persona_extras=None):
            raise RuntimeError("网络错误")

    a = MemoryAgent(responder=BoomResponder(), persona="novelist",
                    cfg=AgentConfig(evolve_search=False))
    ev = a.evolve(with_web=False)
    assert ev["ok"] is False and "llm-error" in ev["reason"]
    assert ev["added"] == []


def test_sleep_evolve_on_sleep_hook():
    a, r = _agent("设定：伏笔——白雀谷一夜尽成灰烬",
                  cfg=AgentConfig(evolve_on_sleep=True, evolve_search=False))
    a.remember_setting("作品：《青州问剑录》", importance=0.9)
    report = a.sleep()
    assert r.calls >= 1                    # 睡眠触发了演化（LLM 调用）
    assert report["evolve_ok"] is True
    assert any("白雀谷" in s for s in report["evolved"])


def test_sleep_no_evolve_when_flag_off():
    a, r = _agent("设定：任意内容", cfg=AgentConfig(evolve_on_sleep=False, evolve_search=False))
    a.sleep()
    assert r.calls == 0                    # 默认不静默调用 LLM


def test_persona_allows_adult_content_with_hard_boundaries():
    """成年露骨内容放开，但铁律文本必须包含：成年/自愿/禁未成年/禁强迫。"""
    assert "露骨" in _NOVELIST_PERSONA
    assert "18 岁以上" in _NOVELIST_PERSONA
    assert "自愿" in _NOVELIST_PERSONA
    assert "未成年" in _NOVELIST_PERSONA
    assert "强迫" in _NOVELIST_PERSONA
    assert "铁律" in _NOVELIST_PERSONA
