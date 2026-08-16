"""compat.call_responder 的 max_tokens 透传测试。

背景（2026-08-16 修复）：长输出（FoxTable 代码生成 / 写章正文）被 responder
默认 max_tokens=1024 拦腰截断——代码块收不了栏、章节卡在 ~1000 字。修复让
call_responder 支持逐调用 max_tokens，并按签名过滤兼容不支持该参数的第三方
responder（旧契约不破坏）。
"""

from memagent.compat import call_responder


class _FullResponder:
    """支持全部可选参数的 responder（LLMResponder 签名形态）。"""

    def __init__(self):
        self.kwargs = None

    def respond(self, query, memories=None, persona_extras=None,
                timeout=None, max_tokens=None):
        self.kwargs = {"memories": memories, "persona_extras": persona_extras,
                       "timeout": timeout, "max_tokens": max_tokens}
        return "ok"


class _LegacyResponder:
    """不支持 max_tokens 的第三方 responder——过滤后调用不报错。"""

    def __init__(self):
        self.kwargs = None

    def respond(self, query, memories=None, timeout=None):
        self.kwargs = {"memories": memories, "timeout": timeout}
        return "legacy"


def test_call_responder_passes_max_tokens():
    r = _FullResponder()
    call_responder(r, "写代码", max_tokens=4096)
    assert r.kwargs["max_tokens"] == 4096


def test_call_responder_max_tokens_defaults_to_none():
    r = _FullResponder()
    call_responder(r, "聊天")
    assert r.kwargs["max_tokens"] is None  # 不传 = 用 responder 自己的默认


def test_call_responder_filters_max_tokens_for_legacy_signature():
    r = _LegacyResponder()
    assert call_responder(r, "聊天", max_tokens=4096) == "legacy"
    assert "max_tokens" not in r.kwargs
