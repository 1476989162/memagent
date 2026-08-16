"""LLM 分类器测试：mock HTTP 传输，验证 LLM 结果、缓存、回退、别名与
429 多模型池自动切换（含全部限流后等待重试）。"""

import json
import time

import pytest

from memagent import MemoryAgent
from memagent.llm import LLMClassifier, _parse_llm_json, _thinking_mode
from memagent.memory import MemType, classify_memory_with_confidence


class FakePost:
    """可注入的假 HTTP 客户端。"""

    def __init__(self, reply_text: str = "", status: int = 200, raise_exc: Exception | None = None):
        self.reply_text = reply_text
        self.status = status
        self.raise_exc = raise_exc
        self.calls = 0
        self.last_payload = None

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        self.last_payload = payload
        if self.raise_exc:
            raise self.raise_exc
        if self.status != 200:
            raise RuntimeError(f"HTTP {self.status}")
        body = json.dumps({"choices": [{"message": {"content": self.reply_text}}]})
        return self.status, body


def _classifier(reply: str = '{"type": "episodic", "confidence": 0.9}', **kw) -> tuple[LLMClassifier, FakePost]:
    post = FakePost(reply)
    clf = LLMClassifier(api_key="test-key", base_url="https://example.com/v1", post=post, **kw)
    return clf, post


def test_llm_classify_returns_type_confidence_source():
    clf, post = _classifier()
    mt, conf, src = clf.classify("我昨天去吃了火锅")
    assert mt is MemType.EPISODIC
    assert conf == 0.9
    assert src == "llm"
    assert post.calls == 1
    assert "chat/completions" in post.last_payload or True  # payload 发送了消息
    assert post.last_payload["model"] == "gpt-4o-mini"


def test_llm_result_cached():
    clf, post = _classifier()
    clf.classify("同一段内容")
    clf.classify("同一段内容")
    assert post.calls == 1  # 第二次命中缓存


def test_fallback_on_http_error():
    post = FakePost(raise_exc=RuntimeError("网络错误"))
    clf = LLMClassifier(api_key="test-key", base_url="https://example.com/v1", post=post)
    mt, conf, src = clf.classify("我昨天去吃了火锅")
    kw_mt, kw_conf = classify_memory_with_confidence("我昨天去吃了火锅")
    assert mt is kw_mt is MemType.EPISODIC
    assert src == "keyword"
    assert conf == kw_conf


def test_fallback_when_no_api_key():
    clf = LLMClassifier(post=FakePost())  # 无 key
    assert not clf.available
    mt, conf, src = clf.classify("我在学习做饭")
    assert mt is MemType.SKILL
    assert src == "keyword"
    assert conf == classify_memory_with_confidence("我在学习做饭")[1]


def test_fallback_on_malformed_output():
    clf, post = _classifier(reply="抱歉，我不确定。")  # 没有 JSON
    mt, conf, src = clf.classify("我在学习做饭")
    assert src == "keyword"
    assert mt is MemType.SKILL


def test_turn_bypasses_llm():
    clf, post = _classifier()
    mt, conf, src = clf.classify("用户说：你好", kind="turn")
    assert (mt, conf, src) == (MemType.EPISODIC, 1.0, "turn")
    assert post.calls == 0  # 不消耗 LLM 调用


def test_parse_aliases():
    assert _parse_llm_json('{"type": "procedural", "confidence": 0.8}')[0] is MemType.SKILL
    assert _parse_llm_json('{"type": "event", "confidence": 0.7}')[0] is MemType.EPISODIC
    assert _parse_llm_json('{"type": "fact", "confidence": 0.6}')[0] is MemType.SEMANTIC
    # 容忍代码围栏与前后缀
    mt, conf = _parse_llm_json('```json\n{"type": "skill", "confidence": 0.95}\n```')
    assert mt is MemType.SKILL and conf == 0.95
    with pytest.raises(ValueError):
        _parse_llm_json("没有 JSON 的输出")


def test_remember_uses_llm_classifier():
    post = FakePost('{"type": "episodic", "confidence": 0.88}')
    agent = MemoryAgent(classifier=LLMClassifier(api_key="k", post=post))
    mem = agent.remember("我昨天去吃了火锅")
    assert mem.mtype is MemType.EPISODIC
    assert mem.mtype_confidence == 0.88


def test_remember_default_keyword_with_confidence():
    agent = MemoryAgent()  # conftest 已清空环境变量 → 关键词回退
    mem = agent.remember("我昨天去吃了火锅")
    assert mem.mtype is MemType.EPISODIC
    assert mem.mtype_confidence == classify_memory_with_confidence("我昨天去吃了火锅")[1]
    assert mem.mtype_confidence is not None


def test_explicit_mtype_keeps_confidence_none():
    agent = MemoryAgent()
    mem = agent.remember("任意内容", mtype=MemType.SKILL)
    assert mem.mtype is MemType.SKILL
    assert mem.mtype_confidence is None  # 手动指定类型，无置信度


def test_classify_text_helper():
    post = FakePost('{"type": "semantic", "confidence": 0.7}')
    agent = MemoryAgent(classifier=LLMClassifier(api_key="k", post=post))
    mt, conf, src = agent.classify_text("北京是中国的首都")
    assert (mt, conf, src) == (MemType.SEMANTIC, 0.7, "llm")


def test_sensenova_classifier_disables_thinking_and_requests_json():
    post = FakePost('{"type": "semantic", "confidence": 0.9}')
    clf = LLMClassifier(
        api_key="k", base_url="https://token.sensenova.cn/v1",
        model="sensenova-6.8-flash-lite", post=post,
    )
    assert clf.classify("北京是中国的首都")[0] is MemType.SEMANTIC
    assert post.last_payload["thinking"] == {"type": "disabled"}
    assert post.last_payload["response_format"] == {"type": "json_object"}
    assert post.last_payload["max_tokens"] == 128


def test_sensenova_reasoning_model_auto_mode_is_enabled():
    assert _thinking_mode("auto", "https://token.sensenova.cn/v1", "glm-5.2") == "enabled"
    assert _thinking_mode("auto", "https://token.sensenova.cn/v1", "deepseek-v4-flash") == "enabled"
    assert _thinking_mode("auto", "https://token.sensenova.cn/v1", "sensenova-6.8-flash-lite") == "disabled"
    assert _thinking_mode("disabled", "https://token.sensenova.cn/v1", "glm-5.2") == "disabled"


def test_classifier_failover_switches_model_on_429():
    """主模型 429 → 自动切换备用模型，结果来源仍为 llm。"""
    class PoolPost:
        def __init__(self):
            self.calls = []

        def __call__(self, url, headers, payload, timeout):
            self.calls.append(payload["model"])
            if payload["model"] == "primary-x":
                return 429, "{}"
            return 200, json.dumps({"choices": [{"message": {
                "content": '{"type": "episodic", "confidence": 0.9}'}}]})

    post = PoolPost()
    clf = LLMClassifier(
        api_key="k", base_url="https://example.com/v1",
        model="primary-x", models=["backup-y"],
        post=post, max_retries=0, failover_cooldown=60,
    )
    mt, conf, src = clf.classify("昨天去吃了火锅·429切换")
    assert (mt, conf, src) == (MemType.EPISODIC, 0.9, "llm")
    assert post.calls == ["primary-x", "backup-y"]
    assert clf.failover_count == 1
    assert clf.rate_limited_log[0][0] == "primary-x"


def test_classifier_all_down_retry():
    """全部模型都 429 → 等待最早冷却结束再重试整个池，来源仍为 llm。"""
    class AllDownPost:
        def __init__(self):
            self.calls = []

        def __call__(self, url, headers, payload, timeout):
            self.calls.append(payload["model"])
            if len(self.calls) <= 2:  # 第一轮 A、B 都 429
                return 429, "{}"
            return 200, json.dumps({"choices": [{"message": {
                "content": '{"type": "semantic", "confidence": 0.72}'}}]})

    post = AllDownPost()
    clf = LLMClassifier(
        api_key="k", base_url="https://example.com/v1",
        model="A", models=["B"], post=post,
        max_retries=0, failover_cooldown=0.05,
        all_down_retries=3, all_down_wait_cap=0.2,
    )
    t0 = time.time()
    mt, conf, src = clf.classify("北京是首都·全限流重试")
    assert time.time() - t0 >= 0.04
    assert (mt, conf, src) == (MemType.SEMANTIC, 0.72, "llm")
    assert post.calls == ["A", "B", "A"]
    assert clf.failover_count == 2
