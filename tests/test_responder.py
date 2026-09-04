"""LLM 回复生成器测试：上下文注入、无记忆直接回答、无 key/出错回退模板、agent 集成、
429 多模型池自动切换（含全部限流后等待重试）与人设自主演化（persona_extras / set_persona）。"""

import json
import time

from memagent import MemoryAgent
from memagent.responder import LLMResponder


class FakePost:
    """记录请求并返回固定文本回复。"""

    def __init__(self, reply="你好，我是小林。"):
        self.reply = reply
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        return 200, json.dumps({"choices": [{"message": {"content": self.reply}}]})


def _mk(reply="你好，我是小林。"):
    post = FakePost(reply)
    return LLMResponder(api_key="k", base_url="https://example.com/v1", post=post), post


def test_not_available_without_key():
    """无 OPENAI_API_KEY（conftest 已清环境变量）→ 不可用且 respond 抛错。"""
    r = LLMResponder()
    assert not r.available
    try:
        r.respond("hi")
        assert False, "应抛 RuntimeError"
    except RuntimeError:
        pass


def test_memory_context_injected_into_prompt():
    r, post = _mk("（LLM）我记得你昨天吃了火锅。")
    txt = r.respond("你记得我昨天吃了什么吗", memories=[("我昨天去吃了火锅", "episodic", 0.42)])
    assert txt == "（LLM）我记得你昨天吃了火锅。"
    user = post.calls[-1]["messages"][-1]["content"]
    assert "用户问题：你记得我昨天吃了什么吗" in user
    assert "我昨天去吃了火锅" in user          # 记忆内容注入 prompt
    assert "[episodic]" in user               # 类型标注
    assert "0.42" in user                     # 强度标注
    assert "直接凭常识回答" not in user


def test_no_memory_direct_answer_mode():
    r, post = _mk()
    r.respond("天空为什么是蓝色的", memories=None)
    user = post.calls[-1]["messages"][-1]["content"]
    assert "检索到的相关记忆" not in user
    assert "直接凭常识回答" in user           # 无记忆 → LLM 直接回答模式


def test_agent_falls_back_to_template_without_responder():
    agent = MemoryAgent()  # 不配 responder
    reply, hits = agent.respond("北京是中国的首都")
    assert "我记得" in reply or "我还不太了解" in reply


def test_agent_uses_responder_when_available():
    r, post = _mk("（LLM 回答）我记得你昨天吃了火锅。")
    agent = MemoryAgent(responder=r)
    agent.remember("我昨天去吃了火锅", importance=0.1)
    reply, hits = agent.respond("我昨天吃了什么")
    assert reply == "（LLM 回答）我记得你昨天吃了火锅。"
    assert "我还不太了解" not in reply
    assert "我昨天去吃了火锅" in post.calls[-1]["messages"][-1]["content"]  # 记忆真的注入了


def test_agent_injects_summary_for_cold_hit():
    """Cold 命中（via_summary）：LLM 路径注入摘要文本而非深藏 content——
    与模板路径（_template_reply 显示摘要）行为一致；命中词只在摘要里。"""
    r, post = _mk("（LLM）好的。")
    agent = MemoryAgent(responder=r)
    m = agent.store.add("用户聊过一次项目背景", importance=0.1)
    m.demote_to_cold("开发决策：AI 分类")
    _reply, hits = agent.respond("ai")
    assert any(h.via_summary for h in hits)          # 确实是索引触发命中
    user = post.calls[-1]["messages"][-1]["content"]
    assert "开发决策：AI 分类" in user               # 注入摘要
    assert "用户聊过一次项目背景" not in user         # 不注入深藏 content


def test_agent_falls_back_to_template_on_responder_error():
    def boom(*a, **k):
        raise RuntimeError("网络错误")
    r = LLMResponder(api_key="k", base_url="https://example.com/v1", post=boom)
    agent = MemoryAgent(responder=r)
    agent.remember("我昨天去吃了火锅", importance=0.1)
    reply, hits = agent.respond("我昨天吃了什么")
    assert "我记得" in reply  # 出错 → 模板回退，检索链路不受影响


def test_agent_direct_answer_mode_when_no_relevant_memory():
    """查询与记忆无关时，responder 收到的 memories 应为 None（LLM 直接回答）。"""
    captured = {}

    class CapPost:
        def __call__(self, url, headers, payload, timeout):
            captured["user"] = payload["messages"][-1]["content"]
            return 200, json.dumps({"choices": [{"message": {"content": "ok"}}]})

    # 空 store：retrieve 无任何命中 → responder 收到 memories=None
    r = LLMResponder(api_key="k", base_url="https://example.com/v1", post=CapPost())
    agent = MemoryAgent(responder=r)
    agent.respond("地球是圆的吗")
    assert "（没有检索到相关记忆" in captured["user"]


def test_sensenova_responder_disables_thinking_and_reads_content():
    class SensePost:
        def __init__(self):
            self.payload = None

        def __call__(self, url, headers, payload, timeout):
            self.payload = payload
            return 200, json.dumps({
                "choices": [{"message": {
                    "role": "assistant", "content": "好的。",
                    "reasoning": "internal reasoning",
                }}]
            })

    post = SensePost()
    r = LLMResponder(
        api_key="k", base_url="https://token.sensenova.cn/v1",
        model="sensenova-6.8-flash-lite", post=post,
    )
    assert r.respond("你好") == "好的。"
    assert post.payload["thinking"] == {"type": "disabled"}
    assert post.payload["max_tokens"] == 1024


def test_sensenova_reasoning_model_enables_thinking_in_auto_mode():
    class SensePost:
        def __init__(self):
            self.payload = None

        def __call__(self, url, headers, payload, timeout):
            self.payload = payload
            return 200, json.dumps({
                "choices": [{"message": {
                    "role": "assistant", "content": "这是最终答案。",
                    "reasoning": "internal reasoning",
                }}]
            })

    post = SensePost()
    r = LLMResponder(
        api_key="k", base_url="https://token.sensenova.cn/v1",
        model="glm-5.2", post=post,
    )
    assert r.respond("请回答") == "这是最终答案。"
    assert post.payload["thinking"] == {"type": "enabled"}
    assert post.payload["max_tokens"] == 16384  # sensenova 推理模型地板，默认 4096 不够


def test_reasoning_only_response_retries_with_thinking_disabled():
    class ReasoningOnlyThenContent:
        def __init__(self):
            self.payloads = []

        def __call__(self, url, headers, payload, timeout):
            self.payloads.append(json.loads(json.dumps(payload)))
            if len(self.payloads) == 1:
                return 200, json.dumps({
                    "choices": [{"message": {"content": "", "reasoning": "draft"}}]
                })
            return 200, json.dumps({
                "choices": [{"message": {"content": "最终正文"}}]
            })

    post = ReasoningOnlyThenContent()
    responder = LLMResponder(
        api_key="k",
        base_url="https://token.sensenova.cn/v1",
        model="glm-5.2",
        post=post,
        max_retries=0,
    )
    assert responder.respond("写作") == "最终正文"
    assert post.payloads[0]["thinking"] == {"type": "enabled"}
    assert post.payloads[1]["thinking"] == {"type": "disabled"}


def test_respond_max_tokens_per_call_override():
    """逐调用 max_tokens 覆盖默认值——长输出（代码生成/写章）必须能放宽，
    否则默认 1024 会把输出拦腰截断（FoxTable 660 轮里 286 轮截断的根因）。"""
    post = FakePost("代码")
    r = LLMResponder(api_key="k", base_url="https://example.com/v1", post=post)
    r.respond("写代码", max_tokens=4096)
    assert post.calls[-1]["max_tokens"] == 4096
    # 不传时回落到既有默认（非 thinking 模型 1024），行为向后兼容
    r.respond("聊天")
    assert post.calls[-1]["max_tokens"] == 1024


def test_max_tokens_constructor_and_call_priority():
    class RecPost(FakePost):
        pass

    post = RecPost("ok")
    r = LLMResponder(api_key="k", base_url="https://example.com/v1",
                     post=post, max_tokens=2048)
    r.respond("默认走构造值")
    assert post.calls[-1]["max_tokens"] == 2048
    r.respond("逐调用覆盖构造值", max_tokens=8192)
    assert post.calls[-1]["max_tokens"] == 8192


def test_reasoning_only_retry_keeps_max_tokens_override():
    """thinking-only 空回复关闭 thinking 重试时，max_tokens 保持解析后的值
    （不再回落到硬编码 4096——逐调用覆盖对重试请求同样生效）。"""
    class ReasoningOnlyThenContent:
        def __init__(self):
            self.payloads = []

        def __call__(self, url, headers, payload, timeout):
            self.payloads.append(json.loads(json.dumps(payload)))
            if len(self.payloads) == 1:
                return 200, json.dumps({
                    "choices": [{"message": {"content": "", "reasoning": "draft"}}]
                })
            return 200, json.dumps({
                "choices": [{"message": {"content": "最终正文"}}]
            })

    post = ReasoningOnlyThenContent()
    responder = LLMResponder(
        api_key="k",
        base_url="https://token.sensenova.cn/v1",
        model="glm-5.2",
        post=post,
        max_retries=0,
    )
    assert responder.respond("写作", max_tokens=2048) == "最终正文"
    assert post.payloads[0]["max_tokens"] == 2048
    assert post.payloads[1]["max_tokens"] == 2048  # 重试不丢覆盖值


def test_sensenova_failover_skips_rate_limited_model_and_uses_backup():
    class PoolPost:
        def __init__(self):
            self.calls = []

        def __call__(self, url, headers, payload, timeout):
            self.calls.append(payload["model"])
            if payload["model"] == "glm-5.2":
                return 429, "{}"
            return 200, json.dumps({"choices": [{"message": {"content": "备用成功"}}]})

    post = PoolPost()
    r = LLMResponder(
        api_key="k", base_url="https://token.sensenova.cn/v1",
        model="glm-5.2", models=["deepseek-v4-flash"],
        post=post, max_retries=0, failover_cooldown=60,
    )
    assert r.respond("你好") == "备用成功"
    assert post.calls == ["glm-5.2", "deepseek-v4-flash"]
    assert r.model == "deepseek-v4-flash"
    assert r.failover_count == 1
    assert r.rate_limited_log and r.rate_limited_log[0][0] == "glm-5.2"


def test_all_models_429_wait_and_retry_until_one_frees():
    """全部模型都 429 → 等待最早冷却结束再重试整个池（一直切换到不限流的模型）。"""
    class AllDownPost:
        def __init__(self):
            self.calls = []

        def __call__(self, url, headers, payload, timeout):
            self.calls.append(payload["model"])
            if len(self.calls) <= 2:  # 第一轮 A、B 都 429
                return 429, "{}"
            return 200, json.dumps({"choices": [{"message": {"content": "冷却后成功"}}]})

    post = AllDownPost()
    r = LLMResponder(
        api_key="k", base_url="https://example.com/v1",
        model="A", models=["B"],
        post=post, max_retries=0,
        failover_cooldown=0.05, all_down_retries=3, all_down_wait_cap=0.2,
    )
    t0 = time.time()
    assert r.respond("你好") == "冷却后成功"
    assert time.time() - t0 >= 0.04        # 确实等待了冷却
    assert post.calls == ["A", "B", "A"]  # 第一轮 A/B 都 429 → 等冷却 → A 成功
    assert r.model == "A"
    assert r.failover_count == 2


def test_all_models_429_budget_exhausted_raises():
    class Always429:
        def __call__(self, url, headers, payload, timeout):
            return 429, "{}"

    r = LLMResponder(
        api_key="k", base_url="https://example.com/v1",
        model="A", models=["B"], post=Always429(),
        max_retries=0, failover_cooldown=100,
        all_down_retries=1, all_down_wait_cap=0.01,
    )
    try:
        r.respond("你好")
        assert False, "应抛 RuntimeError"
    except RuntimeError as e:
        assert "冷却" in str(e)


def test_persona_extras_injected_into_system_prompt():
    class CapPost:
        def __init__(self):
            self.system = None

        def __call__(self, url, headers, payload, timeout):
            self.system = payload["messages"][0]["content"]
            return 200, json.dumps({"choices": [{"message": {"content": "（LLM）好的。"}}]})

    post = CapPost()
    r = LLMResponder(api_key="k", base_url="https://example.com/v1", post=post, persona="novelist")
    r.respond("继续写下一章", persona_extras="• 主角：林尘\n• 境界体系：炼气→筑基→金丹")
    assert "夜航墨客" in post.system                 # 人设
    assert "主角：林尘" in post.system                # 演化档案注入
    assert "境界体系" in post.system
    assert "记忆增强型对话助手" in post.system          # 基础提示仍在


def test_set_persona_rebuilds_system_prompt():
    r, post = _mk()
    assert "夜航墨客" not in r.system_prompt          # 初始无 persona
    r.set_persona("novelist")
    assert "夜航墨客" in r.system_prompt
    assert r.persona.startswith("你是一名长期创作型中文小说家")
    r.set_persona("我是诗人")
    assert "诗人" in r.persona and "夜航墨客" not in r.persona
    r.set_persona(None)
    assert r.persona is None
    assert "夜航墨客" not in r.system_prompt


def test_agent_persona_wires_responder():
    """MemoryAgent(persona=...) 自动创建 LLMResponder；无 key 时回退模板。"""
    agent = MemoryAgent(persona="novelist")
    assert agent.persona == "novelist"
    assert agent.responder is not None
    assert not agent.responder.available          # conftest 已清 key
    reply, hits = agent.respond("北京是中国的首都")
    assert "我记得" in reply or "我还不太了解" in reply


def test_agent_passes_persona_sheet_to_responder():
    """remember_setting 的设定记忆按重要性排序注入 system prompt——人设自主演化。"""
    post = FakePost("（LLM）继续写下一章。")
    r = LLMResponder(api_key="k", base_url="https://example.com/v1", post=post, persona="novelist")
    agent = MemoryAgent(responder=r, persona="novelist")
    agent.remember_setting("境界体系：炼气→筑基→金丹→元婴", importance=0.8)
    agent.remember_setting("主角：林尘，青州林氏旁支少年", importance=0.9)
    sheet = agent.persona_sheet()
    assert sheet is not None
    assert sheet.index("主角：林尘") < sheet.index("境界体系")   # 重要性降序
    assert all(m.kind == "setting" for m in agent.store.all())
    agent.respond("继续写下一章")
    system = post.calls[-1]["messages"][0]["content"]
    assert "夜航墨客" in system
    assert "主角：林尘" in system
    assert "境界体系" in system

    # 无设定记忆 → 不注入档案（人设仍在，行为不变）
    post2 = FakePost("（LLM）好的。")
    r2 = LLMResponder(api_key="k", base_url="https://example.com/v1", post=post2, persona="novelist")
    agent2 = MemoryAgent(responder=r2, persona="novelist")
    assert agent2.persona_sheet() is None
    agent2.respond("你好")
    system2 = post2.calls[-1]["messages"][0]["content"]
    assert "夜航墨客" in system2
    assert "• " not in system2
