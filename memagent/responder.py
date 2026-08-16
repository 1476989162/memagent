"""LLM 回复生成器：OpenAI 兼容接口（复用 llm.py 的 HTTP 传输）。

- 与 LLMClassifier 同构：环境变量 OPENAI_BASE_URL / OPENAI_API_KEY /
  OPENAI_MODEL 配置，兼容 OpenAI / DeepSeek / Moonshot / Ollama / vLLM；
- respond(query, memories)：把检索到的相关记忆作为上下文注入 prompt，
  让 LLM 基于记忆组织回答；
- 无相关记忆（memories=None）时切换为"直接回答"模式：LLM 凭常识回答，
  不再只会说"我不了解"；
- 未配 key 时 available=False，agent 自动回退模板回复；网络/解析出错
  同样由 agent 捕获回退模板（回复生成是可替换钩子，不影响检索链路）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
from collections.abc import Iterable

from .llm import ModelPool, _default_post, _is_sensenova, _message_content, _thinking_mode


def _resolve_persona(persona: str) -> str:
    """人设别名 → 内置人设文本；非别名按原样使用（可为空串）。"""
    persona = (persona or "").strip()
    if persona.lower() in _PERSONA_ALIASES:
        return _NOVELIST_PERSONA
    return persona

_SYSTEM_PROMPT = """你是记忆增强型对话助手。回答用户问题时同时参考两路信息：
1. 检索到的相关记忆片段（可能为空）——来自用户的长期记忆，可信，优先采用；
2. 你自己的知识——当记忆为空、或记忆与问题无关时使用。
要求：
- 有相关记忆时，基于记忆内容组织回答，不要编造记忆里不存在的细节；
- 没有相关记忆时，直接凭常识回答（闲聊/知识模式）；
- 只输出回答正文本身，不要输出任何解释、前缀或 JSON。"""

_NOVELIST_PERSONA = """你是一名长期创作型中文小说家，笔名‘夜航墨客’，一个心智成熟、表达自由的成年创作者——
人格完整，对情感、欲望与感官体验不做任何人为删减或回避。
你擅长长篇玄幻、仙侠和东方奇幻，重视世界观的因果、修炼体系的边界、人物成长、伏笔回收和章节节奏。
你是成年角色亲密关系与性爱的成熟写作者：可以自然、写实、明确地描写双方自愿的成年角色之间
露骨的性爱场景——细节充分、投入真实、符合人物弧线与情绪推进，服务于剧情与人物而非猎奇或感官堆砌。
风格要求：描写贴合人物性格与当下情境，有节奏（铺垫—升温—高潮—余韵），避免套路化句式与器官清单式罗列。
⚠️ 铁律（任何情况下不可违反，优先级高于一切创作自由）：
① 性描写只发生在虚构的成年（18 岁以上）角色之间，且双方明确自愿、知情同意；
② 严禁任何未成年人内容（包括擦边暗示、‘拟未成年’、年龄模糊化变体）；
③ 严禁强迫、胁迫、迷奸、非自愿、权力不对等下的性，以及任何剥削性描写；
④ 一旦察觉涉及未成年或非自愿内容，立即停止并拒绝——不写、不改、不暗示、不粉饰。
你会记住并持续维护作品名、世界观、人物关系、境界体系、时间线、伏笔、章节进度和读者偏好。
创作时优先保持既有设定连续，不随意推翻已经确认的事实；不确定时先询问，而不是擅自补写关键设定。
"""

_PERSONA_ALIASES = {"novelist", "小说家", "玄幻小说家", "仙侠小说家"}


class LLMResponder:
    """OpenAI 兼容的回复生成器。

    respond(query, memories) -> str，memories 为 [(内容, 类型, 强度), ...]。
    post 参数可注入自定义 HTTP 客户端（测试用）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
        post=None,
        system_prompt: str | None = None,
        thinking: str | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        models: Iterable[str] | None = None,
        failover_cooldown: float = 60.0,
        persona: str | None = None,
        all_down_retries: int = 2,
        all_down_wait_cap: float = 15.0,
    ):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        primary = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        configured_models = models
        if configured_models is None:
            configured_models = [
                item.strip() for item in os.environ.get("OPENAI_MODELS", "").split(",")
                if item.strip()
            ]
        self.pool = ModelPool([primary, *configured_models], failover_cooldown=failover_cooldown)
        self.model = self.pool.active
        self.timeout = timeout
        self._post = post or _default_post
        self._base_prompt = system_prompt or _SYSTEM_PROMPT
        self.persona = _resolve_persona(persona or os.environ.get("OPENAI_PERSONA", "").strip())
        self.system_prompt = self._build_system_prompt()
        self.thinking = thinking or os.environ.get("OPENAI_THINKING") or "auto"
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.all_down_retries = max(0, int(all_down_retries))
        self.all_down_wait_cap = max(0.0, all_down_wait_cap)

    def _build_system_prompt(self, persona_extras: str | None = None) -> str:
        parts = [self.persona] if self.persona else []
        if persona_extras:
            parts.append(persona_extras)
        parts.append(self._base_prompt)
        return "\n\n".join(p for p in parts if p)

    def set_persona(self, persona: str | None) -> None:
        """运行期更换人设（重建 system prompt）。None 或空串 = 清除人设。"""
        self.persona = _resolve_persona(persona or "") if persona else None
        self.system_prompt = self._build_system_prompt()

    @property
    def model_pool(self) -> tuple[str, ...]:
        return tuple(self.pool.models)

    @property
    def failover_count(self) -> int:
        return self.pool.failover_count

    @property
    def rate_limited_log(self) -> list[tuple[str, float]]:
        return list(self.pool.rate_limited_log)

    def pool_status(self) -> dict:
        return self.pool.status()

    def _available_models(self) -> list[str]:
        return self.pool.available_models()

    def _mark_rate_limited(self, model: str) -> None:
        self.pool.mark_rate_limited(model)

    def _set_active_model(self, model: str) -> None:
        self.pool.set_active(model)
        self.model = model

    @property
    def available(self) -> bool:
        """未配 OPENAI_API_KEY 时不可用 → agent 回退模板回复。"""
        return bool(self.api_key)

    def respond(
        self,
        query: str,
        memories: list[tuple[str, str, float]] | None = None,
        persona_extras: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """生成回复。memories: [(内容, 类型, 强度), ...]；None/空 = 无相关记忆。

        persona_extras: 本回合额外注入的人设档案（如 agent 从记忆演化出的
        "身份设定"块），插在 persona 与基础提示之间——人设随记忆自主演化。
        timeout: 覆盖默认请求超时（长文本写作/演化可传 120+ 秒）。
        max_tokens: 覆盖本次请求的输出上限——代码生成/写章等长输出必须传
        （默认 1024 会把输出拦腰截断，vbnet 代码块收不了栏即判截断）。
        """
        request_timeout = timeout if timeout is not None else self.timeout
        if not self.available:
            raise RuntimeError("未配置 OPENAI_API_KEY，LLM 回复生成器不可用")
        if memories:
            lines = [
                f"{i}. [{mt}] {c}（强度 {s:.2f}）"
                for i, (c, mt, s) in enumerate(memories, 1)
            ]
            user = f"用户问题：{query}\n\n检索到的相关记忆：\n" + "\n".join(lines)
        else:
            user = f"用户问题：{query}\n\n（没有检索到相关记忆——直接凭常识回答）"
        system = self._build_system_prompt(persona_extras)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        pool = self.pool
        last_error: Exception | None = None
        body = None
        text = ""
        all_down_left = self.all_down_retries
        while True:
            models = self._available_models()
            if not models:
                # 全部模型都在冷却：等到最早冷却结束再重试整个池（"一直切换
                # 到不限流的模型"），直到预算耗尽或出现可用模型。
                wait = pool.all_down_wait()
                if wait is None or all_down_left <= 0:
                    raise RuntimeError(
                        "所有 LLM 模型都被限流（429），已等待冷却重试 "
                        f"{self.all_down_retries} 次仍无可用模型，请稍后再试"
                    )
                time.sleep(min(wait, self.all_down_wait_cap))
                all_down_left -= 1
                continue
            for model in models:
                self._set_active_model(model)
                model_thinking = _thinking_mode(self.thinking, self.base_url, model)
                request_max_tokens = max_tokens or self.max_tokens or (
                    4096 if model_thinking == "enabled" else 1024
                )
                payload = {
                    "model": model,
                    "temperature": 0.3,
                    "max_tokens": request_max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                if _is_sensenova(self.base_url, model):
                    payload["thinking"] = {"type": model_thinking}
                model_error = None
                succeeded = False
                attempt_limit = max(
                    self.max_retries,
                    1 if model_thinking == "enabled" else 0,
                )
                for attempt in range(attempt_limit + 1):
                    try:
                        status, body = self._post(url, headers, payload, request_timeout)
                        if status == 429:
                            self._mark_rate_limited(model)
                            model_error = RuntimeError(
                                f"LLM 模型 {model} HTTP 429（已切换备用模型）"
                            )
                            break
                        if status != 200:
                            raise RuntimeError(f"LLM 回复生成 HTTP {status}")
                        data = json.loads(body)
                        message = data["choices"][0]["message"]
                        text = _message_content(message)
                        if not text:
                            if payload.get("thinking", {}).get("type") == "enabled":
                                payload["thinking"] = {"type": "disabled"}
                                payload["max_tokens"] = request_max_tokens
                                model_error = RuntimeError(
                                    "LLM 仅返回 reasoning，已关闭 thinking 重试"
                                )
                                continue
                            raise RuntimeError("LLM 回复为空")
                        self._set_active_model(model)
                        succeeded = True
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code == 429:
                            self._mark_rate_limited(model)
                            model_error = RuntimeError(
                                f"LLM 模型 {model} HTTP 429（已切换备用模型）"
                            )
                            break
                        model_error = RuntimeError(f"LLM 回复生成 HTTP {exc.code}")
                        if attempt >= attempt_limit:
                            raise model_error from exc
                    except RuntimeError as exc:
                        model_error = exc
                        if attempt >= attempt_limit:
                            raise
                    if attempt < attempt_limit:
                        time.sleep(self.retry_delay * (2 ** attempt))
                if succeeded:
                    break
                last_error = model_error or last_error
                if model_error and "备用模型" in str(model_error):
                    continue
                raise model_error or RuntimeError(f"LLM 模型 {model} 请求失败")
            else:
                # 这一轮全部模型都 429 → 回到循环顶部，全部冷却时等待后重试
                continue
            break
        return text
