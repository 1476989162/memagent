"""LLM 记忆类型分类器：OpenAI 兼容接口（纯 stdlib urllib，零依赖）。

- 兼容任意 OpenAI 风格端点：OpenAI / DeepSeek / Moonshot / 本地 Ollama、vLLM 等；
- 环境变量：OPENAI_BASE_URL、OPENAI_API_KEY、OPENAI_MODEL；
- LLM 返回严格 JSON {"type": "skill|semantic|episodic", "confidence": 0~1}；
- 关键词规则（classify_memory_with_confidence）作为离线回退：未配 key、
  网络错误、输出解析失败时自动降级；
- 对话流水（kind="turn"）是硬规则，直接判情景类，不消耗 LLM 调用；
- 结果按内容缓存，避免重复调用。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import time
from pathlib import Path

from .memory import MemType, classify_memory_with_confidence


def _load_dotenv() -> None:
    """从 .env 文件加载环境变量（如果存在）。不依赖 python-dotenv。"""
    # 查找 .env：当前工作目录 → memagent 包所在项目的根目录
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",  # memagent 上一级
    ]
    for p in candidates:
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # 按 key 是否已存在判断（而非值真假）：显式置空的环境变量
                    # （如 OPENAI_API_KEY=）同样优先于 .env——与 setdefault 语义一致
                    if "=" in line and line.split("=", 1)[0].strip() not in os.environ:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
            except Exception:
                pass  # .env 加载失败不影响主流程


_load_dotenv()

# LLM 返回类型名的别名映射（容忍不同模型的措辞差异）
_TYPE_ALIASES = {
    "skill": MemType.SKILL, "procedural": MemType.SKILL, "procedure": MemType.SKILL,
    "semantic": MemType.SEMANTIC, "fact": MemType.SEMANTIC, "factual": MemType.SEMANTIC,
    "knowledge": MemType.SEMANTIC, "identity": MemType.SEMANTIC, "preference": MemType.SEMANTIC,
    "episodic": MemType.EPISODIC, "event": MemType.EPISODIC, "episode": MemType.EPISODIC,
    "experience": MemType.EPISODIC,
}

_SYSTEM_PROMPT = """分类：skill(技能/操作) | semantic(事实/知识) | episodic(事件/经历)。只输出JSON：{"type":"...","confidence":0到1}"""

# reasoning 模型（sensenova 等）的 _SYSTEM_PROMPT 需要精简，避免触发长篇推理
_SYSTEM_PROMPT_REASONING = """skill|semantic|episodic，输出JSON：{"type":"...","confidence":0.9}"""

_SENSENOVA_REASONING_MODELS = (
    "glm-5.2",
    "deepseek-v4-flash",
)


def _is_sensenova(base_url: str, model: str) -> bool:
    """Whether the endpoint/model needs SenseNova-specific request options."""
    target = f"{base_url} {model}".lower()
    return "sensenova" in target or "sensenova.cn" in target


def _thinking_mode(value: str | None, base_url: str, model: str) -> str:
    """Resolve auto/enabled/disabled for SenseNova's thinking parameter."""
    mode = (value or "auto").strip().lower()
    if mode in {"on", "true", "1", "enabled"}:
        return "enabled"
    if mode in {"off", "false", "0", "disabled"}:
        return "disabled"
    if mode != "auto":
        raise ValueError("thinking must be auto, enabled, or disabled")
    model_l = model.lower()
    if any(name in model_l for name in _SENSENOVA_REASONING_MODELS):
        return "enabled"
    return "disabled"


def _message_content(message: dict) -> str:
    """Extract text from OpenAI and common multimodal message shapes."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip() if content else ""


def _parse_llm_json(text: str) -> tuple[MemType, float]:
    """从 LLM 输出中提取 (类型, 置信度)。容忍代码围栏与前后缀文本。"""
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        raise ValueError("LLM 输出中没有 JSON")
    data = json.loads(m.group(0))
    raw = str(data.get("type", "")).strip().lower()
    mtype = _TYPE_ALIASES.get(raw)
    if mtype is None:
        raise ValueError(f"未知类型: {raw!r}")
    conf = float(data.get("confidence", 0.5))
    return mtype, max(0.0, min(1.0, conf))


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _default_post(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, str]:
    """用 stdlib urllib 发 POST，返回 (HTTP 状态码, 响应体)。

    带浏览器 User-Agent：部分 OpenAI 兼容端点（如 OpenCode Go）在
    Cloudflare 后面，默认 Python-UA 会被误判为机器人（HTTP 403/1010）。
    """
    headers = dict(headers)
    headers.setdefault("User-Agent", _BROWSER_UA)
    headers.setdefault("Accept", "application/json")
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


class ModelPool:
    """多模型池：429 时自动切换下一个模型，全部限流时等待冷却后重试。

    语义：
    - ``available_models()`` 返回当前轮转位置起、不在冷却中的模型；
    - ``mark_rate_limited(model)`` 给模型记冷却（failover_cooldown 秒）、
      轮转到下一个模型，并累计 failover_count / rate_limited_log；
    - ``all_down_wait()`` 在全部模型都在冷却时返回"等到最早冷却结束还需秒数"，
      否则 None——调用方据此 sleep 后重试整个池，实现"一直切换到不限流的模型"；
    - ``set_active(model)`` 记录当前正在使用的模型（供外部展示）。
    """

    def __init__(self, models: list[str], failover_cooldown: float = 60.0):
        self.models = list(dict.fromkeys(m for m in models if m))
        if not self.models:
            raise ValueError("模型池不能为空")
        self.failover_cooldown = max(0.0, failover_cooldown)
        self._index = 0
        self._cooldowns: dict[str, float] = {}
        self.failover_count = 0
        self.rate_limited_log: list[tuple[str, float]] = []
        self.active = self.models[0]

    def available_models(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        ordered = self.models[self._index:] + self.models[:self._index]
        return [m for m in ordered if self._cooldowns.get(m, 0.0) <= now]

    def mark_rate_limited(self, model: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._cooldowns[model] = now + self.failover_cooldown
        if model in self.models:
            self._index = (self.models.index(model) + 1) % len(self.models)
        self.failover_count += 1
        self.rate_limited_log.append((model, now))

    def all_down_wait(self, now: float | None = None) -> float | None:
        """全部模型都在冷却时返回还需等待的秒数；有可用模型时返回 None。"""
        now = time.time() if now is None else now
        if self.available_models(now):
            return None
        if not self._cooldowns:
            return None
        return max(0.0, min(self._cooldowns.values()) - now)

    def set_active(self, model: str) -> None:
        self.active = model
        if model in self.models:
            self._index = self.models.index(model)

    def status(self) -> dict:
        """池状态（CLI /persona、/models 展示用）。"""
        return {
            "active": self.active,
            "pool": list(self.models),
            "failover_count": self.failover_count,
            "recent_429": [(m, t) for m, t in self.rate_limited_log[-10:]],
        }


class LLMClassifier:
    """OpenAI 兼容的分类器，关键词规则做离线回退。

    classify(content, kind) -> (MemType, 置信度, 来源)，来源 ∈ {"llm", "keyword", "turn"}。
    post 参数可注入自定义 HTTP 客户端（测试用）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 15.0,
        post=None,
        cache: bool = True,
        retry_delay: float = 1.0,
        models: list[str] | None = None,
        failover_cooldown: float = 60.0,
        all_down_retries: int = 2,
        all_down_wait_cap: float = 15.0,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        primary = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        configured = models
        if configured is None:
            configured = [x.strip() for x in os.environ.get("OPENAI_MODELS", "").split(",") if x.strip()]
        self.pool = ModelPool([primary, *configured], failover_cooldown=failover_cooldown)
        self.model = self.pool.active
        self.timeout = timeout
        self._post = post or _default_post
        self._cache: dict[str, tuple] = {} if cache else None
        self.retry_delay = max(0.0, retry_delay)
        self.all_down_retries = max(0, int(all_down_retries))
        self.all_down_wait_cap = max(0.0, all_down_wait_cap)
        self.max_retries = max(0, int(max_retries))

    @property
    def models(self) -> list[str]:
        return list(self.pool.models)

    @property
    def failover_count(self) -> int:
        return self.pool.failover_count

    @property
    def rate_limited_log(self) -> list[tuple[str, float]]:
        return list(self.pool.rate_limited_log)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def classify(self, content: str, kind: str = "fact") -> tuple[MemType, float, str]:
        """返回 (类型, 置信度, 来源)。未配 key / 出错时自动回退关键词。"""
        if kind == "turn":  # 硬规则：对话流水必为情景类，不调 LLM
            return MemType.EPISODIC, 1.0, "turn"
        if self._cache is not None and content in self._cache:
            return self._cache[content]
        if not self.available:
            result = self._keyword(content, kind)
        else:
            try:
                result = self._llm(content)
            except Exception:
                result = self._keyword(content, kind)  # 离线回退
        if self._cache is not None:
            self._cache[content] = result
        return result

    def _keyword(self, content: str, kind: str) -> tuple[MemType, float, str]:
        mt, conf = classify_memory_with_confidence(content, kind)
        return mt, conf, "keyword"

    def _llm(self, content: str, max_retries: int | None = None) -> tuple[MemType, float, str]:
        max_retries = self.max_retries if max_retries is None else max_retries
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        pool = self.pool
        last_exc: Exception | None = None
        all_down_left = self.all_down_retries
        while True:
            models = pool.available_models()
            if not models:
                # 全部模型都在冷却：等到最早冷却结束再重试整个池（"一直切换
                # 到不限流的模型"），直到预算耗尽或出现可用模型。
                wait = pool.all_down_wait()
                if wait is None or all_down_left <= 0:
                    break
                time.sleep(min(wait, self.all_down_wait_cap))
                all_down_left -= 1
                continue
            for model in models:
                self.model = model
                pool.set_active(model)
                is_reasoning = _is_sensenova(self.base_url, model)
                prompt = _SYSTEM_PROMPT_REASONING if is_reasoning else _SYSTEM_PROMPT
                max_tokens = 128 if is_reasoning else 200
                rate_limited = False
                for attempt in range(max_retries + 1):
                    payload = {
                        "model": model,
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"{content}"},
                        ],
                    }
                    if is_reasoning:
                        payload["thinking"] = {"type": "disabled"}
                        payload["response_format"] = {"type": "json_object"}
                    try:
                        status, body = self._post(url, headers, payload, self.timeout)
                        if status == 429:
                            pool.mark_rate_limited(model)
                            last_exc = RuntimeError(f"LLM 分类模型 {model} HTTP 429（已切换备用模型）")
                            rate_limited = True
                            break
                        if status != 200 and attempt < max_retries:
                            raise RuntimeError(f"HTTP {status}")
                        if status != 200:
                            raise RuntimeError(f"LLM 分类 HTTP {status}")
                        data = json.loads(body)
                        msg = data["choices"][0]["message"]
                        text = _message_content(msg)
                        if not text.strip():
                            reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "") or ""
                            if reasoning:
                                text = reasoning
                        mtype, conf = _parse_llm_json(text)
                        return mtype, conf, "llm"
                    except Exception as e:
                        last_exc = e
                        if attempt < max_retries:
                            time.sleep(self.retry_delay * (2 ** attempt))
                if not rate_limited:
                    # 非 429 错误（重试耗尽）——不换模型，直接向上抛
                    raise last_exc
                # 429：换下一个模型继续这一轮
            # 这一轮全部模型都 429 → 回到循环顶部，全部冷却时等待后重试
        raise last_exc or RuntimeError("所有 LLM 分类模型都不可用")
    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()
