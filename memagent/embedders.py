"""可插拔语义嵌入后端：接入真语义模型，突破哈希 n-gram 的词汇级上限。

用法（进程内单例，所有 Agent 共用）：
    from memagent.embedding import set_embedder
    from memagent.embedders import RemoteEmbedder
    set_embedder(RemoteEmbedder(base_url="https://api.example.com/v1",
                                api_key="...", model="text-embedding-3-small"))

    # 或本地 sentence-transformers：
    # pip install memagent-local[embed-local]
    # set_embedder(LocalEmbedder("paraphrase-multilingual-MiniLM-L12-v2"))

后端只需提供 `.dim`（int）与 `.embed(text) -> list[float]`（L2 归一化）。
换后端后，存量记忆由 Memory.from_dict 的维度失配检测自动重建向量
（content 为准，Cold 用摘要）；进程内已加载的记忆需重启或自行重建。
词汇重叠保底、子串重排、n-gram 倒排等文本级逻辑不依赖后端。

RemoteEmbedder 用纯 stdlib urllib（与 llm.py 同风格），兼容任意
OpenAI 风格 /embeddings 端点（OpenAI / DeepSeek / 本地 vLLM、Ollama 等）。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class RemoteEmbedder:
    """OpenAI 兼容 /embeddings 远程嵌入。

    base_url 形如 https://api.example.com/v1；维度首次响应后自动缓存，
    无需手配。请求失败带指数退避重试（默认 3 次），全部失败抛 RuntimeError。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dim: int | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        embed_path: str = "embeddings",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self._dim = dim
        self.timeout = timeout
        self.retries = retries
        self.embed_path = embed_path

    @property
    def dim(self) -> int:
        if self._dim is None:
            # 用空串试探一次以获取维度（远程端点通常接受并返回空向量）
            self._dim = len(self.embed(""))
        return self._dim

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        data = self._request(payload)
        try:
            vec = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"嵌入端点响应格式异常（期望 data[0].embedding）：{data!r}"
            ) from e
        if self._dim is None:
            self._dim = len(vec)
        return vec

    def _urlopen(self, req: urllib.request.Request):
        """底层 HTTP 发送钩子（测试可替换，不做网络调用）。"""
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _request(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/{self.embed_path}"
        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with self._urlopen(req) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"嵌入端点 HTTP {e.code}: {e.read()[:200]!r}") from e
            except Exception as e:  # 网络/超时/JSON 解析
                last_err = e
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"嵌入端点重试 {self.retries} 次仍失败: {last_err}")


class LocalEmbedder:
    """本地 sentence-transformers 嵌入（跨措辞语义）。

    依赖较重（torch），放入可选 extra：`pip install memagent-local[embed-local]`。
    未安装时构造即抛 RuntimeError 并给出安装指引。
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "LocalEmbedder 需要 sentence-transformers："
                "pip install 'memagent-local[embed-local]'"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]


class FastEmbedder:
    """本地 ONNX 嵌入（fastembed）——无 torch 的轻量真语义后端。

    依赖：pip install fastembed（onnxruntime 实现，Python 3.14 有 wheel）。
    首次构造会自动下载模型权重到本地缓存（bge-small-zh-v1.5 约 100MB），
    之后完全离线。适合中文场景的默认模型：BAAI/bge-small-zh-v1.5。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "FastEmbedder 需要 fastembed：pip install fastembed"
            ) from e
        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        if self.dim is None:
            self.dim = len(next(self._model.embed(["dim"])))

    dim: int | None = None  # 首次编码后确定

    def embed(self, text: str) -> list[float]:
        import math

        vec = next(self._model.embed([text]))
        v = [float(x) for x in vec]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
