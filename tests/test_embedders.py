"""可插拔语义嵌入后端测试：全局嵌入器替换、维度迁移、远程/本地后端。

核心断言：换语义后端后，哈希嵌入查不到的跨措辞等价（「技术栈」↔
「Python 程序员」）能通过检索命中——这正是 v0.3.2 要突破的词汇级上限。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from memagent import MemoryAgent, embed_text
from memagent.agent import AgentConfig
from memagent.embedding import (
    HashEmbedder, embedding_dim, get_embedder, set_embedder,
)
from memagent.embedders import LocalEmbedder, RemoteEmbedder
from memagent.memory import Memory, MemoryStore


class _FakeSemantic:
    """玩具语义嵌入器：跨措辞等价词映射到同一向量，其余各不同。

    dim=3。只用于验证「embedding 单点替换 → 检索行为变化」这条链路。
    """

    dim = 3
    # 同义词簇：同一簇内的词共享向量
    _CLUSTERS = {
        frozenset({"技术栈", "python 程序员", "写代码的语言"}): [1.0, 0.0, 0.0],
        frozenset({"火锅", "吃火锅", "聚餐"}): [0.0, 1.0, 0.0],
        frozenset({"开会", "会议"}): [0.0, 0.0, 1.0],
    }
    _BASE = [0.1, 0.1, 0.1]

    def embed(self, text: str) -> list[float]:
        t = text.lower().strip()
        for cluster, vec in self._CLUSTERS.items():
            if any(k in t for k in cluster):
                return list(vec)
        return list(self._BASE)


@pytest.fixture(autouse=True)
def _restore_embedder():
    saved = get_embedder()
    yield
    set_embedder(saved)


def test_set_embedder_switches_dim():
    set_embedder(_FakeSemantic())
    assert embedding_dim() == 3
    assert len(embed_text("任意文本")) == 3
    set_embedder(HashEmbedder())
    assert embedding_dim() == 1024


def test_semantic_cross_wording_retrieval():
    """跨措辞等价：哈希嵌入 rel≈0（词汇级查不到），语义后端能命中（单点替换）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    a.remember("我叫小林，是一名 Python 程序员", importance=0.7)
    # 哈希嵌入：词汇级，无共享 n-gram → rel 近零
    hash_rel = a.retrieve("我的技术栈是什么", k=1)[0].relevance
    assert hash_rel < 0.1
    # 换语义后端后：同一文本能命中（rel≈1）
    set_embedder(_FakeSemantic())
    b = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    b.remember("我叫小林，是一名 Python 程序员", importance=0.7)
    hits = b.retrieve("我的技术栈是什么", k=1)
    assert hits and hits[0].relevance > 0.9
    assert "python" in hits[0].memory.content.lower()


def test_from_dict_migrates_to_new_backend_dim():
    """换后端后，旧维度向量在 from_dict 里重建为新后端向量。"""
    set_embedder(_FakeSemantic())
    d = {
        "id": "abc123", "content": "我叫小林，是一名 Python 程序员",
        "tier": "warm", "embedding": [0.1] * 1024,  # 旧哈希维度
    }
    m = Memory.from_dict(d)
    assert len(m.embedding) == 3  # 重建为语义后端维度


def test_store_roundtrip_with_remote_dim(tmp_path):
    """换后端后 store 落盘/加载维度一致（维度失配迁移链完整）。"""
    set_embedder(_FakeSemantic())
    p = tmp_path / "mem.json"
    store = MemoryStore(str(p))
    store.add("技术栈", importance=0.5)
    store.save()
    store2 = MemoryStore(str(p))
    assert all(len(m.embedding) == 3 for m in store2.all())


def test_remote_embedder_request_and_dim():
    """RemoteEmbedder：请求体正确 + 维度缓存 + 响应解析。"""
    captured = {}

    def fake_request(payload: dict) -> dict:
        captured["payload"] = payload
        return {"data": [{"embedding": [1.0, 0.0, 0.0]}]}

    emb = RemoteEmbedder(base_url="https://example.com/v1",
                         api_key="sk-x", model="m1")
    emb._request = fake_request  # 注入假 HTTP 层
    vec = emb.embed("你好")
    assert vec == [1.0, 0.0, 0.0]
    assert emb.dim == 3
    assert captured["payload"] == {"model": "m1", "input": "你好"}
    assert emb.base_url == "https://example.com/v1"
    assert emb.embed_path == "embeddings"


def test_remote_embedder_retries_then_fails(monkeypatch):
    """远程后端：限流重试后仍失败 → 抛 RuntimeError。"""
    import urllib.error

    calls = []

    def flaky(req):
        calls.append(1)
        raise urllib.error.HTTPError("https://example.com/v1", 503, "down", None, None)

    emb = RemoteEmbedder(base_url="https://example.com/v1", retries=2, timeout=1)
    emb._urlopen = flaky
    monkeypatch.setattr("memagent.embedders.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        emb.embed("你好")
    assert len(calls) == 3  # 1 次 + 2 次退避重试


def test_local_embedder_missing_dependency_raises():
    """本地后端未安装依赖 → 清晰的安装指引。"""
    import sys as _sys

    saved = dict(_sys.modules)
    for mod in list(_sys.modules):
        if mod == "sentence_transformers" or mod.startswith("sentence_transformers."):
            del _sys.modules[mod]
    try:
        _sys.modules["sentence_transformers"] = None  # 模拟未安装
        with pytest.raises(RuntimeError, match="embed-local"):
            LocalEmbedder()
    finally:
        _sys.modules.pop("sentence_transformers", None)
        _sys.modules.update(saved)
