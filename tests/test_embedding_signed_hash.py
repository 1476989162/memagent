"""v0.3.1 嵌入升级回归测试：FNV 低位混洗消除系统性假相关 + 旧向量迁移。

背景：旧 256 维 FNV-1a 直接 `% 256` 取低位——低位无雪崩（末字符主导），
无关短查询（「火锅」对「北京是中国的首都」）得到系统性正相似度，短查询
时一两个碰撞格就能把无关记忆顶到检索前列（MCP 场景直接变成「假装查得到」）。

v0.3.1：DIM 1024 + 混洗修正，无关对回到 0.00~0.04；真信号永不抵消。
（符号哈希试验过但放弃：短查询+短记忆单共享 bigram 时伪碰撞会反向抵消
真信号，见 embedding.py 模块注释。）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memagent.embedding import DIM, cosine_similarity, embed_text
from memagent.memory import Memory, MemoryStore, Tier


def test_unrelated_texts_near_zero():
    """无关文本相似度 ≈ 0（混洗后回到高斯噪声），不再有系统性假相关。"""
    pairs = [
        ("火锅", "北京是中国的首都"),
        ("火锅", "用户偏好简洁回复"),
        ("ai", "对照实验靠可注入时钟确定性快进"),
        ("今天天气怎么样", "数据库连接池的配置参数"),
        # FNV 低位缺陷回归：加 _mix 混洗前，末字符主导低位会让这类
        # 零共享 gram 的对产生 0.13 级系统性假相关（v0.3.1 已修）
        ("符号哈希", "开发决策：【斩契】文学审校层：确定性句法节奏守卫+重写循环"),
    ]
    for a, b in pairs:
        sim = cosine_similarity(embed_text(a), embed_text(b))
        assert abs(sim) < 0.1, f"{a!r} vs {b!r}: {sim}"


def test_shared_gram_never_cancels():
    """单共享 bigram 的短查询/短记忆：词汇重叠保底兜底——符号哈希余弦曾
    被伪碰撞精确抵消到 0.0，retrieve() 的保底让「查得到」不被数学事故吞掉。"""
    from memagent import MemoryAgent
    from memagent.agent import AgentConfig

    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    a.remember("我叫小林", importance=0.5)
    hits = a.retrieve("我叫什么名字", k=1)
    assert hits and hits[0].memory.content == "我叫小林"
    assert hits[0].relevance > 0.15, f"我叫什么名字 vs 我叫小林: {hits[0].relevance}"


def test_identical_and_containment():
    """同文本 = 1.0；包含关系给出明确正相关。"""
    assert abs(cosine_similarity(embed_text("火锅"), embed_text("火锅")) - 1.0) < 1e-9
    sim = cosine_similarity(embed_text("火锅"), embed_text("我昨天去吃了火锅"))
    assert sim > 0.2


def test_dim_upgraded():
    assert DIM == 1024


def test_old_dim_embedding_reembedded_on_load():
    """旧版 256 维向量在 from_dict 里按当前嵌入器重建。"""
    d = {
        "id": "abc123", "content": "我昨天去吃了火锅",
        "tier": "warm", "embedding": [0.1] * 256,
    }
    m = Memory.from_dict(d)
    assert len(m.embedding) == DIM  # 已重建为当前维度
    assert abs(cosine_similarity(m.embedding, embed_text("我昨天去吃了火锅")) - 1.0) < 1e-9


def test_old_dim_cold_summary_reembedded():
    """Cold 记忆的索引向量重建时取摘要而非 content（与 __post_init__ 一致）。"""
    d = {
        "id": "cold000001", "content": "我昨天去吃了火锅",
        "tier": "cold", "summary": "火锅聚餐（已归档）",
        "embedding": [0.1] * 256,
    }
    m = Memory.from_dict(d)
    assert len(m.embedding) == DIM
    assert abs(cosine_similarity(m.embedding, embed_text("火锅聚餐（已归档）")) - 1.0) < 1e-9


def test_store_roundtrip_preserves_dim(tmp_path):
    """落盘/加载循环：向量维度与当前嵌入器一致。"""
    p = tmp_path / "mem.json"
    store = MemoryStore(str(p))
    store.add("北京是中国的首都", importance=0.5)
    store.save()
    store2 = MemoryStore(str(p))
    assert all(len(m.embedding) == DIM for m in store2.all())
    assert store2.get(list(store2._memories)[0]) is not None
