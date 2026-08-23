# -*- coding: utf-8 -*-
"""检索质量基准（eval_retrieval.py）的守护测试。

- 数据集良构性：正例唯一、类别合法、干扰项在位
- 评测库同源性：强度完全一致 → 排序差异只来自相关性
- 指标结构：单调性与取值范围、分类别齐全
- 确定性锚点：固定时钟 + 哈希嵌入下，有独特词面的题必须排第一——
  这些锚点守护「同义扩展 / 子串重排 / 相关性排序」链路不回归。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMAGENT_TEST", "1")

from eval_retrieval import CASES, CAT_LABELS, build_agent, evaluate  # noqa: E402
from memagent.memory import MemType  # noqa: E402


def test_dataset_wellformed():
    assert len(CASES) == 50
    poss = [c["pos"] for c in CASES]
    assert len(set(poss)) == len(poss), "正例内容必须互不相同"
    queries = [c["q"] for c in CASES]
    assert len(set(queries)) == len(queries), "查询必须互不相同"
    for c in CASES:
        assert c["cat"] in CAT_LABELS
        assert c["q"].strip() and c["pos"].strip()
        assert 1 <= len(c["neg"]) <= 4
        assert c["pos"] not in c["neg"]
        assert all(n != c["q"] for n in c["neg"])


def test_store_uniform_and_complete():
    """强度完全同源：importance/类型/时刻一致，排序差异只来自相关性。"""
    agent, ids = build_agent()
    mems = agent.store.all()
    by_id = {m.id for m in mems}
    assert set(ids.values()) <= by_id                      # 全部正例+干扰入库
    assert {round(m.importance, 6) for m in mems} == {0.5}
    assert {m.mtype for m in mems} == {MemType.SEMANTIC}
    assert len({m.last_access for m in mems}) == 1         # 同一时刻写入


def test_metrics_structure():
    agent, ids = build_agent()
    rep = evaluate(agent, ids)
    o = rep["overall"]
    assert o["n"] == 50
    for key in ("r1", "r3", "r5", "mrr"):
        assert 0.0 <= o[key] <= 1.0
    assert o["r1"] <= o["r3"] <= o["r5"]                   # 单正例指标单调
    assert set(rep["by_cat"]) == set(CAT_LABELS)
    for m in rep["by_cat"].values():
        assert m["n"] >= 6
    assert len(rep["rows"]) == 50                          # 逐例行可追溯


def test_anchor_cases_rank_first():
    """确定性锚点（固定时钟+哈希嵌入，逐例顺序与 evaluate 一致）：

    覆盖近义区分（独特词面）与短查询（子串重排）两条机制链路。
    """
    anchors = [
        "我去重庆那次吃的火锅是什么样子的？",   # discriminate：九宫格独特词面
        "信用卡的尾号是多少？",                 # discriminate：1024 vs 6688
        "我的工资卡是哪家银行？",               # discriminate：工商 vs 建设
        "我说过自己爱吃哪种水果吗？",           # discriminate：荔枝 vs 过敏芒果
        "我周四晚上有什么安排？",               # discriminate：加班 vs 瑜伽
        "手机号",                               # short：子串重排命中完整号码
        "wifi密码",                             # short：home8888 独特词面
    ]
    agent, ids = build_agent()
    rep = evaluate(agent, ids)
    rank_by_q = {r["q"]: r["rank"] for r in rep["rows"]}
    for q in anchors:
        assert rank_by_q.get(q) == 1, f"锚点未排第一：{q} → rank={rank_by_q.get(q)}"
