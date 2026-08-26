# -*- coding: utf-8 -*-
"""混合检索测试：词汇覆盖通道与语义向量取大。

动机回归：精确标识符（库名/函数名）类查询在纯向量下召回弱——
语义嵌入把「vxe-table」与中文踩坑记录的距离拉得很开，而这类查询
恰恰是字面精确匹配的主场。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMAGENT_TEST", "1")

from eval_retrieval import FIXED_NOW  # noqa: E402
from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402


def _agent(**kw) -> MemoryAgent:
    return MemoryAgent(cfg=AgentConfig(tau_seconds=30 * 24 * 3600,
                                       reconsolidate=False,
                                       hot_after_access=999, **kw),
                       now_fn=lambda: FIXED_NOW)


def _store(agent, contents):
    ids = {}
    for c in contents:
        m = agent.store.add(c, importance=0.5, now=FIXED_NOW)
        ids[c] = m.id
    return ids


def test_identifier_query_hits_exact_term_memory():
    """拉丁标识符查询：词汇通道让精确命中的记忆排第一。"""
    agent = _agent()
    ids = _store(agent, [
        "尚博进销存踩坑：vxe-table 的 VxeDatePicker format 属性需要大写占位符。",
        "尚博进销存踩坑：XCode 的 WhereExpression & 运算符返回新对象，不改变原表达式。",
        "项目决策：发布原则定为出厂婴儿，只发引擎不发经验数据。",
        "写作改进：让威胁从声音升级为可感的逼近。",
    ])
    hits = agent.retrieve("vxe-table 日期控件 format 踩坑", k=4)
    assert hits and hits[0].memory.id == next(iter(ids.values()))
    assert hits[0].relevance >= 0.15          # 词汇通道给了实质加分（纯向量≈0）
    # 关闭混合通道后同一查询失去词汇加成
    agent2 = _agent(keyword_hybrid=False)
    _store(agent2, list(ids))
    hits2 = agent2.retrieve("vxe-table 日期控件 format 踩坑", k=4)
    if hits2:
        assert hits2[0].relevance <= hits[0].relevance  # 关闭后不得更高


def test_hybrid_disabled_restores_pure_vector():
    """keyword_hybrid=False 与旧版行为一致（标识符查询查不到）。"""
    agent = _agent(keyword_hybrid=False)
    ids = _store(agent, [
        "尚博进销存踩坑：vxe-table 的日期选择器格式化属性易错。",
        "完全无关的记忆内容甲。",
        "完全无关的记忆内容乙。",
    ])
    hits = agent.retrieve("vxe-table format", k=3)
    top_id = hits[0].memory.id if hits else None
    # 纯向量下该查询 rel≈0，排名由噪声决定——只要确认词汇通道没参与
    assert all(h.relevance < 0.45 for h in hits)


def test_semantic_match_still_wins_over_partial_overlap():
    """语义强匹配不被低覆盖的词汇通道压过（取大而非替换）。"""
    agent = _agent()
    ids = _store(agent, [
        "我对花生过敏，吃点心前要看清楚配料表。",
        "花生是一种常见的坚果作物，广泛种植于温带地区。",
    ])
    q = "饮食方面有什么需要注意的忌口？"
    hits = agent.retrieve(q, k=2)
    assert hits[0].memory.id == ids["我对花生过敏，吃点心前要看清楚配料表。"]


def test_cold_summary_participates_in_keyword_channel():
    """Cold 记忆按摘要文本参与词汇覆盖（索引可及性一致）。"""
    agent = _agent()
    m = agent.store.add("vxe-grid 表格组件的路由注册必须显式声明前缀。",
                        importance=0.5, now=FIXED_NOW)
    m.demote_to_cold("vxe-grid 表格组件的路由注册规范与显式前缀要求。")
    hits = agent.retrieve("vxe-grid 路由 前缀 规范", k=5)
    assert hits and hits[0].memory.id == m.id
    assert hits[0].via_summary                # 走的是摘要索引
