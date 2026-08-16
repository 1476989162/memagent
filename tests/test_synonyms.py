"""查询同义扩展测试：变体生成、人称互换、检索命中改进、开关兼容。"""

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.embedding import cosine_similarity, embed_text
from memagent.synonyms import expand_query, substring_priority_order


def test_original_query_first():
    assert expand_query("你好")[0] == "你好"


def test_pronoun_swap():
    v = expand_query("您叫什么名字")
    assert "我叫什么名字" in v  # 您 → 我
    v2 = expand_query("你记得我的名字吗")
    assert "我记得我的名字吗" in v2  # 你 → 我（句中已有"我"也替换，无伤大雅）


def test_synonym_replacement():
    v = expand_query("昨天中午用餐了吗")
    assert any("吃" in x for x in v)  # 用餐 → 吃
    v2 = expand_query("请问您的姓名")
    assert any("名字" in x for x in v2)  # 姓名 → 名字
    assert "请问我的姓名" in v2          # 您的 → 我的


def test_common_word_no_useless_variant():
    # "吃"是组首，不生成无益变体；无同义词/人称命中 → 只有原文
    assert expand_query("我昨天吃了火锅") == ["我昨天吃了火锅"]


def test_max_variants_cap():
    long_q = "昨天中午用餐了吗 请问您的姓名 观看电影 前往爬山"
    assert len(expand_query(long_q, max_variants=4)) <= 4


def _first_rel(cfg, memo, q):
    a = MemoryAgent(cfg=cfg)
    a.remember(memo, importance=0.1)
    return a.retrieve(q, k=1)[0].relevance


def test_retrieval_improved_by_expansion():
    memo, q = "我昨天去吃了火锅", "昨天中午用餐了吗"
    off = _first_rel(AgentConfig(query_expansion=False, reconsolidate=False), memo, q)
    on = _first_rel(AgentConfig(query_expansion=True, reconsolidate=False), memo, q)
    assert on > off


def test_pronoun_swap_improves_identity_match():
    off = _first_rel(AgentConfig(query_expansion=False, reconsolidate=False), "我叫小林", "您叫什么名字")
    on = _first_rel(AgentConfig(query_expansion=True, reconsolidate=False), "我叫小林", "您叫什么名字")
    assert on > off


def test_expansion_off_matches_original_behavior():
    """query_expansion=False 时 rel 等于原始查询的余弦 × boost（无变体参与）。"""
    from memagent.embedding import cosine_similarity, embed_text

    a = MemoryAgent(cfg=AgentConfig(query_expansion=False, reconsolidate=False))
    mem = a.remember("我昨天去吃了火锅", importance=0.1)
    h = a.retrieve("昨天中午用餐了吗", k=1)[0]
    expected = cosine_similarity(embed_text("昨天中午用餐了吗"), mem.embedding)
    assert abs(h.relevance - expected) < 1e-9


def test_substring_priority_order_short_topic():
    """短查询（<3 字）：含查询词的条目排前，组内按强度降序。"""
    items = [("开发决策：对照实验靠可注入时钟", 0.44), ("开发决策：遗忘斜率用触底时间", 0.30)]
    out, reranked = substring_priority_order(
        items, "触底", content_of=lambda x: x[0], strength_of=lambda x: x[1],
    )
    assert reranked
    assert "触底" in out[0][0]
    assert [i for i in out] == sorted(out, key=lambda x: ("触底" not in x[0], -x[1]))


def test_substring_priority_order_normal_topic_unchanged():
    """长查询（≥3 字）不重排，原样返回且 reranked=False。"""
    items = [("开发决策：语义化滞回", 0.4), ("开发决策：循环导入", 0.5)]
    out, reranked = substring_priority_order(
        items, "语义化", content_of=lambda x: x[0], strength_of=lambda x: x[1],
    )
    assert not reranked
    assert out == items  # 顺序不变


def test_substring_priority_order_short_topic_no_hits():
    """短查询但无条目含查询词：顺序仍按强度（重排逻辑不破坏），flag=True。"""
    items = [("开发决策：甲", 0.3), ("开发决策：乙", 0.5)]
    out, reranked = substring_priority_order(
        items, "触底", content_of=lambda x: x[0], strength_of=lambda x: x[1],
    )
    assert reranked
    assert [x[1] for x in out] == [0.5, 0.3]  # 强度降序


def test_retrieve_short_query_substring_priority():
    """核心 retrieve：短查询（<3 字）直接返回子串优先排序——即使目标记忆强度更低。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    target = a.remember("遗忘斜率对比用触底时间而非斜率比", importance=0.05)
    noise = a.remember("对照实验靠可注入时钟确定性快进", importance=0.9)
    hits = a.retrieve("触底", k=2)
    target_h = next(h for h in hits if h.memory is target)
    noise_h = next(h for h in hits if h.memory is noise)
    assert target_h.strength < noise_h.strength  # 按强度/总量目标本应排后面
    assert hits[0].memory is target              # 但子串优先把它排最前
    assert hits.index(target_h) < hits.index(noise_h)


def test_retrieve_short_query_substring_priority_includes_beyond_k():
    """重排发生在截断前：子串命中的记忆即使按 total 排在 k 名之外也会被召回。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    a.remember("语义化双阈值滞回避免振荡", importance=0.9)
    a.remember("循环导入用函数级导入解决", importance=0.9)
    a.remember("遗忘斜率对比用触底时间而非斜率比", importance=0.05)
    hits = a.retrieve("触底", k=1)
    assert "触底" in hits[0].memory.content  # 含词记忆挤进 top1


def test_substring_priority_order_case_insensitive():
    """子串检查大小写不敏感，与 n-gram 嵌入的归一化语义对齐。"""
    items = [
        ("开发决策：对照实验靠可注入时钟", 0.44),
        ("开发决策：AI 分类走 OpenAI 兼容接口", 0.30),
    ]
    out, reranked = substring_priority_order(
        items, "ai", content_of=lambda x: x[0], strength_of=lambda x: x[1],
    )
    assert reranked
    assert "AI" in out[0][0]  # 小写查询命中大写内容
    assert "AI" not in out[1][0]
    # 反向：大写查询命中小写内容
    items2 = [("开发决策：对照实验", 0.9), ("开发决策：ai 分类链路", 0.3)]
    out2, _ = substring_priority_order(
        items2, "AI", content_of=lambda x: x[0], strength_of=lambda x: x[1],
    )
    assert "ai" in out2[0][0]


def test_substring_priority_order_matches_cold_summary_case_insensitive():
    """Cold 记忆形态：查询词只在摘要（summary）里、content 不含——子串检查
    经 content_of 提取器覆盖摘要，且大小写不敏感（「ai」↔ 摘要「AI…」双向）。"""
    content_of = lambda x: x[0] + (x[1] or "")  # content + summary（retrieve 的提取器）
    items = [
        ("用户聊过一次项目背景", "开发决策：AI 分类链路已跑通", 0.4),  # 词只在摘要
        ("对照实验靠可注入时钟", "", 0.9),                           # 不含词
    ]
    out, reranked = substring_priority_order(
        items, "ai", content_of=content_of, strength_of=lambda x: x[2],
    )
    assert reranked
    assert out[0][0] == "用户聊过一次项目背景"  # 摘要命中 → 排最前
    assert "ai" in out[0][1].lower()            # 命中来自摘要（小写查询 → 大写摘要）
    # 反向：大写查询 → 小写摘要
    items2 = [
        ("用户聊过一次项目背景", "开发决策：ai 分类链路已跑通", 0.4),
        ("对照实验靠可注入时钟", "", 0.9),
    ]
    out2, _ = substring_priority_order(
        items2, "AI", content_of=content_of, strength_of=lambda x: x[2],
    )
    assert out2[0][0] == "用户聊过一次项目背景" and "ai" in out2[0][1]


def test_summary_reembedding_rel_difference():
    """摘要重嵌入 vs 原始 content 嵌入：被摘要丢掉的词 rel 归零、
    摘要保留的词 rel 不降（摘要更短更稠密，甚至反升）。"""
    content = "我昨天去学了 python 和 GPU 部署"
    summary = "我昨天去学了 python"          # GPU/部署 被摘要丢掉
    e_c, e_s = embed_text(content), embed_text(summary)
    # 丢掉的词：rel 归零
    assert cosine_similarity(embed_text("GPU"), e_s) < 0.01 < cosine_similarity(embed_text("GPU"), e_c)
    assert cosine_similarity(embed_text("部署"), e_s) < 0.01 < cosine_similarity(embed_text("部署"), e_c)
    # 保留的词：rel 不降反升（0.49 → 0.66）
    assert cosine_similarity(embed_text("python"), e_s) > cosine_similarity(embed_text("python"), e_c)


def test_demote_reembedding_drops_rel_below_gate():
    """同一记忆对象 demote 前后：摘要重嵌入使 content-only 词的 rel 跌破 0.05 门槛
    （Warm 0.164 → Cold 0.000）——记忆从"语义可检索"降为"仅子串可搜"。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    m = a.store.add("我昨天去学了 python 和 GPU 部署", importance=0.3)
    r_warm = next(h.relevance for h in a.retrieve("部署", k=3) if h.memory is m)
    m.demote_to_cold("我昨天去学了 python")
    r_cold = next(h.relevance for h in a.retrieve("部署", k=3) if h.memory is m)
    assert r_warm > 0.05      # Warm：content 嵌入可见
    assert r_cold < r_warm    # Cold：摘要重嵌入后 rel 大降
    assert r_cold < 0.05      # 跌破相关门槛（语义检索不可及）


def test_rerank_rescues_cold_content_only_word():
    """重排兜底：content 含查询词但摘要丢弃（rel≈0）的 Cold 记忆，短查询子串优先
    仍把它找回排最前——即使其 total 远低于碰撞噪声（0.000 < 0.238）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("开发决策：遗忘斜率对比用触底时间而非斜率比", importance=0.05)
    cold.demote_to_cold("开发决策：遗忘斜率对比")   # 摘要丢「触底」
    noise = a.store.add("对照实验靠可注入时钟确定性快进", importance=0.9)
    hits = a.retrieve("触底", k=2)
    cold_h = next(h for h in hits if h.memory is cold)
    noise_h = next(h for h in hits if h.memory is noise)
    assert cold_h.total < noise_h.total     # rel≈0 → total 低于噪声
    assert hits[0].memory is cold           # 但子串优先（content 含词）兜底找回


def test_rerank_orders_rescued_cold_by_total_within_group():
    """兜底不越权：同含词的组内按 rel×强度排序——被兜底的 Cold（rel≈0）排在
    有真实 rel 的同词记忆之后（0.243 在前、0.000 在后）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a.store.add("开发决策：GPU 部署细节", importance=0.05)
    cold.demote_to_cold("开发决策：")           # 摘要丢「部署」
    real = a.store.add("开发决策：部署流程记录", importance=0.05)
    hits = a.retrieve("部署", k=2)
    assert hits[0].memory is real       # 真实 rel 排前
    assert hits[1].memory is cold        # 兜底 Cold 排后（组内按 total）
    assert hits[0].relevance > hits[1].relevance


def test_retrieve_rerank_matches_cold_summary_case_insensitive():
    """核心 retrieve：Cold 记忆按摘要参与重排——查询词只在摘要里（content 不含）
    且大小写不敏感（「ai」命中摘要「AI…」）——含词 Cold 记忆压过碰撞噪声排最前；
    关闭重排时噪声（total 更高）排前，证明命中来自重排的摘要检查。"""
    a_on = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    cold = a_on.store.add("用户聊过一次项目背景", importance=0.1)
    cold.demote_to_cold("开发决策：AI 分类链路已跑通（OpenAI 兼容）")
    a_on.store.add("对照实验靠可注入时钟确定性快进", importance=0.9)
    hits = a_on.retrieve("ai", k=2)
    assert hits and hits[0].memory is cold   # Cold 摘要命中排最前
    assert cold.tier.value == "cold" and "AI" in cold.summary
    assert "ai" not in cold.content.lower()  # 词只在摘要、不在 content

    a_off = MemoryAgent(cfg=AgentConfig(reconsolidate=False, rerank_short_query=False))
    c2 = a_off.store.add("用户聊过一次项目背景", importance=0.1)
    c2.demote_to_cold("开发决策：AI 分类链路已跑通（OpenAI 兼容）")
    a_off.store.add("对照实验靠可注入时钟确定性快进", importance=0.9)
    hits_off = a_off.retrieve("ai", k=2)
    assert hits_off and hits_off[0].memory is not c2  # 关重排：噪声 total 更高排前


def test_retrieve_rerank_matches_cold_summary_via_real_sleep():
    """真实压缩管线端到端：sleep 把相似 Warm 记忆合并成 Cold 簇（摘要=抽取原文、
    content 只保留第一源）——短查询「ai」命中**只在合并摘要里**的「AI」（content
    不含），含词 Cold 记忆压过 total 更高的碰撞噪声排最前（默认配置）。"""
    clock = [1000.0]
    a = MemoryAgent(
        cfg=AgentConfig(reconsolidate=False, sleep_interval_turns=999),
        now_fn=lambda: clock[0],
    )
    target = a.store.add("我昨天去学了 python", importance=0.1)           # episodic
    a.store.add("我昨天去学了 python 和 AI 编程", importance=0.1)         # skill：合并进簇
    noise = a.store.add("对照实验靠可注入时钟确定性快进", importance=0.9)
    for m in a.store.all():
        m.last_access = clock[0] - 200 * 24 * 3600  # 老化 → 满足各类型压缩条件
    a.sleep()
    assert target.tier.value == "cold"
    assert "AI" in target.summary and "AI" not in target.content  # 词只在合并摘要
    hits = a.retrieve("ai", k=3)
    assert hits and hits[0].memory is target   # 摘要命中排最前
    assert hits[0].total < next(h.total for h in hits if h.memory is noise)  # 压过更高 total 的噪声


def test_substring_priority_order_score_of_overrides_strength():
    """score_of 传入时组内按该分数降序（而非纯强度）；缺省回退 strength_of。"""
    items = [
        ("开发决策：触底验证", 0.9, 0.30),   # (内容, 强度, 分数 rel×强度)
        ("开发决策：触底时间", 0.4, 0.35),
    ]
    # 缺省：按纯强度（0.9 排前）
    out, _ = substring_priority_order(
        items, "触底", content_of=lambda x: x[0], strength_of=lambda x: x[1],
    )
    assert "触底验证" in out[0][0]
    # score_of：按分数（0.35 > 0.30 → 触底时间排前），强度高的不压过高相关条目
    out2, _ = substring_priority_order(
        items, "触底", content_of=lambda x: x[0], strength_of=lambda x: x[1],
        score_of=lambda x: x[2],
    )
    assert "触底时间" in out2[0][0]


def test_retrieve_rerank_group_internal_order_by_total():
    """核心 retrieve：含词记忆组内按 rel×强度（total）排序——低相关但高强度的
    含词记忆不再压过更高相关（但强度略低）的含词记忆。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    low_rel = a.remember("开发决策：遗忘斜率对比用触底时间而非每 τ 斜率比", importance=0.9)
    high_rel = a.remember("开发决策：触底", importance=0.1)
    hits = a.retrieve("触底", k=2)
    by_content = {h.memory.content: h for h in hits}
    assert "触底" in by_content[high_rel.content].memory.content  # 都含词
    assert by_content[low_rel.content].strength > by_content[high_rel.content].strength
    assert by_content[high_rel.content].total > by_content[low_rel.content].total
    assert hits[0].memory is high_rel  # total 高的排最前（rel 信号）


def test_retrieve_normalizes_query_before_scoring():
    """retrieve() 打分前统一小写：再巩固内容钩子收到归一化后的查询
    （「Python 写代码」→ 钩子收到「python 写代码」），与 rel 计算语义一致。"""
    seen = []
    a = MemoryAgent(
        cfg=AgentConfig(),
        content_updater=lambda m, q, lab: (seen.append(q), m.content)[1],
    )
    a.remember("我喜欢用 Python 写代码", importance=0.1)
    hits = a.retrieve("Python 写代码", k=1)
    assert hits and seen
    assert all(q == q.lower() for q in seen)  # 钩子收到的查询已小写
    assert "Python" not in seen[0] and "python" in seen[0]


def test_retrieve_case_insensitive_scoring_long_query():
    """长英文查询（≥3 字、不触发重排）打分大小写无关：
    rel(「Python 写代码」) == rel(「python 写代码」)——归一化在 rel 计算前生效。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False, query_expansion=False))
    a.remember("我喜欢用 Python 写代码", importance=0.1)
    r_upper = a.retrieve("Python 写代码", k=1)[0].relevance
    r_lower = a.retrieve("python 写代码", k=1)[0].relevance
    assert abs(r_upper - r_lower) < 1e-9


def test_retrieve_short_query_case_insensitive():
    """核心 retrieve：短英文查询大小写不敏感（「ai」命中「AI…」记忆）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    a.remember("开发决策：AI 分类走 OpenAI 兼容接口", importance=0.05)
    a.remember("对照实验靠可注入时钟确定性快进", importance=0.9)
    hits = a.retrieve("ai", k=2)
    assert "AI" in hits[0].memory.content  # 大小写不敏感 → 含词记忆排最前


def test_retrieve_short_query_case_insensitive_custom_len():
    """自定义阈值下英文短词同样大小写不敏感（「gpu」命中「GPU…」记忆）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False, rerank_short_len=5))
    a.remember("开发决策：GPU 显存不够时降批", importance=0.05)
    a.remember("对照实验靠可注入时钟确定性快进", importance=0.9)
    hits = a.retrieve("gpu", k=2)
    assert "GPU" in hits[0].memory.content


def test_substring_priority_order_custom_short_len():
    """函数级：short_len 参数自定义短词阈值（默认 3 → 4 字不重排，放宽到 5 → 重排）。"""
    items = [("开发决策：对照实验靠可注入时钟", 0.44), ("开发决策：遗忘斜率对比用触底时间", 0.30)]
    out, reranked = substring_priority_order(
        items, "触底时间", content_of=lambda x: x[0], strength_of=lambda x: x[1],
    )
    assert not reranked and out == items  # 默认阈值 3：4 字查询不算短
    out, reranked = substring_priority_order(
        items, "触底时间", content_of=lambda x: x[0], strength_of=lambda x: x[1],
        short_len=5,
    )
    assert reranked and "触底时间" in out[0][0]  # 自定义阈值 5：子串优先


def test_retrieve_rerank_flag_disabled():
    """AgentConfig.rerank_short_query=False：短查询也不重排，返回严格按 total 降序。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False, rerank_short_query=False))
    a.remember("遗忘斜率对比用触底时间而非斜率比", importance=0.05)
    a.remember("对照实验靠可注入时钟确定性快进", importance=0.9)
    hits = a.retrieve("触底", k=2)
    totals = [h.total for h in hits]
    assert totals == sorted(totals, reverse=True)  # 关闭后与旧版一致


def test_retrieve_rerank_custom_short_len():
    """AgentConfig.rerank_short_len 自定义阈值：4 字查询默认按 total（碰撞噪声排前），
    阈值放宽到 5 后子串优先（含词记忆排最前）。"""
    def _build(short_len):
        a = MemoryAgent(cfg=AgentConfig(reconsolidate=False, rerank_short_len=short_len))
        a.remember("遗忘斜率对比用触底时间而非斜率比", importance=0.05)
        a.remember("对照实验靠可注入时钟确定性快进", importance=0.9)
        return a.retrieve("触底时间", k=2)

    hits_default = _build(3)   # 默认：4 字不算短 → 严格按 total（噪声 total 更高排前）
    assert "触底时间" not in hits_default[0].memory.content
    hits_custom = _build(5)    # 放宽：4 字算短 → 子串优先
    assert "触底时间" in hits_custom[0].memory.content


def test_retrieve_long_query_unchanged_total_order():
    """长查询（≥3 字）不重排：返回顺序仍按 total 降序（与旧版一致）。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    a.remember("语义化双阈值滞回避免振荡", importance=0.1)
    a.remember("对照实验靠可注入时钟", importance=0.9)
    hits = a.retrieve("语义化", k=3)
    totals = [h.total for h in hits]
    assert totals == sorted(totals, reverse=True)


def test_expansion_never_worse_for_true_relevant():
    """原始查询恒在变体里 → 开扩展的 rel ≥ 关扩展的 rel。"""
    memo, q = "我叫小林，喜欢爬山", "我喜欢登山"
    off = _first_rel(AgentConfig(query_expansion=False, reconsolidate=False), memo, q)
    on = _first_rel(AgentConfig(query_expansion=True, reconsolidate=False), memo, q)
    assert on >= off - 1e-9
