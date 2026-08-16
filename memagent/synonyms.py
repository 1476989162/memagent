"""查询同义扩展：问法与记忆措辞不一致时的检索增强。

字符 n-gram 嵌入对措辞敏感——记忆里存的是"我昨天去吃了火锅"，
用户却问"昨天中午用了什么餐"时重叠不足会漏检。这里对查询做**变体扩展**：

- 人称互换：疑问句里的"你/您"（指用户自己）替换成"我"——\n  如「你叫什么名字」→「我叫什么名字」，与「我叫小林」重叠激增；
- 同义词替换：词族内**罕见词替换为常见词**（组首）——\n  如「用餐」→「吃」、「姓名」→「名字」，方向固定为书面→口语；
- 检索时对每个变体求相似度取最大值，原始查询始终在变体里，故不会变差。

变体数上限 max_variants（默认 8），小记忆量下开销可忽略。
"""

from __future__ import annotations

# 词族：组首为最常见/最可能出现在记忆里的词（口语），替换方向=命中词→组首。
# 命中词已是组首时不生成无益变体，把扩展额度留给其他命中组。
SYNONYM_GROUPS: list[list[str]] = [
    ["吃", "食用", "用餐", "就餐", "尝一尝"],
    ["看", "观看", "瞧", "阅览"],
    ["说", "讲", "告诉", "提到", "谈及"],
    ["喜欢", "喜爱", "中意", "偏爱"],
    ["去", "前往", "来到"],
    ["买", "购买", "购置"],
    ["学", "学习", "练习", "训练"],
    ["会", "能", "可以", "能够"],
    ["名字", "姓名", "称呼"],
    ["住", "居住", "定居"],
    ["工作", "上班", "职业", "任职"],
    ["生日", "生辰", "出生日期"],
    ["昨天", "昨日", "前一天"],
    ["今天", "今日"],
    ["明天", "明日", "第二天"],
    ["爬山", "登山", "攀山"],
    ["跑步", "慢跑", "晨跑"],
    ["游泳", "戏水"],
]

# 人称互换：疑问句里"你/您"常指用户自己，替换为"我"以匹配事实记忆的表述。
PRONOUN_SWAPS: list[tuple[str, str]] = [("您", "我"), ("你", "我")]


SHORT_QUERY_LEN = 3  # 查询/主题少于该字数视为"短"：rel 易被哈希碰撞主导


def is_short_query(topic: str, short_len: int = SHORT_QUERY_LEN) -> bool:
    """查询是否算"短"（len(去空白) < short_len）——短查询重排触发的唯一判据。"""
    return len(topic.strip()) < short_len


def substring_priority_order(items, topic: str, content_of, strength_of,
                             short_len: int = SHORT_QUERY_LEN, score_of=None):
    """短查询子串优先重排：内容含查询词的条目排前面，消除哈希嵌入泛化命中
    把无关条目顶到前面的干扰。

    子串检查**大小写不敏感**（两侧都 lower）——与 n-gram 嵌入的归一化语义对齐：
    「AI」查「ai」、英文记忆「GPU」查「gpu」都能命中。

    组内排序默认按**强度**降序（strength_of）；传入 score_of 时改用该提取器
    （如 rel×强度 = total），让低相关但高强度的含词条目不再压过高相关条目。

    items 为任意结构列表；content_of / strength_of 分别提取内容与强度。
    返回 (重排后的列表, 是否发生重排)——长查询不重排、直接原样返回。
    """
    if is_short_query(topic, short_len):
        topic_l = topic.strip().lower()
        score = score_of or strength_of
        return (
            sorted(items, key=lambda it: (topic_l not in content_of(it).lower(), -score(it))),
            True,
        )
    return items, False


def expand_query(text: str, max_variants: int = 8) -> list[str]:
    """把查询扩展为一组检索变体（原始查询恒在首位）。"""
    variants = [text]
    # 1) 人称互换（单向：你/您 → 我）——「你昨天去了哪里」→「我昨天去了哪里」，
    #    与「我昨天去爬山」类事实记忆直接重叠；即使句中原有"我"，替换变体
    #    也能创造"我记得…"这类与记忆重叠的片段，max 取最大不会变差。
    for old, new in PRONOUN_SWAPS:
        if old in text:
            variants.append(text.replace(old, new))
            if len(variants) >= max_variants:
                return variants[:max_variants]
    # 2) 同义词替换：每个命中词替换为组首（常见词），每词一个变体
    for group in SYNONYM_GROUPS:
        common = group[0]
        for w in group[1:]:
            if w in text and common not in text:
                variants.append(text.replace(w, common))
                if len(variants) >= max_variants:
                    return variants[:max_variants]
    return variants[:max_variants]
