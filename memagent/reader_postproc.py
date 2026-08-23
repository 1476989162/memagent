# -*- coding: utf-8 -*-
"""读者友好度 post-processor：自动为未内嵌解释的专有术语注入解释短语。

策略：不再依赖 LLM 自觉遵守写作规则——在 LLM 生成文本后，程序化扫描术语
首次出现位置，若前后 25 字内无【有效解释性标志词】，则插入「——解释短语」。
每章每个术语最多解释 1 次。

出厂婴儿原则（v3）：**包内零词表**。术语解释是作品的人格数据，
从作品目录的 `term_explanations.json` 加载（{"术语": "——解释"}），
没有配置文件 = 功能静默关闭。机制随产品发布，词表随作品走。

关键修正（v2）：
- 破折号「——」在中文小说里大量用于叙述断句，不是解释标志
- 仅当「——」后接定义/描述性词汇（是/叫/为/本/指/即）时，才视为解释
"""
import json
import re
from pathlib import Path
from typing import List, Tuple, Union

# 纯解释性标志词（不含「——」这类会被误判的标点）
STRONG_EXPLAINERS: List[str] = [
    "（", "叫", "名为", "是指", "就是", "即", "原本叫", "实为",
    "本是", "原为", "原来叫", "即是", "意为",
]

# 破折号「——」只有在后面紧跟定义类词时才视为解释
DASH_DEFINERS: List[str] = [
    "是", "叫", "为", "本", "指", "即", "实", "种", "原", "意", "乃",
]


def _is_real_explanation(text: str, pos: int) -> bool:
    """检查 pos 位置前的 25 字和后 25 字内是否有【有效】解释。"""
    window_before = text[max(0, pos - 25): pos]
    window_after = text[pos : pos + 25]

    # 强标志词（括号/叫/名为/是指/就是/即/实为/本是...）
    for e in STRONG_EXPLAINERS:
        if e in window_before or e in window_after:
            return True

    # 破折号「——」：只有后面紧跟定义类词才算。
    # 前/后窗分别检索——原实现把拼接串下标当 window_after 下标用，整体错位。
    for window in (window_before, window_after):
        for m in re.finditer("——", window):
            # 破折号后 3 字内的内容
            after_dash = window[m.end(): m.end() + 3]
            if after_dash and any(d in after_dash for d in DASH_DEFINERS):
                return True

    return False


def load_terms_for_work(chapter_dir: Union[str, Path]) -> dict:
    """从作品目录加载术语表（出厂婴儿：包内零词表，作品自配）。

    查找顺序：<chapter_dir>/term_explanations.json → 其父目录。
    找不到或文件损坏返回 {}（功能静默关闭）。JSON 形态：{"术语": "——解释"}。
    """
    p = Path(chapter_dir)
    for cand in (p / "term_explanations.json", p.parent / "term_explanations.json"):
        try:
            if not cand.is_file():
                continue
            data = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if str(v).strip()}
    return {}


def inject_explanations(text: str, terms: dict) -> Tuple[str, list[str]]:
    """为文本中未内嵌解释的术语注入解释短语。

    terms: {"术语": "——解释短语"}（从作品目录加载，见 load_terms_for_work）。
    返回 (processed_text, injected_terms_list)。terms 为空时原文返回。
    """
    if not text or not terms:
        return text, []

    # 分离标题与正文
    first_newline = text.find("\n")
    if first_newline >= 0 and text.startswith("#"):
        header, body = text[: first_newline + 1], text[first_newline + 1 :]
    else:
        header, body = "", text

    if not body.strip():
        return text, []

    injected: list[str] = []
    # 从长到短排序，避免短术语先被替换影响长术语（如"错季"影响"错季相"）
    terms_sorted = sorted(terms.keys(), key=len, reverse=True)

    for term in terms_sorted:
        pattern = re.compile(re.escape(term))
        expl = terms[term]
        # 该术语是某些更长术语的前缀子串时，命中位置若正是长术语的起始，
        # 必须跳过——否则短术语会吃进长术语的字内部位造成二次注入
        super_terms = [t for t in terms_sorted
                       if len(t) > len(term) and term in t]
        for m in pattern.finditer(body):
            raw_start = m.start()
            if any(body.startswith(t, raw_start) for t in super_terms):
                continue
            window = body[max(0, raw_start - 25) : raw_start + len(term) + 25]
            if _is_real_explanation(body, raw_start):
                break  # 已有有效解释，跳过
            # 注入
            insert_pos = raw_start + len(term)
            tail = body[insert_pos : insert_pos + 30]
            punct_m = re.search(r"([，。、；：！？\s\n])", tail)
            if punct_m and punct_m.end() < 30:
                final_pos = insert_pos + punct_m.start()
            else:
                final_pos = insert_pos
            body = body[:final_pos] + expl + body[final_pos:]
            injected.append(term)
            break  # 每章每个术语只解释第一次

    return header + body, injected