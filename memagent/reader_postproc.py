# -*- coding: utf-8 -*-
"""读者友好度 post-processor：自动为未内嵌解释的专有术语注入解释短语。

策略：不再依赖 LLM 自觉遵守写作规则——在 LLM 生成文本后，程序化扫描术语
首次出现位置，若前后 25 字内无【有效解释性标志词】，则插入「——解释短语」。
每章每个术语最多解释 1 次。

关键修正（v2）：
- 破折号「——」在中文小说里大量用于叙述断句，不是解释标志
- 仅当「——」后接定义/描述性词汇（是/叫/为/本/指/意/意/实/意/种/为/原/是/即）时，才视为解释
"""
import re
from typing import List, Tuple

TERM_EXPLANATIONS: dict[str, str] = {
    "塔纹": "——他掌心自幼就有的九道银色裂痕",
    "淬生气": "——从他骨髓里抽出来的暖流灵气",
    "锈脉": "——蚀灭极霜气在他血脉里凿出的暗管",
    "错季": "——每隔一段日子就出现的时间错位现象",
    "骨契": "——以骨为契、以血为印的古老契约",
    "残蜕": "——从沈昭灵魂中蜕出的旧壳，他的另一半自我",
    "造血深渊": "——沧澜旧都地宫最深处孕育诡异生灵的暗渊",
    "当票": "——浮屠商会典当之时开给当主的凭证",
    "脊骨": "——沈昭在断魂崖试炼地获得的远古遗骨",
    "骨片": "——记载契约条款的薄如蝉翼的骨制刻片",
    "当主": "——在浮屠商会典当物品的人",
    "命格": "——决定一个人命运走向的先天印记",
    "错季相": "——同时掌控淬生与蚀灭两种灵气相位的至高境界",
    "锁扣": "——浮屠商会暗柜上用于封印契约的铁制机关",
    "错季裂隙": "——错季发生时空间与时间之间的裂缝",
    "未时": "——错季纪年中对应下午一点至三点的时辰",
    "未月": "——错季纪年中月亮位置特殊的月份",
    "九川": "——传说中被灵气两极排序统治的九州疆土",
    "折丹": "——逆转金丹修为、以退为进的禁忌修炼法",
    "封镇": "——用灵气和契约封印妖神的古老阵法",
    "骨画": "——骨片上以朱砂绘制的契约图案",
    "水漏": "——残蜕逼近时从脊骨残片中传来的九声骨落之声",
    "残响": "——残蜕读取沈昭情绪峰值时留下的共鸣残响",
    "霜意": "——蚀灭极灵气的外在表现，冰冷刺骨",
    "空壳": "——霜意填满后失去自我、只剩躯体的存在",
    "三魂": "——人的灵魂三态：魂、魄、识",
    "影": "——残蜕在沈昭面前呈现的虚像",
}

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


def inject_explanations(text: str) -> Tuple[str, list[str]]:
    """为文本中未内嵌解释的术语注入解释短语。

    返回 (processed_text, injected_terms_list)。
    """
    if not text:
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
    # 从长到短排序，避免"错季"先被替换影响"错季相"
    terms_sorted = sorted(TERM_EXPLANATIONS.keys(), key=len, reverse=True)

    for term in terms_sorted:
        pattern = re.compile(re.escape(term))
        expl = TERM_EXPLANATIONS[term]
        for m in pattern.finditer(body):
            raw_start = m.start()
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