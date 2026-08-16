# -*- coding: utf-8 -*-
"""读者友好度 post-processor：自动为未内嵌解释的专有术语注入解释短语。

策略：不再依赖 LLM 自觉遵守写作规则——在 LLM 生成文本后，程序化扫描术语
首次出现位置，若前后 25 字内无解释性标志词，则插入「——解释短语」。
每章每个术语最多解释 1 次。
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

EXPLAINERS: List[str] = [
    "——", "（", "叫", "名为", "是指", "就是", "即", "原本叫", "实为",
    "本是", "原为", "原来叫", "即是", "意为", "一种", "原本", "意为",
]


def inject_explanations(text: str) -> Tuple[str, list[str]]:
    """为文本中未内嵌解释的术语注入解释短语。

    返回 (processed_text, injected_terms_list)。
    注：必须从前往后处理（避免插入偏移后续位置）；
        每章调用时传入该章正文即可获得"每章最多解释 1 次"的语义。
    """
    if not text:
        return text, []

    injected: list[str] = []
    # 从长到短排序，避免"错季"先被替换影响"错季相"
    terms_sorted = sorted(TERM_EXPLANATIONS.keys(), key=len, reverse=True)

    # 记录插入偏移（每插入一次，后续索引都要往后挪）
    offset = 0

    for term in terms_sorted:
        # 用 re 在偏移后的文本中查找
        pattern = re.compile(re.escape(term))
        for m in pattern.finditer(text):
            raw_start = m.start()
            actual_start = raw_start + offset
            window = text[
                max(0, actual_start - 25) : actual_start + len(term) + 25
            ]
            if any(e in window for e in EXPLAINERS):
                break  # 已有解释，跳过
            # 找第一次未解释的出现
            expl = TERM_EXPLANATIONS[term]
            insert_pos = actual_start + len(term)
            # 尝试在下一个标点前插入，保持中文格式自然
            tail = text[insert_pos : insert_pos + 30]
            punct_m = re.search(r"([，。、；：！？\）\s\n])", tail)
            if punct_m and punct_m.end() < 30:
                final_pos = insert_pos + punct_m.start()
            else:
                final_pos = insert_pos
            text = text[:final_pos] + expl + text[final_pos:]
            offset += len(expl)
            injected.append(term)
            break  # 每章每个术语只解释第一次

    return text, injected