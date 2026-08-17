"""文学审校（确定性）：句法节奏、对话密度、心理/感官密度、极简回应重复。

与 continuity.py 的连续性审校互补——连续性管"不错"，本模块管"好看"。

设计原则：
  ① 全模块不调 LLM，纯字符串匹配 + 计数（避免在重写循环里引入额外 API 开销）；
  ② 所有阈值都来自真实参考文本量化采样（陆月十九《从斩妖除魔开始长生不死》
     第3-15章，15,351字/519段），见 voice_profile.md 的原始数据；
  ③ 命中即报，不区分场景/对白——文学性问题是结构性的，全局指标。

典型命中的文本形态：
  - 第8章「对。」41次 → "极简回应词过量（41 次，上限 8）"
  - 第8章连续 60 行纯对话 → "连续 4 段以上纯对话段"
  - 第8章整章无感官词 → "全章无感官描写（每千字应 >= 1 次）"
  - 第8章整章无心理词 → "全章无心理/内心描写（每千字应 >= 1 次）"
"""

from __future__ import annotations

import re
from typing import Any


# ---------- 字符常量（避免在 r-string 里用 \u 转义导致的歧义）----------
# 各种引号形态
_LQ_L = chr(0x300C)  # 「
_LQ_R = chr(0x300D)  # 」
_CQ_L = chr(0x201C)  # "
_CQ_R = chr(0x201D)  # "
_DQ = chr(0x22)      # "
# 标点
_PERIOD = chr(0x3002)     # 。
_EXCL = chr(0xFF01)      # ！
_QUEST = chr(0xFF1F)     # ？
_ELLIP = chr(0x2026)     # …

# 段落缩进（中文全角空格）
_FULLWSP = chr(0x3000)


# ---------- 正则 ----------
_DIALOG_MARK = re.compile(_LQ_L + "|" + _LQ_R + "|" + _CQ_L + "|" + _CQ_R + "|" + _DQ)

_SIMPLE_REPLY_HEADS = re.compile(
    chr(0x5BF9) + "|"  # 对
    + chr(0x55EF) + "|"  # 嗯
    + chr(0x662F) + "|"  # 是
    + chr(0x597D) + "|"  # 好
    + chr(0x884C) + "|"  # 行
    + chr(0x5514) + "|"  # 唔
    + chr(0x54E6) + "|"  # 哦
    + chr(0x54C8) + "|"  # 哈
    + chr(0x5475)           # 呵
)
_SIMPLE_REPLY_TAIL = re.compile(
    _PERIOD + "|" + _EXCL + "|" + _QUEST + "|" + _ELLIP
    + "|" + _CQ_R + "|" + _LQ_R + "|" + _DQ + "|\\s"
)
_SENT_SPLIT = re.compile(_PERIOD + "|" + _EXCL + "|" + _QUEST + "|" + _ELLIP + "|\\n")


# ---------- 词表 ----------
_ACTION_WORDS = frozenset({
    "伸手", "抬手", "握紧", "松开", "收回", "转身", "抬眼", "垂眼", "眯眼",
    "皱眉", "挑眉", "咬牙", "点头", "摇头", "侧头", "低头", "抬头", "闭眼",
    "蹲下", "跪下", "扑倒", "踹", "砸", "砍", "捏", "按", "抓", "拽", "扔",
    "丢", "抽", "甩", "扫", "刺", "斩", "劈", "挡", "招架", "闪身", "翻身",
    "冲", "退", "拦", "闪避", "攥", "撑", "抵", "靠", "倚",
    "攥紧", "拍", "拍案", "摔", "扣", "抬脚", "迈步", "跨步",
})

_SENSORY_WORDS = frozenset({
    "风", "冷", "热", "凉", "暖", "烫", "灼", "刺骨", "血腥", "汗味", "铁锈",
    "泥土", "灰尘", "腥气", "味道", "气味", "烟味", "酒味", "腥甜", "冰冷",
    "滚烫", "刺疼", "钻心", "发颤", "发紧", "发沉", "发麻", "指尖", "指节",
    "喉结", "瞳孔", "后颈", "脊背", "太阳穴", "胸口", "呼吸", "心跳", "嗡嗡",
    "耳鸣", "鼻息", "耳膜", "冷汗", "燥热", "凉意", "寒意", "灼热", "阴冷",
})

_PSYCH_WORDS = frozenset({
    "忽然", "猛地", "下意识", "本能", "反应过来", "意识到", "觉得", "心中",
    "心头", "脑海", "闪过", "涌上", "冒出", "没忍住", "压不住", "咽", "喉头",
    "屏息", "喘息", "屏住", "吸了口气", "倒吸", "呼吸一滞", "胸口一滞",
    "胸口发闷", "心头一沉", "心头一跳", "心一沉", "心口", "心尖",
    "沉默", "怔", "愣", "发怔", "发呆", "呆住", "愕然", "一怔", "微怔",
    "心中一沉", "心中一凛", "心口一紧", "胸口一紧", "脊背一凉", "后背一凉",
})


# ---------- 段落切分 ----------

def _split_paragraphs(text: str) -> list[str]:
    """把章节正文切成段落。

    兼容两种主流格式：
      ① 空行分段落（Markdown 风格，.md 章节文件默认）
      ② 全角缩进分段落（原始 txt 风格，\u3000\u3000 开头）
    标题行（# 开头或"第N章"开头）不算段落。
    """
    if _FULLWSP in text:
        lines = text.split("\n")
        paras: list[str] = []
        cur: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                if cur:
                    paras.append("".join(cur))
                    cur = []
                continue
            # 标题行不入段
            if (s.startswith("#") or s.startswith("第")) and not s.startswith("第0章"):
                if cur:
                    paras.append("".join(cur))
                    cur = []
                continue
            if ln.startswith(_FULLWSP):
                if cur:
                    paras.append("".join(cur))
                cur = [s]
            elif cur:
                cur.append(s)
        if cur:
            paras.append("".join(cur))
        if paras:
            return paras

    paras = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        body_lines = [ln for ln in lines if not ln.startswith("#")]
        if body_lines:
            paras.append(" ".join(body_lines))
    return paras


# ---------- 计数辅助 ----------

def _count_word_hits(text: str, words: frozenset[str]) -> int:
    return sum(text.count(w) for w in words)


def _count_simple_replies(text: str) -> int:
    """计数极简回应词：形如"对。""嗯。""是。"…… 或"对"或"好"（前后标点/空白）。"""
    hits = 0
    i = 0
    n = len(text)
    while i < n:
        m = _SIMPLE_REPLY_HEADS.match(text, i)
        if not m:
            # 跳到下一个可能位置（跳过非汉字加速）
            # 简单起见，只进一位
            i += 1
            continue
        j = m.end()
        # 需要后跟标点或引号或空白或行尾
        if j < n and _SIMPLE_REPLY_TAIL.match(text, j):
            hits += 1
            i = j + 1
        else:
            i = j
    return hits


# ---------- 单段分类 ----------

def _is_pure_dialogue_para(p: str) -> bool:
    if len(p) > 20:
        return False
    if not _DIALOG_MARK.search(p):
        return False
    if any(w in p for w in _ACTION_WORDS):
        return False
    if any(w in p for w in _SENSORY_WORDS):
        return False
    if any(w in p for w in _PSYCH_WORDS):
        return False
    return True


# ---------- 阈值 ----------

_MAX_SIMPLE_REPLIES = 8
_MAX_CONSECUTIVE_PURE_DIALOG = 3
_MIN_PARAS_25_40_RATIO = 0.20
_MAX_PARAS_UNDER_15_RATIO = 0.30
_MIN_AVERAGE_SENT_LEN = 18
_MAX_DIALOG_RATIO = 0.60
_MIN_WORDS_25_40_PARAS = 15  # 硬下限：绝对数至少 15 段呼吸段（短句堆叠保护）
# 密度下限（每千字）：与 voice_profile 量化采样一致
_MIN_SENSORY_PER_1K = 2.0
_MIN_PSYCH_PER_1K = 1.5
_MIN_ACTION_PER_1K = 3.0


# ---------- 单条检查 ----------

def check_simple_replies(text: str) -> str | None:
    n = _count_simple_replies(text)
    if n > _MAX_SIMPLE_REPLIES:
        return f"极简回应词（对/嗯/是/好/行等）过量：{n} 次（上限 {_MAX_SIMPLE_REPLIES}）"
    return None


def check_consecutive_pure_dialogue(paras: list[str]) -> str | None:
    max_run = 0
    run = 0
    for p in paras:
        if _is_pure_dialogue_para(p):
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run > _MAX_CONSECUTIVE_PURE_DIALOG:
        return (f"连续 {_MAX_CONSECUTIVE_PURE_DIALOG} 段以上纯对话段"
                f"（检测到 {max_run} 段连续，上限 {_MAX_CONSECUTIVE_PURE_DIALOG}）")
    return None


def check_paragraph_length_mix(paras: list[str]) -> str | None:
    if not paras:
        return None
    n = len(paras)
    under_15 = sum(1 for p in paras if len(p) < 15)
    if under_15 / n > _MAX_PARAS_UNDER_15_RATIO:
        return (f"超短段（<15 字）占比过高：{under_15/n:.0%}"
                f"（{under_15}/{n}，上限 {_MAX_PARAS_UNDER_15_RATIO:.0%}）")
    mid_count = sum(1 for p in paras if 25 <= len(p) <= 40)
    mid_ratio = mid_count / n
    if mid_ratio < _MIN_PARAS_25_40_RATIO and mid_count < _MIN_WORDS_25_40_PARAS:
        return (f"呼吸段（25-40 字）占比过低：{mid_ratio:.0%}"
                f"（{mid_count}/{n}，下限 {_MIN_PARAS_25_40_RATIO:.0%} 且绝对数 ≥ {_MIN_WORDS_25_40_PARAS}）")
    return None


def check_average_sentence_length(text: str) -> str | None:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) >= 2]
    if not sents:
        return None
    avg = sum(len(s) for s in sents) / len(sents)
    if avg < _MIN_AVERAGE_SENT_LEN:
        return f"平均句长过短：{avg:.1f} 字（下限 {_MIN_AVERAGE_SENT_LEN} 字）"
    return None


def check_dialogue_ratio(paras: list[str]) -> str | None:
    if not paras:
        return None
    n = len(paras)
    dialog_paras = sum(1 for p in paras if _DIALOG_MARK.search(p))
    ratio = dialog_paras / n
    if ratio > _MAX_DIALOG_RATIO:
        return (f"对话占段比过高：{ratio:.0%}"
                f"（{dialog_paras}/{n}，上限 {_MAX_DIALOG_RATIO:.0%}）")
    return None


def check_no_sensory(text: str) -> str | None:
    hits = _count_word_hits(text, _SENSORY_WORDS)
    per_k = hits * 1000 / max(len(text), 1)
    if per_k < _MIN_SENSORY_PER_1K:
        return (f"感官描写密度过低：{per_k:.1f} 次/千字"
                f"（{_MIN_SENSORY_PER_1K:.0f} 下限；参考陆月十九约 3-6 次/千字）——"
                f"补足风/冷/烫/麻/指尖/瞳孔/呼吸/胸口/喉咙/汗/血/腥等感官细节")
    return None


def check_no_psych(text: str) -> str | None:
    hits = _count_word_hits(text, _PSYCH_WORDS)
    per_k = hits * 1000 / max(len(text), 1)
    if per_k < _MIN_PSYCH_PER_1K:
        return (f"心理/内心描写密度过低：{per_k:.1f} 次/千字"
                f"（{_MIN_PSYCH_PER_1K:.0f} 下限）——"
                f"补：忽然/猛地/心头/脑海中/咽/喉头/沉默/怔等反应段，"
                f"尤其关键事件后主角必须有 1 段内心")
    return None


def check_no_action(text: str) -> str | None:
    hits = _count_word_hits(text, _ACTION_WORDS)
    per_k = hits * 1000 / max(len(text), 1)
    if per_k < _MIN_ACTION_PER_1K:
        return (f"动作/神态引导词密度过低：{per_k:.1f} 次/千字"
                f"（{_MIN_ACTION_PER_1K:.0f} 下限）——"
                f"补：伸手/抬眼/皱眉/转身/咬牙/垂眼/点头/低头/眯眼等，"
                f"对话段之间必须穿插")
    return None


# ---------- 编排 ----------

def literary_checks(chapter_text: str) -> list[str]:
    """跑完所有文学审校检查，返回问题清单（空 = 通过）。"""
    issues: list[str] = []
    issues.append(check_simple_replies(chapter_text))
    paras = _split_paragraphs(chapter_text)
    issues.append(check_consecutive_pure_dialogue(paras))
    issues.append(check_paragraph_length_mix(paras))
    issues.append(check_average_sentence_length(chapter_text))
    issues.append(check_dialogue_ratio(paras))
    issues.append(check_no_sensory(chapter_text))
    issues.append(check_no_psych(chapter_text))
    issues.append(check_no_action(chapter_text))
    return [x for x in issues if x]


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    t = open(p, encoding="utf-8").read()
    print(f"段落数: {len(_split_paragraphs(t))}")
    print(f"字數: {len(t)}")
    for i in literary_checks(t):
        print(f"  X {i}")