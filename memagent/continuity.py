"""小说连续性审校：事实台账（story bible）+ 写前节拍表 + 知识状态因果链核对。

为什么需要这个模块（2026-08-16，读者实测反馈驱动）：LLM 逐句局部生成，
没有持续查询的"世界状态"，导致人脑几乎不会犯的硬伤——死人报信、金额前后
矛盾、角色知道不该知道的事、连续两章情节同构。人脑靠三样东西避免这些：
情境模型（工作记忆里持续更新的世界状态）、惊讶信号（预测违背即警觉）、
读者与作者是两个脑子。本模块把这三样外化成机制：

- **FactLedger（事实台账）**：works/<书名>/facts.json 的结构化世界状态表
  ——人物（生死/位置/年龄/已知信息）、数字（金额数量，带别名关键词）、
  物品、时间线。写作时注入提示（唯一权威来源），审校时比对；
- **beat_sheet（写前节拍表）**：动笔前先回答"本场每个角色进场时知道什么、
  从哪知道"——检查前移到生成之前；
- **knowledge_state_review（知识状态审校）**：逐句问"角色说出的每条信息，
  是否存在因果路径让他此刻知道它"——把人脑自动的心智理论变成显式核对；
- **deterministic_checks**：数字/年龄/死人开口的确定性检查——不需要 LLM
  的部分不用 LLM，便宜、确定、不疲劳。

与既有 critique 门槛建制同构：审校不评文采，只对事实；问题条目可注入
write_chapter(constraints=...) 触发带修正的重写。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .compat import call_responder
from .io_utils import atomic_write_json

# 连续性问题判定：确定性硬伤必拦；LLM 审校问题数超过该阈值也拦
MAX_LLM_ISSUES = 0


def _cn_to_int(s: str) -> int | None:
    """汉字数字 → 整数（≤ 万级："两千"→2000、"十六"→16、"一百五十"→150）。"""
    digits = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = num = 0
    for ch in s:
        if ch in digits:
            num = digits[ch]
        elif ch in units:
            total += (num or 1) * units[ch]
            num = 0
        else:
            return None
    return total + num


def _parse_num(tok: str) -> int | None:
    """解析数字 token：阿拉伯（含千分位）或汉字。"""
    tok = tok.strip().replace(",", "").replace("，", "")
    if not tok:
        return None
    if tok.isdigit():
        return int(tok)
    return _cn_to_int(tok)


_NUM_TOKEN = r"([\d,，]+|[零一两二三四五六七八九十百千万]+)"


def _strip_dialogue(text: str) -> str:
    """去掉对白引号内的内容，只留叙述。

    数字/年龄矛盾检查只针对叙述：叙述里的数字是"作者陈述的事实"，
    对白里的数字可以是角色的错误认知或故意说错再被纠正（《斩契》第1章
    "三条命，抵了三千？"→ 邱万山当场纠正为一千五——查对白会误报）。
    """
    return re.sub(r"[“「\"][^”」\"]*[”」\"]", "", text)


def _robust_json(reply: str) -> dict | None:
    """从回复里鲁棒地抠出 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    if not reply:
        return None
    text = reply.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


class FactLedger:
    """结构化事实台账（story bible），持久化在作品目录 facts.json。

    schema：
      characters: {名字: {status, location, age, knowledge: [条目], updated_chapter}}
      numbers:    [{desc, value, unit, aliases: [关键词], chapter}]   # value 统一存数字
      items:      [条目字符串]
      timeline:   [{chapter, event}]
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"characters": {}, "numbers": [], "items": [], "timeline": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {"characters": {}, "numbers": [], "items": [], "timeline": []}
        for key in ("characters", "numbers", "items", "timeline"):
            data.setdefault(key, {} if key == "characters" else [])
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.data, backup=True)

    # ---------- 合并 ----------

    def merge_chapter(self, chapter_no: int, facts: dict) -> None:
        """把 extract_facts 的产物合并进台账（后写覆盖同名字段，知识取并集去重）。

        characters 接受两种形态：抽取器输出的 [{name, ...}] 列表，或
        {名字: {...}} 字典（手写台账形态）——统一归一到字典。
        """
        raw_chars = facts.get("characters") or {}
        if isinstance(raw_chars, list):
            raw_chars = {c.get("name", ""): c for c in raw_chars
                         if isinstance(c, dict) and c.get("name")}
        for name, info in raw_chars.items():
            if not isinstance(info, dict):
                continue
            cur = self.data["characters"].setdefault(
                name, {"status": "", "location": "", "age": None, "knowledge": []})
            for field in ("status", "location"):
                if info.get(field):
                    cur[field] = str(info[field])
            if info.get("age") is not None:
                try:
                    cur["age"] = int(info["age"])
                except (ValueError, TypeError):
                    pass
            for k in info.get("knowledge") or []:
                if isinstance(k, str) and k and k not in cur["knowledge"]:
                    cur["knowledge"].append(k)
            cur["updated_chapter"] = chapter_no
        seen: dict[str, int] = {n["desc"]: i for i, n in
                                enumerate(self.data["numbers"])
                                if isinstance(n, dict) and n.get("desc")}
        for num in facts.get("numbers") or []:
            if not isinstance(num, dict) or not num.get("desc"):
                continue
            # 数值解析：LLM 常直接给 JSON 数字（1500.0 按字符串去点会放大 10 倍），
            # 或给汉字数词——先吃原生数字，再回退到 _parse_num；解析失败整条跳过
            raw = num.get("value")
            if isinstance(raw, bool):
                continue
            if isinstance(raw, int):
                value: int | None = raw
            elif isinstance(raw, float) and raw.is_integer():
                value = int(raw)
            else:
                value = _parse_num(str(raw or "").strip())
            if value is None or value < 0:
                continue
            entry = {"desc": str(num["desc"]), "value": value,
                     "unit": str(num.get("unit") or ""),
                     # 别名净化：去掉数字并要求 ≥2 字——LLM 常给"三页"这种
                     # 自带数字的别名（序数词"第三页是一…"会被误判数值矛盾），
                     # 去数字后剩单字单位（页/年）又过泛（满篇误报）
                     "aliases": [a for a in (
                         re.sub(r"[零一二两三四五六七八九十百千万\d,，\s]+", "", str(x))
                         for x in (num.get("aliases") or [])) if len(a) >= 2][:4],
                     "chapter": chapter_no}
            if num["desc"] in seen:
                self.data["numbers"][seen[num["desc"]]] = entry
            else:
                seen[num["desc"]] = len(self.data["numbers"])
                self.data["numbers"].append(entry)
        for it in facts.get("items") or []:
            if isinstance(it, str) and it and it not in self.data["items"]:
                self.data["items"].append(it)
        for ev in facts.get("timeline") or []:
            if isinstance(ev, dict) and ev.get("event"):
                self.data["timeline"].append(
                    {"chapter": chapter_no, "event": str(ev["event"])})

    # ---------- 呈现 ----------

    def sheet(self, max_knowledge: int = 3) -> str:
        """台账压缩成提示词友好的文本块（写作注入 / 审校比对共用）。"""
        lines: list[str] = []
        chars = self.data.get("characters") or {}
        if chars:
            lines.append("【人物】")
            for name, c in chars.items():
                bits = [f"状态={c.get('status') or '？'}"]
                if c.get("location"):
                    bits.append(f"位置={c['location']}")
                if c.get("age"):
                    bits.append(f"年龄={c['age']}岁")
                line = f"- {name}（{'；'.join(bits)}）"
                knows = (c.get("knowledge") or [])[-max_knowledge:]
                if knows:
                    line += " 已知：" + "／".join(knows)
                lines.append(line)
        nums = self.data.get("numbers") or []
        if nums:
            lines.append("【关键数字（唯一权威来源）】")
            for n in nums:
                lines.append(f"- {n['desc']}：{n['value']}{n.get('unit', '')}（第{n['chapter']}章）")
        items = self.data.get("items") or []
        if items:
            lines.append("【物品】")
            lines.extend(f"- {it}" for it in items[-8:])
        tl = self.data.get("timeline") or []
        if tl:
            lines.append("【时间线（最近）】")
            lines.extend(f"- 第{e['chapter']}章：{e['event']}" for e in tl[-4:])
        return "\n".join(lines)


def extract_facts(responder, title: str, chapter_no: int, chapter_text: str,
                  timeout: float = 60.0) -> dict | None:
    """从一章正文抽取结构化事实（LLM，严格 JSON）。失败返回 None（调用方跳过）。"""
    if responder is None or not getattr(responder, "available", False):
        return None
    prompt = (
        f"你是小说的事实记录员。从《{title}》第{chapter_no}章正文抽取事实台账，"
        f"只记录正文明确写出的内容，不要推测。\n\n"
        f"—— 正文 ——\n{chapter_text[:6000]}\n\n"
        "输出严格 JSON（不要任何多余文字）：\n"
        '{"characters": [{"name": "名字", "status": "活/死/失踪等", '
        '"location": "章末所在地", "age": 年龄数字或null, '
        '"knowledge": ["本章他得知/确认的信息，注明从谁或何事得知"]}], '
        '"numbers": [{"desc": "这笔数指什么（如：邱万山欠雾外赌债）", "value": 数字, '
        '"unit": "灵石/年/张等", "aliases": ["正文里指代它的2-4字关键词"]}], '
        '"items": ["本章出现的关键物品（一句带状态）"], '
        '"timeline": [{"event": "本章发生的关键事件（一句）"}]}\n'
        "正文没有的类别给空数组；人物 knowledge 只写本章新得知或本章内被引用的旧信息。"
    )
    try:
        reply = call_responder(responder, prompt, memories=None, timeout=timeout,
                               max_tokens=2048,
                               persona_extras="你是数据抽取助手，只输出 JSON，不创作。")
    except Exception:
        return None
    data = _robust_json(reply or "")
    return data if data and any(data.get(k) for k in
                                ("characters", "numbers", "items", "timeline")) else None


def beat_sheet(responder, title: str, chapter_no: int, goal: str | None,
               fact_sheet: str, tail: str, timeout: float = 60.0) -> list[str] | None:
    """写前节拍表：动笔前先回答"每个角色进场时知道什么、从哪知道"。

    人脑作家管这叫 outline——检查前移到生成之前：如果"阿蘅怎么知道抵押的事"
    在节拍表阶段就被回答，就轮不到写完再抓。
    """
    if responder is None or not getattr(responder, "available", False):
        return None
    prompt = (
        f"你是小说的结构编辑。为《{title}》第{chapter_no}章排一张节拍表（3-6 条），"
        f"每条一行，格式：\n"
        f"· 拍子内容（本场谁出场；每人进场时已知道什么、从哪知道）→ 章末推进到哪\n\n"
        f"—— 本章剧情目标 ——\n{goal or '（无明确目标）'}\n\n"
        f"—— 事实台账（节拍不得与之矛盾）——\n{fact_sheet or '（台账为空）'}\n\n"
        f"—— 上一章结尾 ——\n{tail or '（第一章）'}\n\n"
        "只输出节流行（每行以 · 开头），不要解释。"
    )
    try:
        reply = call_responder(responder, prompt, memories=None, timeout=timeout,
                               max_tokens=1024,
                               persona_extras="你是结构编辑，只输出节拍行，不写正文。")
    except Exception:
        return None
    beats = [ln.strip().lstrip("·- ").strip()
             for ln in (reply or "").splitlines()
             if ln.strip().startswith(("·", "-", "•"))]
    return beats or None


def deterministic_checks(ledger: FactLedger, chapter_text: str) -> list[str]:
    """确定性连续性检查（纯代码，不调 LLM）：死人开口 / 年龄矛盾 / 数字矛盾。

    死人开口查全文（对白行动都算）；年龄与数字只查叙述（对白可含角色口误，
    见 _strip_dialogue）。
    """
    issues: list[str] = []
    narration = _strip_dialogue(chapter_text)
    # ⓪ 禁用名/污染词（旧作人名与术语——模型幻觉会原样漏进正文，
    # 《斩契》试写第1章与第4章两次实测出现"残蜕/沈昭"，词表兜底）
    for bad in ledger.data.get("banned_names") or []:
        if isinstance(bad, str) and bad and bad in chapter_text:
            issues.append(f"禁用名「{bad}」出现在正文（旧作词汇/污染词表命中）")
    for name, c in (ledger.data.get("characters") or {}).items():
        status = str(c.get("status") or "")
        # ① 死人开口/行动（死人报信类硬伤的直接形态）
        if "死" in status:
            if re.search(rf"{re.escape(name)}[^。！？\n]{{0,6}}(?:说|道|问|喊|笑|开口|告诉|递|点头|摇头)",
                         chapter_text):
                issues.append(f"「{name}」台账状态为{status}，但本章有其说话/行动——"
                              f"死人不报信（若非 flashback/诈死需先更新台账并明写）")
        # ② 年龄矛盾（角色名 20 字内出现 "N岁"，N 支持汉字数字：十六岁/十七岁）
        age = c.get("age")
        if age:
            for m in re.finditer(rf"{re.escape(name)}[^。！？\n]{{0,20}}?{_NUM_TOKEN}\s*岁",
                                 narration):
                val = _parse_num(m.group(1))
                if val is not None and val != age:
                    issues.append(f"「{name}」台账年龄 {age} 岁，本章写 {m.group(1)}岁")
                    break
    # ③ 数字矛盾（台账数字的别名关键词 12 字内跟了不同数字，同样支持汉字数字）
    for n in ledger.data.get("numbers") or []:
        for alias in n.get("aliases") or []:
            for m in re.finditer(rf"{re.escape(alias)}[^。！？\n]{{0,12}}?{_NUM_TOKEN}",
                                 narration):
                val = _parse_num(m.group(1))
                if val is not None and val != n["value"]:
                    issues.append(f"「{n['desc']}」台账为 {n['value']}{n.get('unit', '')}，"
                                  f"本章在“{alias}”附近写了 {m.group(1)}")
                    break
    return issues


def knowledge_state_review(responder, title: str, chapter_no: int,
                           chapter_text: str, fact_sheet: str,
                           timeout: float = 60.0) -> list[dict]:
    """知识状态因果链审校（LLM）：逐句问"角色说的每条信息，他从何得知"。

    这是人脑自动做的心智理论（theory of mind）的显式版本——生成时模型只顾
    措辞顺滑，不会自己问信息来源；这里换一个"审校员身份"专门只问这一件事。
    """
    if responder is None or not getattr(responder, "available", False):
        return []
    prompt = (
        f"你是连续性审校员，只核对事实因果，不评文采、不给写作建议。\n"
        f"审《{title}》第{chapter_no}章。只报告**明确的矛盾**，不要报告"
        f"**台账未记载**或**推导不够详细**的内容——台账没有的数字/背景不算问题，"
        f"只有与台账或本章前文**冲突**才算：\n"
        f"1. 信息因果矛盾：角色说出/做出的判断，其信息来源在时间上不可能——"
        f"死人报信；当场才发生的事被提前引用；A 只对 B 说过的秘密 C 却知道"
        f"（无转述路径）；角色知道本人不在场时发生的事（无目击/传闻交代）。\n"
        f"2. 数字/年龄矛盾：与台账**同一笔**数字写了不同值（角色对白里说错"
        f"后当场被纠正的不算）。\n"
        f"3. 时序矛盾：事件顺序与台账时间线直接冲突。\n"
        f"4. 章间同构：本章主场景与台账时间线里上一章同构"
        f"（同一对象再对峙、同一信息再揭示、同一揭示再演一遍）。\n\n"
        f"—— 事实台账 ——\n{fact_sheet or '（台账为空）'}\n\n"
        f"—— 待审章节 ——\n{chapter_text[:6000]}\n\n"
        f"发现问题输出严格 JSON："
        f'{{"issues": [{{"quote": "原文引文（≤30字）", "problem": "违反哪条", '
        f'"fix": "一句话改法"}}]}}；无问题输出 {{"issues": []}}。只输出 JSON。'
    )
    try:
        reply = call_responder(responder, prompt, memories=None, timeout=timeout,
                               max_tokens=2048,
                               persona_extras="你是连续性审校员，只核对事实因果，只输出 JSON。")
    except Exception:
        return []
    data = _robust_json(reply or "")
    issues = []
    for it in (data or {}).get("issues") or []:
        if isinstance(it, dict) and it.get("quote") and it.get("problem"):
            issues.append({"quote": str(it["quote"])[:60],
                           "problem": str(it["problem"]),
                           "fix": str(it.get("fix") or "")})
    return issues


# 通用量词白名单（数词后跟这些单位不算新造度量）；货币单位必须来自台账
# 术语构词核心（含这些字的 2-3 字子串是"体系名词"候选，新造需报）
_TERM_CORES = ("契", "境", "纹", "虫")
# 货币构词（首位是财物字、次位是货币量字的 2 字组合）；台账货币是唯一合法值
_CURRENCY_PAT = re.compile(r"[金银灵玉晶贝币](?:石|砂|两|币|子|票|锭|叶)")
# 噪声字符：虚词、数字、量词、常用动词与泛指字——候选子串里出现任何一个即弃
# （"的纹路/份命契/一卷契"是虚词量词粘连，"依契/契取/红纹/虫啃"是动宾/偏正切缝）
_NOISE_CHARS = set(
    "的了是在和与就也都把被上下里中不有个这那出入于地得着过之其而又"
    "一二三四五六七八九十百千万两半几零"
    "章节卷张份只把条个种座柄盏页行文块枚位名句年份"
    "依取归抵还收啃押画给卖买换做拿来去说看听进出起开着行走到回转送退"
    "求找想知记忘问笑哭喊叫之光红黑白青灰紫蓝色面身物事处地家"
    "人手眼心门路边方时大新高低"
    "些真守验道同档成书字旧空已未无有全整深浅密细粗厚薄从")
# 3 字候选的首尾领域字——"契档阁"（契+阁）是造词，"暗红纹/沈斩契/虫啃食"
# 这类跨词切缝首尾不是领域字，不报（中文无词边界，靠首尾约束保精度）
_DOMAIN_CHARS = set("契虫纹境司赤印档符阵器丹阁台楼坊殿观")


def term_guard(chapter_text: str, prev_texts: list[str],
               ledger: "FactLedger") -> list[str]:
    """专有名词守卫（确定性）：新造术语与伪币货币。

    《斩契》第 4 章实测漏网：模型自造「灵砂」（台账货币是灵石）、「痕契」
    「收契刀」（超出六名词封顶）。中文没有词边界，切词式匹配会大量误报
    （"的纹路/是契虫/命契印"），改用**子串计数 + 噪声过滤**：
    ① 货币：构词正则（灵X/金X/银X…）命中的 2 字货币词，不在台账单位表
       即报（全文含对白——对白里造货币同样是造词）；
    ② 术语：含核心字（契/境/纹/虫）的 2-3 字子串，滤掉含虚词/数字/量词/
       标点的碎片；台账 canon 词表（terms/items/timeline/人名）与前文里
       出现即既有，是已知术语的子串或超串也豁免；本章 ≥2 次且全新 → 报。
    """
    issues: list[str] = []
    canonical = {str(n.get("unit")) for n in ledger.data.get("numbers") or []
                 if n.get("unit")}
    # 全文扫描（含对白）：对白里造新货币/新术语同样是作者层的造词
    for tok in sorted(set(_CURRENCY_PAT.findall(chapter_text))):
        if tok not in canonical:
            issues.append(f"疑似新造货币「{tok}」（台账货币只有："
                          f"{', '.join(sorted(canonical)) or '无'}）")
    known = [str(t) for t in (ledger.data.get("terms") or [])]
    known += [str(t) for t in (ledger.data.get("items") or [])]
    known += [e.get("event", "") for e in (ledger.data.get("timeline") or [])]
    known += list((ledger.data.get("characters") or {}).keys())
    known += [t for t in prev_texts if t]
    core = "".join(_TERM_CORES)
    candidates: set[str] = set()
    for m in re.finditer(rf"[{core}]", chapter_text):
        i = m.start()
        for tok in (chapter_text[max(0, i - 1):i + 1],   # 2 字：核心在前（痕契）
                    chapter_text[i:i + 2],               # 2 字：核心在后（契虫）
                    chapter_text[max(0, i - 1):i + 2],   # 3 字窗口
                    chapter_text[max(0, i - 2):i + 1],
                    chapter_text[i:i + 3]):
            if 2 <= len(tok) <= 3:
                candidates.add(tok)
    flagged: list[str] = []
    for tok in sorted(candidates):
        # 噪声过滤：含虚词/数字/量词/常用动词的碎片不是术语
        if any(ch in _NOISE_CHARS or not ("一" <= ch <= "龥") for ch in tok):
            continue
        # 3 字候选必须首尾都是领域字（跨词切缝防护，见 _DOMAIN_CHARS 注释）
        if len(tok) == 3 and (tok[0] not in _DOMAIN_CHARS or tok[-1] not in _DOMAIN_CHARS):
            continue
        if chapter_text.count(tok) < 2:
            continue
        # 已知判定：canon 词表/前文里出现即既有；是已知术语的子串或超串也豁免
        # （"白契纸"包含已知的"契纸"、"契堂制"包含已知的"契堂"）
        if any(tok in ref for ref in known):
            continue
        if any(k in tok or tok in k for k in known if k):
            continue
        flagged.append(tok)
    # 包含去重：「契档」是「契档阁」的子串，报长的就够了
    issues += [f"新造术语「{tok}」（本章出现 {chapter_text.count(tok)} 次，"
               f"前文与台账名词表均无）——专有名词有封顶，新词需人设明确批准"
               for tok in flagged
               if not any(tok != other and tok in other for other in flagged)]
    return issues


def _shingles(text: str, n: int = 5) -> set[str]:
    """字符 n-gram 集合（去空白）——语句级相似度的确定性度量。"""
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def _last_sentences(text: str, k: int = 2) -> str:
    """章末钩子：最后 k 句（按 。！？ 切）。"""
    parts = [p.strip() for p in re.split(r"(?<=[。！？?!])", text) if p.strip()]
    return "".join(parts[-k:])


def _sentences_at(text: str, pos: int = -2) -> str:
    """倒数第 |pos| 句的单句文本（钩子区检测用）。"""
    parts = [p.strip() for p in re.split(r"(?<=[。！？?!])", text) if p.strip()]
    try:
        return parts[pos] if parts else ""
    except IndexError:
        return ""


def repetition_check(chapter_text: str,
                     prev_chapters: list[tuple[int, str]],
                     echo_threshold: float = 0.25,
                     hook_threshold: float = 0.6) -> list[str]:
    """章间重复检测（确定性，不调 LLM）。

    《斩契》旧版第 1/2 章实测事故：连续两章"对峙同一对象→入履约境→再揭示同一
    落款"——结构同构 + 钩子原句重复（第 4 章曾原样复用第 1 章结尾句
    "他要斩的契，落款是他自己的名字"）。两路信号：
    ① 语句回声：全章 5-gram shingle Jaccard ≥ echo_threshold；
    ② 钩子重复：**章末最后一句**与某前章最后一句的 shingle Jaccard ≥
      hook_threshold——钩子是章章都有的结构位，同一句钩子复读即报。
    结构同构（同对象再对峙、同信息再揭示）仍由知识状态审校的第 4 项覆盖。
    """
    issues: list[str] = []
    cur = _shingles(chapter_text)
    cur_hooks = [s for s in (_shingles(_last_sentences(chapter_text, k=1)),
                             _shingles(_sentences_at(chapter_text, -2))) if s]
    for no, prev in prev_chapters:
        prev_sh = _shingles(prev)
        if cur and prev_sh:
            jac = len(cur & prev_sh) / len(cur | prev_sh)
            if jac >= echo_threshold:
                issues.append(f"与第{no}章语句回声：5-gram 重合率 {jac:.0%}"
                              f"（≥{echo_threshold:.0%}）——大量复述既有内容")
        # 钩子区比对（最后两句任一句）：同一句钩子复读即报——《斩契》第6章
        # 实测把复读句藏在倒数第二位躲过单句检测
        prev_hooks = [s for s in (_shingles(_last_sentences(prev, k=1)),
                                  _shingles(_sentences_at(prev, -2))) if s]
        for ch_ in cur_hooks:
            for ph in prev_hooks:
                hj = len(ch_ & ph) / len(ch_ | ph)
                if hj >= hook_threshold:
                    issues.append(f"章末钩子区与第{no}章结尾句重复"
                                  f"（重合率 {hj:.0%}）——同一句钩子不能用两次")
    return issues


def review_chapter(responder, ledger: FactLedger, chapter_text: str,
                   chapter_no: int, title: str,
                   prev_chapters: list[tuple[int, str]] | None = None,
                   timeout: float = 60.0) -> dict:
    """编排一次完整连续性审校：确定性检查（含章间重复）+ 知识状态审校。

    prev_chapters: [(章号, 正文), ...] 供章间重复检测，通常传前 1-2 章。
    """
    det = deterministic_checks(ledger, chapter_text)
    det += repetition_check(chapter_text, prev_chapters or [])
    det += term_guard(chapter_text, [t for _, t in (prev_chapters or [])], ledger)
    llm = knowledge_state_review(responder, title, chapter_no, chapter_text,
                                 ledger.sheet(), timeout=timeout)
    ok = not det and len(llm) <= MAX_LLM_ISSUES
    return {"ok": ok, "det_issues": det, "llm_issues": llm}
