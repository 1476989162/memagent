"""continuity.py 单元测试：事实台账 / 确定性检查 / 知识状态审校解析 / 节拍表。

全部离线（FakeResponder），不碰真实 LLM 与网络。
背景（2026-08-16）：读者在试写三章里抓出死人报信、金额矛盾、元叙述泄漏、
章间重复——人脑靠情境模型/惊讶信号/读者独立视角避免这些，本模块把它们
外化成机制，这里固化回归保护。
"""
import json

from memagent.continuity import (
    FactLedger,
    beat_sheet,
    deterministic_checks,
    extract_facts,
    knowledge_state_review,
    repetition_check,
    review_chapter,
    term_guard,
    _robust_json,
)


class FakeResponder:
    def __init__(self, reply: str):
        self.reply = reply
        self.queries: list[str] = []

    @property
    def available(self):
        return True

    def respond(self, query, memories=None, persona_extras=None, timeout=None,
                max_tokens=None):
        self.queries.append(query)
        return self.reply


LEDGER_MIN = {
    "characters": {
        "阿蘅": {"status": "活", "location": "青槐镇", "age": 16, "knowledge": ["哥哥死于契虫"]},
        "阿蘅之兄": {"status": "死（第1章矿难）", "location": "—", "age": None, "knowledge": []},
    },
    "numbers": [
        {"desc": "三条命契折价", "value": 1500, "unit": "灵石", "aliases": ["折", "抵"], "chapter": 1},
    ],
    "items": ["哑铜铃"],
    "timeline": [{"chapter": 1, "event": "矿工死于契虫"}],
}


def _ledger(tmp_path, data=LEDGER_MIN):
    p = tmp_path / "facts.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return FactLedger(p)


# ---------- FactLedger ----------

def test_ledger_roundtrip_and_sheet(tmp_path):
    led = _ledger(tmp_path)
    text = led.sheet()
    assert "阿蘅" in text and "1500" in text and "哑铜铃" in text
    led.save()
    again = FactLedger(tmp_path / "facts.json")
    assert again.data["characters"]["阿蘅"]["age"] == 16


def test_ledger_merge_upserts_and_dedups(tmp_path):
    led = _ledger(tmp_path)
    led.merge_chapter(4, {
        "characters": [{"name": "阿蘅", "status": "活", "location": "双契城",
                        "age": 16, "knowledge": ["哥哥死于契虫", "新消息：契堂收人契"]}],
        "numbers": [{"desc": "三条命契折价", "value": 1800, "unit": "灵石",
                     "aliases": ["折"], "chapter": 4}],
        "items": ["哑铜铃", "旧契底档"],
        "timeline": [{"event": "阿蘅赴双契城"}],
    })
    a = led.data["characters"]["阿蘅"]
    assert a["location"] == "双契城"
    assert len(a["knowledge"]) == 2          # 去重：旧条目不重复
    nums = [n for n in led.data["numbers"] if n["desc"] == "三条命契折价"]
    assert len(nums) == 1 and nums[0]["value"] == 1800   # 同 desc 覆盖
    assert led.data["timeline"][-1]["chapter"] == 4


# ---------- 确定性检查 ----------

def test_dead_character_speaking_flagged(tmp_path):
    led = _ledger(tmp_path)
    issues = deterministic_checks(led, "夜里，阿蘅之兄低声说：「别去雾外。」")
    assert any("阿蘅之兄" in i and "死" in i for i in issues)


def test_dead_character_alive_scene_not_flagged(tmp_path):
    led = _ledger(tmp_path)
    # 提及死者但不是他在说话/行动
    issues = deterministic_checks(led, "阿蘅想起哥哥生前说过的话，握紧了拳。")
    assert not any("阿蘅之兄" in i for i in issues)


def test_age_conflict_flagged(tmp_path):
    led = _ledger(tmp_path)
    issues = deterministic_checks(led, "阿蘅今年十七岁，个子又抽高了。")
    assert any("年龄" in i for i in issues)


def test_number_conflict_flagged(tmp_path):
    led = _ledger(tmp_path)
    issues = deterministic_checks(led, "那三条命契折了整整两千灵石。")
    assert any("折价" in i or "2000" in i for i in issues)


def test_dialogue_numbers_not_flagged(tmp_path):
    """对白里的错误数字（角色口误/故意说错后被纠正）不判矛盾——只查叙述。"""
    led = _ledger(tmp_path)
    issues = deterministic_checks(led, '"三条命，抵了三千？"\n'
                                       '邱万山摇头："只折一千五百。"')
    assert not any("折价" in i or "抵" in i for i in issues)


def test_narration_number_conflict_still_flagged(tmp_path):
    led = _ledger(tmp_path)
    issues = deterministic_checks(led, "他数了数，那笔抵债最终是两千灵石整。")
    assert any("折价" in i or "两千" in i for i in issues)


def test_consistent_text_clean(tmp_path):
    led = _ledger(tmp_path)
    assert deterministic_checks(led, "阿蘅十六岁，把那笔一千五百灵石的赔偿分给了三户人家。") == []


# ---------- extract / review 解析 ----------

def test_extract_facts_parses_fenced_json():
    r = FakeResponder("```json\n" + json.dumps({
        "characters": [{"name": "沈砚", "status": "活", "location": "雾外",
                        "age": None, "knowledge": ["铜铃响了"]}],
        "numbers": [{"desc": "待收契", "value": 37, "unit": "张", "aliases": ["待收契"]}],
        "items": ["哑铜铃"], "timeline": [{"event": "入雾"}],
    }, ensure_ascii=False) + "\n```")
    facts = extract_facts(r, "斩契", 3, "正文……")
    assert facts and facts["characters"][0]["name"] == "沈砚"
    assert "事实" in r.queries[0] or "记录员" in r.queries[0]


def test_extract_facts_garbage_returns_none():
    assert extract_facts(FakeResponder("我不会"), "斩契", 3, "正文") is None
    assert extract_facts(None, "斩契", 3, "正文") is None


def test_knowledge_review_parses_issues():
    payload = {"issues": [{"quote": "我哥他们说", "problem": "死人不能报信",
                           "fix": "改为逃出的工友所说"}]}
    r = FakeResponder(json.dumps(payload, ensure_ascii=False))
    issues = knowledge_state_review(r, "斩契", 1, "正文……", "台账……")
    assert len(issues) == 1 and issues[0]["problem"] == "死人不能报信"


def test_review_chapter_aggregation(tmp_path):
    led = _ledger(tmp_path)
    ok_reply = json.dumps({"issues": []}, ensure_ascii=False)
    cont = review_chapter(FakeResponder(ok_reply), led, "干净的正文。", 2, "斩契")
    assert cont["ok"] is True
    bad = json.dumps({"issues": [{"quote": "x", "problem": "y", "fix": ""}]},
                     ensure_ascii=False)
    cont2 = review_chapter(FakeResponder(bad), led, "干净的正文。", 2, "斩契")
    assert cont2["ok"] is False and len(cont2["llm_issues"]) == 1
    # 确定性硬伤即使 LLM 说没问题也拦
    cont3 = review_chapter(FakeResponder(ok_reply), led, "阿蘅之兄笑道：好。", 2, "斩契")
    assert cont3["ok"] is False and cont3["det_issues"]


# ---------- 节拍表 ----------

def test_beat_sheet_parses_bullets():
    r = FakeResponder("· 破庙开场（阿蘅已知哥哥死讯，来源：工友）→ 沈砚接案\n"
                      "· 入镇对质（邱万山知道债务数额）→ 履约境入口\n")
    beats = beat_sheet(r, "斩契", 1, "目标", "台账", "结尾")
    assert beats and len(beats) == 2 and "来源" in beats[0]


def test_beat_sheet_no_llm_returns_none():
    assert beat_sheet(None, "斩契", 1, None, "", "") is None


def test_robust_json_tolerance():
    assert _robust_json('废话 {"a": 1} 废话') == {"a": 1}
    assert _robust_json("```json\n{\"a\": 2}\n```") == {"a": 2}
    assert _robust_json("完全不是JSON") is None
    assert _robust_json("[1,2]") is None


# ---------- 章间重复检测 ----------

def test_repetition_echo_flagged():
    """全章大段复述前章内容 → 5-gram 回声告警（《斩契》旧版第1/2章同构事故形态）。"""
    base = "沈砚走进雾外，雾里的收契集市上挂满了各式命契，铜铃在袖中轻响。" * 15
    issues = repetition_check(base + "结尾略有不同。", [(3, base)])
    assert any("回声" in i for i in issues)


def test_repetition_hook_flagged():
    """章末最后一句与某前章结尾句重复 → 钩子重复告警（第4章原样复用第1章结尾句）。"""
    hook = "他要斩的契，落款是他自己的名字。"
    a = "第一章：沈砚在青槐镇接下第一桩案子，与老聋子对坐到天明。" * 25 + hook
    b = "第二章：阿蘅留在镇上，逐张核对矿工的命契，学会了三十味药。" * 25 + hook
    issues = repetition_check(b, [(1, a)])
    assert any("钩子" in i for i in issues)


def test_repetition_distinct_chapters_clean():
    a = "沈砚在雾外见到收契人的集市，摊位上挂满各式命契，他用三张空白契纸换了一盏雾灯。" * 12
    b = "阿蘅留在青槐镇，每日核对赔偿账目，跟着老聋子学抓药，认得了三十味药材。" * 12
    assert repetition_check(b, [(2, a)]) == []


def test_review_chapter_includes_repetition(tmp_path):
    """review_chapter 编排：章间重复进 det_issues 并拉低 ok。"""
    led = _ledger(tmp_path)
    base = "沈砚验过那张契，违约之痕淡得几乎看不见。" * 20
    ok_reply = json.dumps({"issues": []}, ensure_ascii=False)
    cont = review_chapter(FakeResponder(ok_reply), led, base + "新结尾。",
                          3, "斩契", prev_chapters=[(2, base)])
    assert cont["ok"] is False and any("回声" in i for i in cont["det_issues"])


# ---------- 禁用名 / 术语守卫 ----------

def test_banned_name_flagged(tmp_path):
    """旧作人名/术语漏进正文（"残蜕/沈昭"实测两次）→ 词表命中即报。"""
    data = dict(LEDGER_MIN)
    data["banned_names"] = ["沈昭", "残蜕", "裴枕灯", "塔纹"]
    led = _ledger(tmp_path, data)
    issues = deterministic_checks(led, "门口站着一个身影——残蜕在沈昭面前的虚像。")
    assert any("残蜕" in i for i in issues) and any("沈昭" in i for i in issues)


def test_term_guard_new_currency_and_term_flagged(tmp_path):
    """新造货币单位（灵砂）与新造术语（痕契）——第4章实测漏网形态。"""
    led = _ledger(tmp_path)
    prev = ["沈砚用命契换了一盏雾灯，契虫在灯罩上爬。他验过三张待收契。"]
    text = ("他先签了一张痕契，痕契上画着契虫的纹路。"
            "灰袍人说欠了三万两灵砂的账，还得押上命契。")
    issues = term_guard(text, prev, led)
    assert any("灵砂" in i for i in issues)
    assert any("痕契" in i for i in issues)


def test_term_guard_existing_terms_clean(tmp_path):
    led = _ledger(tmp_path)
    prev = ["他验过那张命契，违约之痕爬满了纸背，契虫啃食着墨迹。" * 3]
    text = "命契上的违约之痕又深了一层，契虫安静了下来，三十七张待收契码得整齐。"
    assert term_guard(text, prev, led) == []
