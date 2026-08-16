"""_parse_critique 的格式漂移回归测试。

背景：ch55 自评丢改进建议（LLM 未按「改进：」前缀）、ch56 丢五维分数
（LLM 用了 `**1. 文风一致：9.0**` 等漂移格式）——同类问题出现两次后
把解析器宽容化，这里固化回归保护。
"""
from __future__ import annotations

from memagent.critique import _DIMENSIONS, _parse_critique


def _scores(reply: str) -> dict:
    return _parse_critique(reply, 56, "t").scores


def test_standard_colon():
    s = _scores("文风一致：9.0\n节奏：8.5\n")
    assert s["文风一致"] == 9.0
    assert s["节奏"] == 8.5


def test_bold_numbered_prefix():
    """ch56 实况：`**1. 文风一致：9.0**`（去粗体后 `1. 文风一致：9.0`）。"""
    reply = "\n".join(
        f"**{i}. {name}：{score}**\n依据句" 
        for i, (name, _) in enumerate(_DIMENSIONS, 1)
        for score in [9.0]
    )
    s = _scores(reply)
    assert all(name in s for name, _ in _DIMENSIONS)
    assert s["文风一致"] == 9.0


def test_ascii_colon():
    s = _scores("文风一致: 9.0\n")
    assert s["文风一致"] == 9.0


def test_fullwidth_equals():
    s = _scores("节奏＝8.5\n")
    assert s["节奏"] == 8.5


def test_score_suffix_分():
    s = _scores("伏笔回收 9.0 分\n")
    assert s["伏笔回收"] == 9.0


def test_score_slash_denominator():
    s = _scores("人物弧光 8.5/10\n")
    assert s["人物弧光"] == 8.5


def test_space_separated():
    s = _scores("文风一致 9.0\n")
    assert s["文风一致"] == 9.0


def test_mixed_drift_full_reply():
    """ch56 raw 回复的典型形态：粗体编号行 + 依据 + 各节。"""
    reply = """## 五维逐条评分

**1. 文风一致：9.0**
开篇承接上一章叙事断层，整体语感克制。

**2. 节奏：8.5**
段落长短交替得当。

**3. 伏笔回收：9.0**
高效兑现多条旧线。

**4. 露骨场景分寸：10.0**
本章无亲密描写，合规满分。

**5. 人物弧光：8.5**
人物有明确的给出—承纳—离开—回头弧线。

**综合均分：9.0**

---

## 具体改进建议

**改进：** 下一章开头用三行以内给出硬性定量。

## 亮点

**亮点：** 开篇十三步霜线合拢。

## 综合评语
本章以 9.0 的高质量完成了开门章的职责。"""
    c = _parse_critique(reply, 56, "t")
    assert c.scores == {
        "文风一致": 9.0,
        "节奏": 8.5,
        "伏笔回收": 9.0,
        "露骨场景分寸": 10.0,
        "人物弧光": 8.5,
    }
    assert len(c.improvements) == 1
    assert "硬性定量" in c.improvements[0]
    assert len(c.strengths) == 1
    assert "霜线合拢" in c.strengths[0]
    assert c.overall


def test_missing_scores_tolerated():
    """分数段完全缺失时不抛错，仅空 dict。"""
    s = _scores("综合评语：还行。")
    assert s == {}
