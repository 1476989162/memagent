"""代码级遵守证据（recent_obeyed）的单元测试。

背景（2026-08-15）：遵守率此前只有五维「铁律遵守」代理分。本功能让审查端
对注入的每条规则给出「遵守/未遵守/未涉及」代码级实证，写回记忆的
recent_obeyed 计数器（遵守 +1 / 未遵守 +1），可统计真实遵守率。
本测试固化：
  - build_critique_prompt 给注入规则编号（铁律1/2/...）且 return_selected；
  - parse_compliance 解析审查回复的核对表；
  - _apply_compliance_evidence 写回记忆计数器；
  - 容错（未涉及跳过、格式缺失不崩）。
"""

import time

import autonomous_coder as ac


def _mem(content, imp=0.85, acc=None):
    m = {"id": "x", "kind": "skill", "mtype": "skill", "content": content,
         "importance": imp, "created_at": time.time()}
    if acc is not None:
        m["access_count"] = acc
    return m


def test_build_critique_prompt_numbers_rules_and_returns_selected():
    """审查 prompt 给注入规则编号，return_selected 返回规则列表。"""
    mems = [_mem("[A/坑] 铁律甲"), _mem("[A/改进] 教训乙"), _mem("[A/代码] 模板丙")]
    p, selected = ac.build_critique_prompt("A", "题目", "code", mems, return_selected=True)
    assert isinstance(p, str) and len(selected) == len(mems)
    # 编号出现在 prompt 里
    assert "铁律1:" in p and "铁律2:" in p and "铁律3:" in p
    # 核对表格式在 prompt 里
    assert "铁律遵守核对表" in p
    assert "遵守/未遵守/未涉及" in p
    # 默认行为（不传 return_selected）保持 str
    p2 = ac.build_critique_prompt("A", "题目", "code", mems)
    assert isinstance(p2, str)


def test_parse_compliance_extracts_statuses():
    """解析「铁律遵守核对表」→ 每条规则的遵守状态。"""
    selected = [_mem("[A/坑] 铁律甲"), _mem("[A/改进] 教训乙"), _mem("[A/代码] 模板丙")]
    reply = """综合评语：代码整体可用，但铁律2未遵守。
审查清单结果：
  - 字符型加引号：通过
铁律遵守核对表：
  - 铁律1: 遵守（代码里用了该写法）
  - 铁律2: 未遵守（违反了）
  - 铁律3: 未涉及（本轮没用到）
语法正确性：8/10"""
    out = ac.parse_compliance(reply, selected)
    assert len(out) == 3
    assert out[0]["status"] == "遵守" and out[0]["rule"] == 1
    assert out[1]["status"] == "未遵守"
    assert out[2]["status"] == "未涉及"
    assert out[0]["content"] == "[A/坑] 铁律甲"


def test_parse_compliance_tolerant_of_missing_section():
    """审查回复缺核对表段时不崩，返回空。"""
    reply = "综合评语：还行。\n语法正确性：7/10"
    assert ac.parse_compliance(reply, [_mem("[A/坑] 铁律甲")]) == []


def test_apply_compliance_evidence_counts():
    """遵守 +1 / 未遵守 +1 / 未涉及不计数。"""
    m1 = _mem("[A/坑] 铁律甲")
    m2 = _mem("[A/改进] 教训乙")
    mems = [m1, m2]
    compliance = [
        {"rule": 1, "content": "[A/坑] 铁律甲", "status": "遵守"},
        {"rule": 2, "content": "[A/改进] 教训乙", "status": "未遵守"},
        {"rule": 3, "content": "[A/代码] 模板丙", "status": "未涉及"},
    ]
    n = ac._apply_compliance_evidence(mems, compliance)
    assert n == 2  # 只更新了命中的 2 条
    assert m1["recent_obeyed"] == {"obeyed": 1, "violated": 0}
    assert m2["recent_obeyed"] == {"obeyed": 0, "violated": 1}
    # 再核对一次：遵守累积
    ac._apply_compliance_evidence(mems, [{"rule": 1, "content": "[A/坑] 铁律甲", "status": "遵守"}])
    assert m1["recent_obeyed"] == {"obeyed": 2, "violated": 0}


def test_apply_compliance_evidence_missing_content_skipped():
    """content 为空或未匹配到记忆时跳过，不崩。"""
    assert ac._apply_compliance_evidence([_mem("[A/坑] 铁律甲")],
                                         [{"rule": 1, "content": "", "status": "遵守"}]) == 0
    assert ac._apply_compliance_evidence([_mem("[A/坑] 铁律甲")],
                                         [{"rule": 1, "content": "[B/坑] 别家", "status": "遵守"}]) == 0
