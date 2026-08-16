"""休息间隙规则治理的单元测试（不碰真实记忆库）。

背景：2026-08-15 发现记忆库被 critique 误判污染（ChartAreas/AxisType/PostData 等
不存在的 API 被教进规则）且存在内容截断的残缺规则。本测试保护治理判定：
  - _is_truncated_rule：断在标点前/极短判残缺；代码模板与闭合引用结尾不误判；
  - _is_poison_rule：教使用狐表语境已确认不存在的 API 判毒；带否定/修正上下文不误判；
  - _dedupe_rules：只对 /改进 /坑 prose 去重，/API /代码 速查模板全部保留
    （2026-08-15 实测 16 条 API 模板曾被误删：SQLInsertFile vs SQLLoadFile 等）；
  - govern_memory：三类治理汇总。
"""
import time
import uuid

import pytest
import autonomous_coder as ac


def _mem(content, imp=0.85):
    return {"id": str(uuid.uuid4()), "kind": "skill", "mtype": "skill", "content": content,
            "importance": imp, "created_at": time.time()}


# ---------- _is_truncated_rule ----------

def test_truncated_ends_with_punctuation():
    assert ac._is_truncated_rule("[A/改进] 题目「X」：建议统一格式，")
    assert ac._is_truncated_rule("[A/改进] 题目「X」：设置请求头：")
    assert ac._is_truncated_rule("[A/改进] 题目「X」：同时增加 openId、")


def test_truncated_too_short():
    assert ac._is_truncated_rule("[A/坑] s.Add")
    assert ac._is_truncated_rule("[A/坑] ")


def test_code_template_not_truncated():
    # /代码 /API /范例 模板是 code[:300] 压缩的紧凑格式，以注释/括号结尾是正常形态
    assert not ac._is_truncated_rule("[A/代码] Dim dt As DataTable = DataTables(\"订单\")     '表")
    assert not ac._is_truncated_rule("[A/API] dr.SQLLoadFile(Field, FileName)")


def test_closed_backtick_not_truncated():
    # 闭合引用结尾（...使用 `CoreWebView2`）是完整规则，不能误判
    assert not ac._is_truncated_rule("[A/坑] 初始化后才能使用 `CoreWebView2`")


def test_complete_rule_not_truncated():
    assert not ac._is_truncated_rule("[A/改进] 题目「X」：用 TryCast 代替 CType 以安全转换。")


# ---------- _is_poison_rule ----------

def test_poison_teaches_fake_api():
    # 教使用狐表不存在的 API → 毒
    assert ac._is_poison_rule("[图表/改进] 题目「X」：建议使用 chart.ChartAreas(0).AxisY2")
    assert ac._is_poison_rule("[图表/坑] AxisType = AxisTypeEnum.Secondary 的标准 API")
    assert ac._is_poison_rule("[A/改进] 题目「X」：将按钮签名改为 Async Sub Button1_Click")
    assert ac._is_poison_rule("[A/改进] 题目「X」：改用 hc.PostData() 发送")


def test_poison_not_on_negation():
    # 带否定/修正上下文的是警告或好规则，不判毒
    assert not ac._is_poison_rule("[图表/坑] 狐表 Chart 无 AxisType/ChartAreas 属性")
    assert not ac._is_poison_rule("[A/坑] 不要使用 PostData（狐表无此方法）")
    assert not ac._is_poison_rule("[A/改进] 题目「X」：将 System.Net.Http.HttpClient 改为狐表 HttpClient")


# ---------- _dedupe_rules ----------

def test_dedupe_skips_api_templates():
    # API 短模板相似度高但不是重复（SQLInsertFile vs SQLLoadFile 功能不同）
    mems = [_mem("[二进制列/API] dr.SQLInsertFile(Field, FileName)"),
            _mem("[二进制列/API] dr.SQLLoadFile(Field, FileName)"),
            _mem("[异步编程/API] Functions.ExecuteAsync(\"函数名\", 参数)"),
            _mem("[异步编程/API] Functions.Execute(\"函数名\", 参数)")]
    assert ac._dedupe_rules(mems) == []


def test_dedupe_similar_prose_removes_lower_priority():
    mems = [_mem("[A/改进] 题目「X」：用 TryCast 代替 CType 安全转换。"),
            _mem("[A/改进] 题目「X」：用 TryCast 代替 CType 做安全转换。")]
    removed = ac._dedupe_rules(mems)
    assert len(removed) == 1
    keep_id, drop_id, r = removed[0]
    assert r >= 0.88


def test_dedupe_keeps_pitfall_over_improvement():
    pit = _mem("[A/坑] 先设 Length 再赋值 X/Y。")
    imp = _mem("[A/改进] 题目「X」：先设 Length 再赋值 X/Y。")
    removed = ac._dedupe_rules([pit, imp])
    assert len(removed) == 1
    keep_id = removed[0][0]
    assert keep_id == pit["id"]   # /坑 铁律保留


def test_dedupe_cross_domain_not_removed():
    # 跨领域同知识各有用处，不算重复
    mems = [_mem("[HttpClient/坑] HttpClient 不能复用，需 Clone()"),
            _mem("[大模型API/坑] HttpClient 不能复用，需 Clone()")]
    assert ac._dedupe_rules(mems) == []


# ---------- govern_memory ----------

def test_govern_memory_cleans_all_three():
    mems = [_mem("[A/改进] 题目「X」：设置请求头："),                      # 残缺
            _mem("[B/改进] 题目「X」：用 chart.ChartAreas(0) 设置"),       # 毒
            _mem("[C/改进] 题目「X」：用 TryCast 代替 CType 安全转换。"),   # 正常
            _mem("[C/改进] 题目「X」：用 TryCast 代替 CType 安全转换.。"),   # 与上条相似
            _mem("[D/API] dr.SQLInsertFile(Field, FileName)"),
            _mem("[D/API] dr.SQLLoadFile(Field, FileName)")]              # API 模板保留
    before = len(mems)
    rep = ac.govern_memory(mems)
    assert rep["truncated"] == 1 and rep["poison"] == 1 and rep["deduped"] == 1
    assert rep["total"] == 3
    ids = {m["id"] for m in mems}
    assert len(ids) == before - 3
    # API 模板全保留
    assert any("SQLLoadFile" in m["content"] for m in mems)
    assert any("SQLInsertFile" in m["content"] for m in mems)
