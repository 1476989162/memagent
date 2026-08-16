"""注入窗口自适应（新教训不再被截）+ 复犯自动升级 /坑 铁律 的单元测试。

背景：
  - 轮112：/坑 铁律排第 16 被 [:15] 截掉 → LLM 没看到铁律复现 1.2 分（已修）；
  - 轮123：JSON相关 20 条记忆里第 18 位的新 .DataTable 教训被窗口截掉，
    LLM 下次还是看不到 → 规则入库≠注入。本测试保证新教训能进窗口。
"""
import json
import time

import pytest
import autonomous_coder as ac


@pytest.fixture(autouse=True)
def _silence_log(monkeypatch):
    """persist_improvements 内部调用 ac.log 会写真实日志文件——测试必须静音。"""
    monkeypatch.setattr(ac, "log", lambda msg: None)


def _mem(content, imp=0.85, created=None, acc=None):
    m = {"id": "x", "kind": "skill", "mtype": "skill", "content": content,
         "importance": imp, "created_at": created or time.time()}
    if acc is not None:
        m["access_count"] = acc
    return m


# ---------- 提交前自检清单（铁律升级为行为约束）----------

def test_httpclient_checklist_in_prompt():
    mems = [_mem(f"[HttpClient/坑] 陷阱{i}") for i in range(4)]
    p = ac.build_prompt(mems, "HttpClient", "上传文件")
    assert "提交前自检清单" in p
    # HttpClient 专属 5 条全部注入
    for item in ac.DOMAIN_CHECKLISTS["HttpClient"]:
        assert item.replace("`", "") in p.replace("`", "")
    # 通用清单也注入
    for item in ac.GENERIC_CHECKLIST:
        assert item.replace("`", "") in p.replace("`", "")


def test_generic_checklist_only_for_other_domains():
    mems = [_mem("[Table/坑] 行过滤用 Select")]
    p = ac.build_prompt(mems, "Table", "下拉")
    assert "提交前自检清单" in p
    assert "SkipError" not in p                      # HttpClient 专属不进其他领域
    assert "不得中途截断" in p                        # 通用防截断条目全领域生效


def test_checklist_does_not_break_pitfall_injection():
    # 清单是追加段落，/坑 全量注入逻辑不受影响
    mems = [_mem(f"[HttpClient/坑] 陷阱{i}") for i in range(9)]
    p = ac.build_prompt(mems, "HttpClient", "上传")
    cov = ac.verify_injection(p, "HttpClient", mems)
    assert cov["missing"] == 0


def test_critique_prompt_has_foxtable_context():
    # 审查端必须带狐表语境速查，防止用标准 VB.NET/.NET 规则误判狐表自研语法
    # （轮130 把 New HttpClient(url) 判错、轮179 把 '''Async 判成需 Async Sub）
    p = ac.build_critique_prompt("HttpClient", "上传", "Dim hc As New HttpClient(url)", [])
    assert "狐表特殊语法速查" in p
    assert "'''Async" in p and "**不需要** `Async Sub`" in p
    assert "内置 Newtonsoft.Json" in p
    assert "非 IDisposable" in p


# ---------- select_injection_window ----------

def test_pitfalls_always_in_window():
    mems = [_mem(f"[A/坑] 铁律{i}") for i in range(8)] + [_mem(f"[A/改进] 建议{i}") for i in range(20)]
    win = ac.select_injection_window(mems, window=15)
    contents = [m["content"] for m in win]
    assert sum(1 for c in contents if "/坑" in c) == 8  # 全部铁律进窗口
    assert len(win) >= 15


def test_improves_newest_first():
    t0 = time.time()
    old = _mem("[A/改进] 旧教训：用 Select 而非遍历", created=t0 - 1000)
    new = _mem("[A/改进] 新教训：DataTables() 返回 FoxTable.DataTable", created=t0)
    win = ac.select_injection_window([old, new], window=15)
    assert win[0]["content"] == new["content"]  # 新近优先，不再被末尾截掉


def test_window_expands_with_memory_count():
    # 轮123 场景：JSON相关 20 条记忆，窗口应扩容到 30，新教训不再被截
    mems = [_mem(f"[JSON相关/改进] 建议{i}号") for i in range(19)]
    late = _mem("[JSON相关/改进] 将 DataTables(\"产品\") 改为 Dim dt = DataTables(\"产品\")",
                created=time.time())
    mems.append(late)  # 最后沉淀的教训（原实现排第 20，被 15 条窗口截掉）
    win = ac.select_injection_window(mems, window=15)
    assert len(win) >= 15
    assert any(m["content"] == late["content"] for m in win)  # 新教训进窗口


def test_pitfall_exceeds_window_still_all():
    mems = [_mem(f"[B/坑] 铁律{i}") for i in range(25)] + [_mem(f"[B/改进] 建议{i}") for i in range(10)]
    win = ac.select_injection_window(mems, window=15)
    assert sum(1 for m in win if "/坑" in m["content"]) == 25


def test_build_prompt_uses_window_and_keeps_pitfalls():
    mems = [_mem(f"[分级数据/坑] 铁律{i}") for i in range(5)]
    mems += [_mem(f"[分级数据/改进] 建议{i}") for i in range(30)]
    prompt = ac.build_prompt(mems, "分级数据", "某题")
    for i in range(5):
        assert f"[分级数据/坑] 铁律{i}" in prompt
    cov = ac.verify_injection(prompt, "分级数据", mems)
    assert cov["missing"] == 0


# ---------- 回放活性参与注入排序（断点研究修复）----------

def test_pitfalls_sorted_by_access_count():
    """/坑 内部按 access_count 排序：被反复回放的铁律排最前。"""
    low = _mem("[A/坑] 普通铁律", imp=1.2, acc=2)
    high = _mem("[A/坑] 反复验证铁律", imp=1.2, acc=15)
    win = ac.select_injection_window([low, high], window=15)
    assert win[0]["content"] == high["content"]  # 活性高的铁律在前


def test_improves_sorted_by_access_count_then_newest():
    """/改进 排序：importance 优先 → access_count 次之 → 新近兜底。"""
    t0 = time.time()
    old_active = _mem("[A/改进] 旧但被回放多次", imp=0.9, created=t0 - 100, acc=10)
    new_idle = _mem("[A/改进] 新但没被回放", imp=0.9, created=t0, acc=2)
    win = ac.select_injection_window([new_idle, old_active], window=15)
    assert win[0]["content"] == old_active["content"]  # 活性优先于新近


def test_build_prompt_marks_active_pitfalls():
    """被回放 ≥3 次的规则在 prompt 里带高优先级标注。"""
    mems = [_mem("[A/坑] 反复验证铁律", imp=1.2, acc=15),
            _mem("[A/坑] 普通铁律", imp=1.2, acc=2),
            _mem("[A/改进] 普通改进", imp=0.9, acc=2)]
    prompt = ac.build_prompt(mems, "A", "题目")
    assert "【高优先级·反复验证】[A/坑] 反复验证铁律" in prompt
    assert "【高优先级·反复验证】[A/坑] 普通铁律" not in prompt
    assert "【高优先级·反复验证】[A/改进] 普通改进" not in prompt
    # 注入覆盖仍全量
    cov = ac.verify_injection(prompt, "A", mems)
    assert cov["missing"] == 0


# ---------- persist_improvements 复犯升级 ----------

def test_repeated_lesson_upgraded_to_pitfall():
    mems = [_mem("[HttpClient/改进] 题目「上传」：改为无参构造：Dim hc As New HttpClient()")]
    ac.persist_improvements(mems, "HttpClient", "上传文件", "code",
                            ["改为无参构造：Dim hc As New HttpClient()，然后设置 Url"], {})
    upgraded = [m for m in mems if "/坑" in m["content"]]
    assert len(upgraded) == 1
    assert upgraded[0]["importance"] >= 1.2
    assert "题目「" not in upgraded[0]["content"]  # 铁律不带题目前缀


def test_existing_pitfall_skips_duplicate():
    mems = [_mem("[HttpClient/坑] 无参构造：Dim hc As New HttpClient() 再设 Url 属性",
                 imp=1.2)]
    n0 = len(mems)
    ac.persist_improvements(mems, "HttpClient", "上传", "code",
                            ["改为无参构造：Dim hc As New HttpClient()，然后设置 Url"], {})
    assert len(mems) == n0  # 已被铁律覆盖 → 跳过，不重复沉淀


def test_distinct_lesson_normal_append():
    mems = []
    ac.persist_improvements(mems, "Table", "排序", "code",
                            ["排序前检查金额列是否存在"], {})
    assert len(mems) == 1
    assert "/改进" in mems[0]["content"]


def test_core_lesson_strips_prefixes():
    assert ac._core_lesson("[A/改进] 题目「排序」：先查列") == "先查列"
    assert ac._core_lesson("[A/坑] 铁律") == "铁律"
