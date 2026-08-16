"""截断检测 + 「禁止截断」告警 + 强制重练机制的单元测试（不碰 LLM / 真实记忆库）。

背景：轮7/轮112 分级数据都因代码中途截断得 1.2 分（`Dim filterLevel As` 断在半路）。
本测试保护：
  - truncation_heuristic 能识别收尾结构 / 语句中途截断，且不误报完整声明行；
  - has_antitruncation_rule 正确判断领域是否已沉淀「禁止截断」铁律；
  - _force_retrain / _take_forced_domain / pick_task 的强制重练链路；
  - extract_code 对未闭合代码块取全部截断代码（不是 reply[:500] 兜底）；
  - 同一领域连续截断达到上限后停止强制重练（防死循环烧配额）。
"""
import pytest

import autonomous_coder as ac
from memagent.memory import MemoryStore


@pytest.fixture(autouse=True)
def _reset_forced_state():
    """强制重练是进程内单例状态——每个测试前后清空，保证隔离。"""
    ac._reset_forced_retrain_state()
    yield
    ac._reset_forced_retrain_state()


# ---------- truncation_heuristic ----------

def test_complete_code_endings():
    for ending in ("End Sub", "End Function", "End If", "End While", "End Select",
                   "End Try", "End Class", "End With", "End Using", "Next", "Loop"):
        assert ac.truncation_heuristic(f"Dim x As Integer\n{ending}") == "完整"
    assert ac.truncation_heuristic("Dim x As Integer\nReturn") == "完整"
    assert ac.truncation_heuristic("tbl.DataSource = dt\n}") == "完整"
    # 轮160 误报形态：完整函数调用以 ) 结尾（MessageBox.Show(msg) 收尾），
    # 与轮155 事故的 `parentObj(`（以 ( 结尾）可区分。
    assert ac.truncation_heuristic("For Each kv In result\n    msg &= kv.Key & \" : \" & kv.Value & Environment.NewLine\nNext\nMessageBox.Show(msg)") == "完整"


def test_call_open_paren_truncated():
    # 轮155 事故形态：调用未完以 ( 结尾
    assert ac.truncation_heuristic("parentObj(") == "疑似截断"


def test_unclosed_string_truncated():
    assert ac.truncation_heuristic("Output.Show(\"abc") == "疑似截断"  # 奇数个引号


def test_truncated_mid_declaration():
    # 轮7/轮112 的真实截断形态：最后一行断在声明中途
    assert ac.truncation_heuristic("For Each r As DataRow In tbl\nDim filterLevel As") == "疑似截断"
    assert ac.truncation_heuristic("Dim sql As String\nDim cmd As") == "疑似截断"
    assert ac.truncation_heuristic("Dim x =") == "疑似截断"


def test_truncated_mid_property():
    # 轮112 的截断形态：`If tbl.G` 断在属性访问中途
    assert ac.truncation_heuristic("For Each r In tbl\nIf tbl.G") == "疑似截断"


def test_truncated_bare_end():
    assert ac.truncation_heuristic("If x > 0 Then\nEnd") == "疑似截断"


def test_empty_code():
    assert ac.truncation_heuristic("") == "空代码"
    assert ac.truncation_heuristic("   \n\n  ") == "空代码"


def test_complete_declaration_not_misflagged():
    # verify_fenji_fix 历史误报场景：完整声明行（As 后面还有内容）不能被判截断
    assert ac.truncation_heuristic("Dim rows As DataTable\nDim hideRows As New List(Of DataRow)") != "疑似截断"
    assert ac.truncation_heuristic("Dim rows As DataTable") != "疑似截断"


def test_trailing_comment_needs_manual_confirm():
    assert ac.truncation_heuristic("End Sub\n' 注意：xxx") == "未以收尾结构结束（需人工确认）"


# ---------- has_antitruncation_rule ----------

def test_has_antitruncation_rule():
    mems = [
        {"kind": "skill", "content": "[分级数据/坑] 输出禁止中途截断（历史两轮因截断只得 1.2）"},
        {"kind": "skill", "content": "[Table/坑] 行过滤用 Select 而非逐行遍历"},
        {"kind": "router", "content": "[分级数据] 路由说明"},
    ]
    assert ac.has_antitruncation_rule("分级数据", mems)
    assert not ac.has_antitruncation_rule("Table", mems)
    assert not ac.has_antitruncation_rule("不存在领域", mems)


# ---------- 强制重练链路 ----------

def test_force_retrain_consume_once():
    ac._force_retrain("分级数据")
    assert ac._take_forced_domain() == "分级数据"
    assert ac._take_forced_domain() is None  # 消费后清空，不会重复强制


def test_pick_task_force_domain_priority():
    d, t = ac.pick_task(force_domain="分级数据")
    assert d == "分级数据"
    assert t in ac.TASK_POOL["分级数据"]


def test_pick_task_normal_random():
    d, t = ac.pick_task()
    assert d in ac.TASK_POOL
    assert t in ac.TASK_POOL[d]


# ---------- extract_code：未闭合块取全部截断代码 ----------

def test_extract_code_closed_block_runs_heuristic():
    code, status = ac.extract_code("```vbnet\nDim x As Integer\nEnd Sub\n```")
    assert code == "Dim x As Integer\nEnd Sub"
    assert status == "完整"


def test_extract_code_unclosed_block_takes_full_partial():
    """块未闭合（LLM 输出被 max_tokens 截断）→ 取开栏后全部代码 + 未闭合状态。

    旧版取 reply[:500]：设计思路等正文混进代码、真代码只剩开头，
    自检验与沉淀都基于失真输入（660 轮里 279 轮截断均为此形态）。
    """
    reply = ("设计思路：先按层级过滤，再逐行输出。\n\n```vbnet\n"
             "Dim rows As DataTable\nFor Each r As DataRow In tbl\nIf tbl.G")
    code, status = ac.extract_code(reply)
    assert status == "截断（代码块未闭合）"
    assert "设计思路" not in code          # 正文不混入
    assert code.startswith("Dim rows")     # 从开栏处取起
    assert code.endswith("If tbl.G")       # 截断前的代码完整保留


def test_extract_code_no_fence_falls_back_to_head():
    """完全没有 ```vbnet 开栏（纯正文回复）→ 旧兜底：前 500 字符。"""
    reply = "抱歉，这个任务太难了。" * 100
    code, status = ac.extract_code(reply)
    assert status == "截断（代码块未闭合）"
    assert len(code) == 500
    assert code == reply[:500]


# ---------- 强制重练连续上限 ----------

def test_note_forced_truncation_streak_cap():
    """同一领域连续截断：上限内继续强制，达到上限后不再强制。"""
    assert ac._note_forced_truncation("分级数据") is True   # 第 1 次
    assert ac._note_forced_truncation("分级数据") is True   # 第 2 次
    assert ac._note_forced_truncation("分级数据") is False  # 第 3 次 → 停
    assert ac._note_forced_truncation("分级数据") is False  # 之后持续停
    # 换领域重新起算
    assert ac._note_forced_truncation("Table") is True


# ---------- one_cycle 端到端：截断 → 告警 → 强制下轮重练 ----------

class _DummyAgent:
    responder = None

    def __init__(self):
        self.store = MemoryStore()  # one_cycle ⑥ 会 sync 进 agent.store

    def sleep(self):
        return {}


def _truncated_reply() -> str:
    # vbnet 代码块没有收尾 ``` → one_cycle 只取 reply[:500]，判为截断
    return "设计思路：先按层级过滤。\n```vbnet\nDim rows As DataTable\nFor Each r As DataRow In tbl\nIf tbl.G"


def _run_one_cycle(monkeypatch, mems, reply):
    calls: list[str] = []
    monkeypatch.setattr(ac, "load_ft_memory", lambda: mems)
    monkeypatch.setattr(ac, "save_ft_memory", lambda m: None)
    monkeypatch.setattr(ac, "atomic_write_text", lambda *a, **k: None)
    monkeypatch.setattr(ac, "respond_with_retry", lambda resp, prompt, **k: reply)
    monkeypatch.setattr(ac, "parse_scores", lambda r: {"scores": {}, "improvements": [], "overall": ""})
    monkeypatch.setattr(ac, "log", lambda msg: calls.append(msg))
    ok = ac.one_cycle(_DummyAgent(), 999, do_critique=True)
    return ok, calls


def test_one_cycle_truncation_with_rule_forces_retrain(monkeypatch):
    # 用强制重练把领域固定为「分级数据」——这也是真实链路：上轮告警 → 本轮重练仍截断 → 再次告警
    ac._force_retrain("分级数据")
    mems = [{"kind": "skill", "content": "[分级数据/坑] 输出禁止中途截断（历史两轮因截断只得 1.2）"}]
    ok, calls = _run_one_cycle(monkeypatch, mems, _truncated_reply())
    assert ok is True
    assert any("强制重练·上轮截断告警" in c for c in calls)  # 本轮确为强制重练轮
    # 高优先级告警已记录
    assert any("⚠ 高优先级截断告警" in c and "分级数据" in c for c in calls)
    # 强制重练领域已设好，下轮 one_cycle 会再次消费它（截断不修复就持续重练）
    assert ac._take_forced_domain() == "分级数据"
    assert ac._take_forced_domain() is None


def test_one_cycle_truncation_without_rule_no_alert(monkeypatch):
    ac._force_retrain("Table")
    mems = [{"kind": "skill", "content": "[Table/坑] 行过滤用 Select 而非逐行遍历"}]
    ok, calls = _run_one_cycle(monkeypatch, mems, _truncated_reply())
    assert ok is True
    assert not any("高优先级截断告警" in c for c in calls)
    assert ac._take_forced_domain() is None


def test_one_cycle_complete_code_no_alert(monkeypatch):
    ac._force_retrain("分级数据")
    mems = [{"kind": "skill", "content": "[分级数据/坑] 输出禁止中途截断"}]
    reply = "```vbnet\nDim rows As DataTable\nFor Each r In tbl\nNext\nEnd Sub\n```"
    ok, calls = _run_one_cycle(monkeypatch, mems, reply)
    assert ok is True
    assert not any("高优先级截断告警" in c for c in calls)
    assert ac._take_forced_domain() is None


def test_one_cycle_forced_domain_consumed_in_pick(monkeypatch):
    """上轮告警 → 本轮抽题日志带「强制重练·上轮截断告警」标记。"""
    ac._force_retrain("分级数据")
    mems = [{"kind": "skill", "content": "[分级数据/坑] 输出禁止中途截断"}]
    reply = "```vbnet\nEnd Sub\n```"
    ok, calls = _run_one_cycle(monkeypatch, mems, reply)
    assert ok is True
    assert any("强制重练·上轮截断告警" in c and "分级数据" in c for c in calls)
    assert ac._take_forced_domain() is None


def test_one_cycle_unclosed_block_feeds_real_code_to_critique(monkeypatch):
    """未闭合块 → 自检验 prompt 收到开栏后的真实截断代码（非正文前 500 字符）。"""
    captured: dict = {}

    def fake_crit_prompt(domain, task, code, mems, **kw):
        captured["code"] = code
        return "critique prompt", []

    monkeypatch.setattr(ac, "build_critique_prompt", fake_crit_prompt)
    monkeypatch.setattr(ac, "parse_compliance", lambda reply, sel: [])
    ac._force_retrain("分级数据")
    mems = [{"kind": "skill", "content": "[分级数据/坑] 输出禁止中途截断"}]
    ok, _ = _run_one_cycle(monkeypatch, mems, _truncated_reply())
    assert ok is True
    assert captured["code"].startswith("Dim rows")     # 真代码，不是"设计思路"
    assert captured["code"].endswith("If tbl.G")


def test_one_cycle_streak_cap_stops_forcing(monkeypatch):
    """同一领域连续 3 轮截断 → 第 3 轮起不再设置强制重练并提示暂停。"""
    monkeypatch.setattr(ac, "pick_task", lambda **kw: ("分级数据", "按层级过滤输出"))
    mems = [{"kind": "skill", "content": "[分级数据/坑] 输出禁止中途截断"}]
    last_calls: list[str] = []
    for _ in range(ac.FORCED_RETRAIN_MAX_STREAK):
        ok, calls = _run_one_cycle(monkeypatch, mems, _truncated_reply())
        assert ok is True
        last_calls = calls
    assert any("暂停强制重练" in c for c in last_calls)   # 第 3 轮提示暂停
    assert ac._take_forced_domain() is None                # 不再强制
