"""track_coder_trend 多源重建的单元测试（不碰真实日志/记忆库）。

背景：2026-08-15 重启事故把 foxtable_coder.log 清空（轮1-165 丢失）。
本测试保护重建逻辑：
  - parse_log_file：标准日志格式（主日志 / coder_stdout 同格式）
  - parse_recovery_doc：恢复文档逐轮分数表（含尾注行/无分数行）
  - parse_cycle_files：普通轮与验证轮文件头的领域/代码提取
  - parse_coverage_snapshot：领域均分快照
  - merge_cycles：多源优先级 + 验证轮分离
"""
import track_coder_trend as tc


def _mk(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(tc, "LOG", tmp_path / "foxtable_coder.log")
    monkeypatch.setattr(tc, "STDOUT_LOG", tmp_path / "coder_stdout.log")
    monkeypatch.setattr(tc, "RECOVERY_DOC", tmp_path / "recovery.md")
    monkeypatch.setattr(tc, "CYCLE_DIR", tmp_path / "cycles")
    monkeypatch.setattr(tc, "COVERAGE_DOC", tmp_path / "coverage.md")
    (tmp_path / "cycles").mkdir(exist_ok=True)


# ---------- parse_log_file ----------

def test_parse_log_file(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    log = _mk(tmp_path, "coder_stdout.log", """\
[2026-08-15 12:43:08] === 第 160 轮 ===
[2026-08-15 12:43:08] 抽题: 领域「分级数据」题目：实现BOM展开...
[2026-08-15 12:43:34] 生成代码: 1451 字符（未以收尾结构结束（需人工确认））→ cycle_160_分级数据.md
[2026-08-15 12:45:42] 自检验: 五维 {语法正确性=5.0, API 规范性=4.0, 铁律遵守=5.0, 实战可用性=3.0, 最佳实践=4.0}
[2026-08-15 12:46:00]   沉淀 3 条
""")
    rows = tc.parse_log_file(log)
    assert len(rows) == 1
    c = rows[0]
    assert c["n"] == 160 and c["domain"] == "分级数据" and c["code"] == 1451
    assert c["scores"]["语法正确性"] == 5.0 and c["scores"]["最佳实践"] == 4.0
    assert c["distilled"] == 3 and c["src"] == "coder_stdout.log"


def test_parse_log_file_fail(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    log = _mk(tmp_path, "coder_stdout.log", """\
[2026-08-15 13:09:40] === 第 162 轮 ===
[2026-08-15 13:09:40] 抽题: 领域「高级开发指南-客户端类」...
[2026-08-15 13:10:18] 生成代码: 500 字符（截断（代码块未闭合））→ cycle_162.md
[2026-08-15 13:10:33] 自检验异常: LLM 回复为空
""")
    c = tc.parse_log_file(log)[0]
    assert c["fail"] and "LLM" in c["fail"]


# ---------- parse_recovery_doc ----------

RECOVERY_SAMPLE = """\
# 日志恢复记录

## 已知分数汇总（从幸存数据恢复）

### 轮 145~165（对话历史 + coder_stdout.log）

| 轮 | 领域 | 五维 | 均分 |
|---|---|---|---|
| 148 | 用Excel报表生成网页 | （失败，无分数） | — |
| 150 | 事件编程 | 无分数 | — |
| 151 | 软件加密 | 0/0/0/0/0（219 字符截断） | 0.0 |
| 152 | OpenQQ | 7/9/8/4/6 | 6.8 |
| 153 | 窗口设计 | 10/10/10/9/8 | 9.4 |

### 专项名单受控验证

| 领域 | 修复前 | 修复后自然 | 验证轮 | 结论 |
|---|---|---|---|---|
| JSON相关 | 7.2 | 4.6 | **6.60** | ✅ 拉起 |
"""


def test_parse_recovery_doc(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk(tmp_path, "recovery.md", RECOVERY_SAMPLE)
    rows = {c["n"]: c for c in tc.parse_recovery_doc()}
    assert set(rows) == {148, 150, 151, 152, 153}
    # 正常五维行
    assert rows[152]["scores"] == {"语法正确性": 7.0, "API 规范性": 9.0, "铁律遵守": 8.0,
                                   "实战可用性": 4.0, "最佳实践": 6.0}
    assert rows[152]["fail"] == ""
    # 尾注行（0/0/0/0/0（219 字符截断））→ 五维全 0 解析成功
    assert rows[151]["scores"]["语法正确性"] == 0.0 and rows[151]["fail"] == ""
    # 无分数行 → scores 空 + fail 标记
    assert rows[148]["scores"] == {} and rows[148]["fail"] == "无分数"
    assert rows[150]["scores"] == {} and rows[150]["fail"] == "无分数"
    # 验证轮结论表不产生逐行条目
    assert not any("验证" in c["domain"] for c in tc.parse_recovery_doc())


# ---------- parse_cycle_files ----------

def test_parse_cycle_files(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk(tmp_path, "cycles/cycle_001_Table.md",
        "# 第1轮 · Table\n\n## 题目\n在Table中设置日期下拉\n\n## 生成代码\n\n```vbnet\nDim x As Integer = 1\n```\n")
    _mk(tmp_path, "cycles/cycle_155_JSON相关.md",
        "# JSON相关验证第1轮\n\n## 生成代码\n\n```vbnet\nDim a As String = \"x\"\n```\n")
    rows = {c["n"]: c for c in tc.parse_cycle_files()}
    assert rows[1]["domain"] == "Table" and rows[1]["code"] == 20
    assert rows[155]["domain"] == "JSON相关（验证）"
    assert rows[155]["src"] == "cycle_155_JSON相关.md"


# ---------- parse_coverage_snapshot ----------

def test_parse_coverage_snapshot(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk(tmp_path, "coverage.md", """\
# 领域覆盖
| 领域 | 练习 | 失败 | 打分轮 | 均分 | 沉淀规则 | 最近轮次 |
|---|---|---|---|---|---|---|
| 分级数据 | 11 | 1 | 7 | 5.2 | 37 | 专项26、112 |
| 生成图表 | 7 | 1 | 4 | 6.0 | 26 | 3、58 |
""")
    snap = tc.parse_coverage_snapshot()
    assert snap == {"分级数据": 5.2, "生成图表": 6.0}


# ---------- merge_cycles ----------

def test_merge_priority_and_verify_separation(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    # 主日志：轮 177 完整
    _mk(tmp_path, "foxtable_coder.log", """\
[2026-08-15 14:50:00] === 第 177 轮 ===
[2026-08-15 14:50:00] 抽题: 领域「本地WEB」...
[2026-08-15 14:50:30] 生成代码: 500 字符（截断（代码块未闭合））→ cycle_177_本地WEB.md
[2026-08-15 14:50:40] 自检验: 五维 {语法正确性=3.0, API 规范性=2.0, 铁律遵守=2.0, 实战可用性=1.0, 最佳实践=2.0}
""")
    # stdout：轮 160 有分数（主日志没有 → 应来自 stdout）
    _mk(tmp_path, "coder_stdout.log", """\
[2026-08-15 12:43:08] === 第 160 轮 ===
[2026-08-15 12:43:08] 抽题: 领域「分级数据」...
[2026-08-15 12:45:42] 自检验: 五维 {语法正确性=5.0, API 规范性=4.0, 铁律遵守=5.0, 实战可用性=3.0, 最佳实践=4.0}
""")
    # 恢复文档：轮 152 有分数
    _mk(tmp_path, "recovery.md", """\
### 轮 145~165

| 轮 | 领域 | 五维 | 均分 |
|---|---|---|---|
| 152 | OpenQQ | 7/9/8/4/6 | 6.8 |
| 160 | 分级数据 | 5/4/5/3/4 | 4.2 |
""")
    # cycle 文件：轮 1 无分数、验证轮 155
    _mk(tmp_path, "cycles/cycle_001_Table.md", "# 第1轮 · Table\n\n## 生成代码\n\n```vbnet\nDim x As Integer = 1\n```\n")
    _mk(tmp_path, "cycles/cycle_155_JSON相关.md", "# JSON相关验证第1轮\n\n## 生成代码\n\n```vbnet\nDim a = 1\n```\n")
    _mk(tmp_path, "cycles/cycle_160_分级数据.md", "# 第160轮 · 分级数据\n\n## 生成代码\n\n```vbnet\nDim b = 2\n```\n")

    cycles, verify = tc.merge_cycles()
    by_n = {c["n"]: c for c in cycles}
    # 主日志轮 177 保留（最高优先级来源 log）
    assert by_n[177]["src"] == "foxtable_coder.log"
    # 轮 160 只有 stdout 有分数 → 用 stdout（优先级高于恢复文档的同一轮）
    assert by_n[160]["src"] == "coder_stdout.log" and by_n[160]["scores"]["语法正确性"] == 5.0
    # 轮 152 来自恢复文档
    assert by_n[152]["src"] == "recovery.md" and by_n[152]["scores"]["API 规范性"] == 9.0
    # 轮 1 来自 cycle 文件（无分数）
    assert by_n[1]["src"] == "cycle_001_Table.md" and by_n[1]["scores"] == {}
    # 验证轮分离：155 不在主轮号序列
    assert 155 not in by_n
    v155 = [c for c in verify if c["n"] == 155]
    assert len(v155) == 1 and "验证" in v155[0]["domain"]
