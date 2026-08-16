"""日志完整性自检与重建的单元测试（不碰真实日志/记忆库）。

背景：2026-08-15 重启事故用 `>` 重定向清空 foxtable_coder.log（轮1-165 丢失）。
本测试保护：
  - detect_log_truncation：对比 cycle 文件普通轮与日志轮头，识别日志被清空；
  - rebuild_log_from_sources：从 cycle 文件/coder_stdout/恢复文档重建缺失轮行，
    不伪造分数、保留当前日志已有轮、幂等可重跑。
"""
import os
import time

import pytest
import autonomous_coder as ac


@pytest.fixture(autouse=True)
def _silence_log(monkeypatch):
    monkeypatch.setattr(ac, "log", lambda msg: None)


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "LOG_PATH", tmp_path / "foxtable_coder.log")
    monkeypatch.setattr(ac, "WORK_DIR", tmp_path / "cycles")
    (tmp_path / "cycles").mkdir(exist_ok=True)


def _mk_cycle(tmp_path, n: int, domain: str, code: str = "Dim x As Integer = 1\nEnd Sub\n"):
    p = tmp_path / "cycles" / f"cycle_{n:03d}_{domain}.md"
    p.write_text(f"# 第{n}轮 · {domain}\n\n## 题目\n{domain}的测试题{n}\n\n## 生成代码\n\n```vbnet\n{code}```\n",
                 encoding="utf-8")
    t = time.time() - (1000 - n)
    os.utime(p, (t, t))
    return p


def _mk_log(tmp_path, text: str):
    p = tmp_path / "foxtable_coder.log"
    p.write_text(text, encoding="utf-8")
    return p


def _mk_cycles(tmp_path, nums, domain="Table"):
    for n in nums:
        _mk_cycle(tmp_path, n, domain)


# ---------- detect_log_truncation ----------

def test_detect_clean_when_log_complete(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycles(tmp_path, range(1, 9))
    _mk_log(tmp_path, "".join(f"[2026-08-15 10:0{n}:00] === 第 {n} 轮 ===\n" for n in range(1, 9)))
    det = ac.detect_log_truncation()
    assert det["ok"] is True and det["missing"] == []


def test_detect_clean_when_few_missing(monkeypatch, tmp_path):
    # 只缺 1 轮（写入失败场景）不误报
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycles(tmp_path, range(1, 9))
    _mk_log(tmp_path, "".join(f"[2026-08-15 10:0{n}:00] === 第 {n} 轮 ===\n" for n in range(1, 8)))
    det = ac.detect_log_truncation()
    assert det["ok"] is True


def test_detect_truncated_when_log_cleared(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycles(tmp_path, range(1, 16))
    # 日志只剩 2 轮（被清空后重新积累的少量轮）
    _mk_log(tmp_path, "[2026-08-15 14:00:00] === 第 15 轮 ===\n[2026-08-15 14:10:00] === 第 16 轮 ===\n")
    det = ac.detect_log_truncation()
    assert det["ok"] is False
    assert 1 in det["missing"] and 14 in det["missing"]
    assert det["reason"] == "日志疑似被清空"


def test_detect_skips_when_no_cycles(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    det = ac.detect_log_truncation()
    assert det["ok"] is True and det["reason"].startswith("无 cycle")


def test_detect_ignores_verify_cycle_files(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycle(tmp_path, 1, "Table")
    vp = tmp_path / "cycles" / "cycle_155_JSON相关.md"
    vp.write_text("# JSON相关验证第1轮\n\n## 生成代码\n\n```vbnet\nDim a = 1\n```\n", encoding="utf-8")
    det = ac.detect_log_truncation()
    assert det["cycle_count"] == 1  # 只统计普通轮，验证轮不计


# ---------- rebuild_log_from_sources ----------

def test_rebuild_restores_missing_rounds(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycles(tmp_path, range(1, 9))
    # 日志只有轮 8（缺失 1-7）
    _mk_log(tmp_path, "[2026-08-15 14:00:00] === 第 8 轮 ===\n[2026-08-15 14:00:00] 抽题: 领域「Table」题目：已有\n")
    n_rebuilt = ac.rebuild_log_from_sources()
    assert n_rebuilt == 7
    txt = tmp_path.joinpath("foxtable_coder.log").read_text(encoding="utf-8")
    assert "=== 第 1 轮 ===" in txt and "=== 第 7 轮 ===" in txt
    assert "抽题: 领域「Table」题目：Table的测试题1...（重建）" in txt
    assert "（完整）→ cycle_001_Table.md（重建）" in txt
    assert txt.count("=== 第 8 轮 ===") == 1   # 已有轮不重复
    assert (tmp_path / "foxtable_coder.log.bak-pre-rebuild").exists()


def test_rebuild_idempotent(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycles(tmp_path, range(1, 9))
    _mk_log(tmp_path, "[2026-08-15 14:00:00] === 第 1 轮 ===\n")
    assert ac.rebuild_log_from_sources() == 7
    assert ac.rebuild_log_from_sources() == 0   # 重建后完整 → 幂等
    assert ac.detect_log_truncation()["ok"] is True


def test_rebuild_does_not_fake_scores(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycles(tmp_path, range(1, 9))
    _mk_log(tmp_path, "")
    assert ac.rebuild_log_from_sources() == 8
    txt = tmp_path.joinpath("foxtable_coder.log").read_text(encoding="utf-8")
    assert "自检验" not in txt   # 无分数来源 → 不伪造


def test_rebuild_adds_recovery_scores(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    # 用真实恢复文档里有分数的轮（152 OpenQQ 7/9/8/4/6）作样本
    _mk_cycles(tmp_path, [145, 146, 147, 148, 149, 150, 151, 152])
    _mk_log(tmp_path, "")
    assert ac.rebuild_log_from_sources() == 8
    txt = tmp_path.joinpath("foxtable_coder.log").read_text(encoding="utf-8")
    assert "自检验: 五维 {语法正确性=7.0, API 规范性=9.0, 铁律遵守=8.0, 实战可用性=4.0, 最佳实践=6.0}（重建）" in txt


def test_rebuild_keeps_current_log_rounds_after_rebuilt(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _mk_cycles(tmp_path, range(1, 9))
    _mk_log(tmp_path, "[2026-08-15 15:00:00] FoxTable 启动\n"
                      "[2026-08-15 15:00:01] === 第 186 轮 ===\n"
                      "[2026-08-15 15:00:05] 抽题: 领域「Table」题目：当前\n")
    assert ac.rebuild_log_from_sources() == 8
    txt = tmp_path.joinpath("foxtable_coder.log").read_text(encoding="utf-8")
    assert txt.index("FoxTable 启动") < txt.index("=== 第 1 轮 ===") < txt.index("=== 第 186 轮 ===")
    assert txt.count("=== 第 186 轮 ===") == 1
