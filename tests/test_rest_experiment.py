"""休息时长实验对比脚本（compare_rest_experiment.py）解析逻辑的单元测试。"""
import re

import compare_rest_experiment as cre


def _txt():
    # 轮 1：完整轮（有分/重试/休息）；轮 2：受控验证轮（应被排除）；
    # 轮 3：无分轮（自检验异常，只有重试）；轮 4：失败轮
    return """[t] === 第 1 轮 ===
[t] 抽题: 领域「Table」题目：...
[t] 自检验: 五维 {语法正确性=7.0, API 规范性=8.0, 铁律遵守=6.0, 实战可用性=9.0, 最佳实践=5.0}
[t] LLM 回复为空/异常（第 1/3 次），5s 后重试：LLM 回复为空
[t] 休息 450s ...
[t] === 第 2 轮 ===
[t] 抽题: 领域「DataTable」题目：...（受控验证·注入修复后管线）
[t] 自检验: 五维 {语法正确性=3.0, API 规范性=3.0, 铁律遵守=3.0, 实战可用性=3.0, 最佳实践=3.0}
[t] === 第 3 轮 ===
[t] 抽题: 领域「JSON相关」题目：...
[t] LLM 回复为空/异常（第 1/3 次）
[t] LLM 回复为空/异常（第 2/3 次）
[t] === 第 4 轮 ===
[t] 抽题: 领域「HttpClient」题目：...
[t] 生成代码失败: LLM 回复为空
"""


def test_load_segments_marks_verify():
    segs = cre.load_segments(_txt())
    assert set(segs.keys()) == {1, 2, 3, 4}
    assert segs[2]["verify"] is True
    assert segs[1]["verify"] is False


def test_analyze_rounds_excludes_verify():
    segs = cre.load_segments(_txt())
    rows = cre.analyze_rounds(segs, 1, 4)
    rounds = [r["round"] for r in rows]
    assert rounds == [1, 3, 4]  # 轮 2 受控验证被排除


def test_score_is_first_five_dim_mean():
    segs = cre.load_segments(_txt())
    rows = cre.analyze_rounds(segs, 1, 1)
    assert rows[0]["score"] == 7.0  # (7+8+6+9+5)/5 = 7.0
    assert rows[0]["retries"] == 1
    assert rows[0]["rest"] == 450


def test_no_score_round_counts_retries():
    segs = cre.load_segments(_txt())
    rows = cre.analyze_rounds(segs, 3, 3)
    assert rows[0]["score"] is None
    assert rows[0]["retries"] == 2  # 两次空回复重试都计入抖动率


def test_fail_round_detected():
    segs = cre.load_segments(_txt())
    rows = cre.analyze_rounds(segs, 4, 4)
    assert rows[0]["fails"] == 1
    assert rows[0]["score"] is None


def test_summarize_metrics():
    segs = cre.load_segments(_txt())
    rows = cre.analyze_rounds(segs, 1, 4)
    s = cre.summarize(rows)
    assert s["rounds"] == 3          # 1/3/4
    assert s["scored"] == 1
    assert s["avg"] == 7.0
    assert s["retries"] == 3         # 轮1 一次 + 轮3 两次
    assert s["retries_per_round"] == 1.0
    assert s["fails"] == 1
    assert s["rest_mean"] == 450.0
