"""轮间休息时长实验对比：不同休息区间下的均分 / LLM 抖动率 / 失败率。

用法：
    python compare_rest_experiment.py            # 全量输出（基线 vs 实验段）
    python compare_rest_experiment.py --csv      # 同时导出每轮明细 CSV

分段约定（按日志轮头切段，排除「受控验证」轮）：
    --baseline-lo/hi  基线轮段（默认 186-189：300-900s 间隔的接入后自然轮）
    --exp-lo/hi       实验轮段（默认 190-209：60-120s 间隔 × 20 轮）

指标口径与 track_coder_trend 一致：
    均分 = 该段第一条「自检验: 五维」五维均值（取首条，排除综合行）；
    抖动率 = 段内「LLM 回复为空/异常（第 N/3 次）」行数 / 有效轮数；
    失败率 = 段内「生成代码失败」轮数 / 总轮数；
    休息 = 段内「休息 Ns」行（每轮一条，取首条）。

设计依据（2026-08-15）：
    之前诊断「休息没有降低抖动」——轮间 300-900s 休息只放慢节奏，空回复在
    休息后 3 分钟内仍复发（服务端问题）。本实验把间隔缩到 60-120s 跑 20 轮，
    用均分 / 抖动率 / 失败率对比，数据决定休息时长是否合理。
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "works" / "foxtable_coder.log"


def load_segments(txt: str) -> dict[int, dict]:
    """按「=== 第 N 轮 ===」轮头切段，标记受控验证轮。"""
    segments: dict[int, dict] = {}
    cur: int | None = None
    for ln in txt.splitlines():
        m = re.search(r"=== 第 (\d+) 轮 ===", ln)
        if m:
            cur = int(m.group(1))
            segments.setdefault(cur, {"lines": [], "verify": False})
            continue
        if cur is None:
            continue
        seg = segments[cur]
        seg["lines"].append(ln)
        if "受控验证" in ln:
            seg["verify"] = True
    return segments


def analyze_rounds(segments: dict, lo: int, hi: int,
                   exclude_verify: bool = True) -> list[dict]:
    """解析轮段 [lo, hi]，返回每轮明细 dict 列表。"""
    rows: list[dict] = []
    for k in sorted(segments):
        if not (lo <= k <= hi):
            continue
        seg = segments[k]
        if exclude_verify and seg["verify"]:
            continue
        score: float | None = None
        retries = 0
        fails = 0
        rest: int | None = None
        for ln in seg["lines"]:
            if score is None and "自检验: 五维" in ln:
                nums = [float(x) for x in re.findall(r"=(\d+\.\d)", ln)]
                # 五维行可能缺某维（如 LLM 只给 4 维）——有几维按几维均分
                if len(nums) >= 3:
                    score = sum(nums) / len(nums)
            if "LLM 回复为空/异常" in ln:
                retries += 1
            if "生成代码失败" in ln:
                fails += 1
            m = re.search(r"休息 (\d+)s", ln)
            if m and rest is None:
                rest = int(m.group(1))
        rows.append({"round": k, "score": score, "retries": retries,
                     "fails": fails, "rest": rest})
    return rows


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["score"] is not None]
    rests = [r["rest"] for r in rows if r["rest"] is not None]
    total = len(rows)
    return {
        "rounds": total,
        "scored": len(scored),
        "avg": statistics.mean([r["score"] for r in scored]) if scored else None,
        "min": min((r["score"] for r in scored), default=None),
        "max": max((r["score"] for r in scored), default=None),
        "retries": sum(r["retries"] for r in rows),
        "retries_per_round": (sum(r["retries"] for r in rows) / total) if total else 0.0,
        "fails": sum(r["fails"] for r in rows),
        "fail_rate": (sum(r["fails"] for r in rows) / total) if total else 0.0,
        "rest_mean": statistics.mean(rests) if rests else None,
        "rest_min": min(rests) if rests else None,
        "rest_max": max(rests) if rests else None,
    }


def fmt(v, digits: int = 2) -> str:
    return f"{v:.{digits}f}" if v is not None else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-lo", type=int, default=186)
    ap.add_argument("--baseline-hi", type=int, default=189)
    ap.add_argument("--exp-lo", type=int, default=190)
    ap.add_argument("--exp-hi", type=int, default=209)
    ap.add_argument("--csv", action="store_true", help="导出每轮明细 CSV")
    ap.add_argument("--no-baseline", action="store_true",
                    help="只输出实验段（无基线轮段时用）")
    args = ap.parse_args()

    txt = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    segments = load_segments(txt)

    exp_rows = analyze_rounds(segments, args.exp_lo, args.exp_hi)
    exp = summarize(exp_rows)

    print(f"数据源: {LOG_PATH}")
    print(f"实验段: 轮 {args.exp_lo}-{args.exp_hi}（休息 60-120s）")
    if not exp["scored"]:
        print("实验段尚无有效轮（日志还没积累够），稍后再跑。")
        return 1

    if not args.no_baseline:
        base_rows = analyze_rounds(segments, args.baseline_lo, args.baseline_hi)
        base = summarize(base_rows)
        if not base["scored"]:
            print("基线段无有效轮，仅输出实验段。")
        else:
            print(f"基线段: 轮 {args.baseline_lo}-{args.baseline_hi}（休息 300-900s）")
            print()
            print(f"{'指标':<14}{'基线(300-900s)':>16}{'实验(60-120s)':>16}{'Δ':>10}")
            print("-" * 58)
            print(f"{'有效轮':<14}{base['rounds']:>16}{exp['rounds']:>16}{'':>10}")
            print(f"{'有分轮':<14}{base['scored']:>16}{exp['scored']:>16}{'':>10}")
            print(f"{'均分':<14}{fmt(base['avg']):>16}{fmt(exp['avg']):>16}"
                  f"{fmt(exp['avg']-base['avg'] if base['avg'] and exp['avg'] else None, 2):>10}")
            print(f"{'min/max':<14}{fmt(base['min'])+'/'+fmt(base['max']):>16}"
                  f"{fmt(exp['min'])+'/'+fmt(exp['max']):>16}{'':>10}")
            print(f"{'空回复次数':<14}{base['retries']:>16}{exp['retries']:>16}{'':>10}")
            print(f"{'空回复/轮':<14}{base['retries_per_round']:>16.2f}{exp['retries_per_round']:>16.2f}"
                  f"{exp['retries_per_round']-base['retries_per_round']:>+10.2f}")
            print(f"{'失败轮':<14}{base['fails']:>16}{exp['fails']:>16}{'':>10}")
            print(f"{'失败率':<14}{base['fail_rate']:>16.2%}{exp['fail_rate']:>16.2%}{'':>10}")
            print(f"{'休息mean':<14}{fmt(base['rest_mean'],0):>16}{fmt(exp['rest_mean'],0):>16}{'':>10}")
    else:
        print()
        print(f"实验段（轮 {args.exp_lo}-{args.exp_hi}）: 均分 {fmt(exp['avg'])} · "
              f"空回复 {exp['retries']} 次 = {exp['retries_per_round']:.2f}/轮 · "
              f"失败 {exp['fails']}（{exp['fail_rate']:.0%}）")
        if exp["rest_mean"]:
            print(f"休息 mean {exp['rest_mean']:.0f}s（{exp['rest_min']}-{exp['rest_max']}s）")

    if args.csv:
        out = Path("works/rest_experiment_rounds.csv")
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "score", "retries", "fails", "rest_seconds", "segment"])
            for r in base_rows:
                w.writerow([r["round"], r["score"], r["retries"], r["fails"],
                            r["rest"], "baseline"])
            for r in exp_rows:
                w.writerow([r["round"], r["score"], r["retries"], r["fails"],
                            r["rest"], "experiment"])
        print(f"\n每轮明细已导出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
