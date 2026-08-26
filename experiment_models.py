#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多模型写作对照实验：同卵婴儿种子、同一技法库，各写 N 章，固定评委盲评。

公平性设计：
- 隔离：每个模型一个独立目录（独立记忆库+作品目录），互不污染；
- 同卵：所有模型从**完全相同的种子状态**出生——同一故事设定、同一组
  写作技法记忆，唯一变量是模型本身；
- 固定评委：评分用 --judge 指定的模型（默认取第一个写作模型，注意
  自评偏好偏差；严肃对比请显式指定第三方评委）。

用法：
  python experiment_models.py --models glm-5.2 deepseek-v4-flash \
      --chapters 10 --words 2200 --judge glm-5.2
  python experiment_models.py --smoke          # 单模型单章小字数验证链路

⚠️ 成本提示：10 章 × N 模型 ≈ 数十万 token 与数小时（含审校重写循环），
建议夜间运行；先用 --smoke 验证链路。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os_environ_guard = None  # 本脚本需要真实 .env（OPENAI_*），不要设 MEMAGENT_TEST

ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT / "experiments"

# ---------- 同卵种子：所有模型共享的出生状态 ----------

SEED_PREMISE = [
    ("《测试之书》是本玄幻小说，讲述少年林澈在灵气复苏的现代都市觉醒古法修"
     "炼能力的故事", "setting", 0.95),
    ("主角林澈：19 岁大学生，性格外冷内热，觉醒能力为「触物回溯」",
     "setting", 0.9),
    ("配角苏晚：考古系学姐，冷静理性，知晓林澈的秘密并协助他", "setting", 0.85),
    ("反派组织「归墟会」：试图收集上古灵物打开封印，行事隐秘", "setting", 0.85),
    ("世界观：灵气每百年潮汐一次，本次潮汐伴随空间裂隙出现", "setting", 0.8),
    ("文风要求：短句为主，画面感强，每章至少一个钩子结尾", "skill", 0.85),
]

SEED_TECHNIQUES = [
    "开篇三段内必须出现具体冲突或异常信号，禁止天气铺垫式开场",
    "战斗描写先写声音与光影变化再写招式名，动作结果留白一格给读者",
    "对话每次不超过三行就插入一个动作或环境拍点，防止话剧化",
    "伏笔遵循「埋设-强化-回收」三步，同一伏笔间隔至少两章再回收",
    "心理描写用身体反应替代直陈情绪（手心出汗而非他很紧张）",
    "章节结尾钩子分三级：危机型/悬念型/反转型，连续两章不得同级",
]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", name).strip("-")[:40]


def _make_agent(model: str, run_dir: Path):
    from memagent import MemoryAgent
    from memagent.agent import AgentConfig
    from memagent.responder import LLMResponder

    store_path = run_dir / "memory.json"
    agent = MemoryAgent(
        persist_path=str(store_path),
        cfg=AgentConfig(
            chapter_save_dir=str(run_dir / "works"),
            evolve_on_sleep=False,
            query_expansion=True,
        ),
        responder=LLMResponder(model=model, timeout=180.0),
    )
    return agent


def _seed_baby(agent) -> None:
    """写入同卵种子：设定 + 技法（所有模型完全一致）。"""
    for content, kind, imp in SEED_PREMISE:
        agent.remember(content, kind=kind, importance=imp)
    for t in SEED_TECHNIQUES:
        agent.remember(t, kind="skill", importance=0.75)


def _avg_score(crit) -> float | None:
    if crit is None or not getattr(crit, "scores", None):
        return None
    vals = [float(v) for v in crit.scores.values()]
    return round(sum(vals) / len(vals), 2) if vals else None


def run_model(model: str, chapters: int, words: int,
              judge_model: str, exp_dir: Path) -> dict:
    """单个模型的完整实验：婴儿种子 → 连续 N 章 → 逐章评委打分。"""
    from memagent.critique import self_critique
    from memagent.responder import LLMResponder

    run_dir = exp_dir / _sanitize(model)
    run_dir.mkdir(parents=True, exist_ok=True)
    agent = _make_agent(model, run_dir)
    _seed_baby(agent)

    judge = (LLMResponder(model=judge_model, timeout=120.0)
             if judge_model else None)

    rows = []
    failures = 0
    for ch in range(1, chapters + 1):
        print(f"  [{model}] 第 {ch}/{chapters} 章…", flush=True)
        try:
            result = agent.write_chapter(target_words=words, with_web=False)
        except Exception as e:
            result = {"ok": False, "reason": f"exception:{type(e).__name__}"}
        if not result.get("ok"):
            failures += 1
            rows.append({"chapter": ch, "ok": False,
                         "reason": result.get("reason", "?")})
            continue
        path = Path(result["path"])
        text = path.read_text(encoding="utf-8")
        title = result.get("chapter_title") or f"第{ch}章"
        crit = None
        if judge is not None:
            try:
                crit = self_critique(text, ch, title, judge,
                                     persona_sheet=agent.persona_sheet(),
                                     n_samples=2)
            except Exception:
                crit = None
        rows.append({
            "chapter": ch, "ok": True, "title": title,
            "words": len(text), "score": _avg_score(crit),
            "sub": crit.scores if crit else {},
            "path": str(path),
        })
        print(f"    -> {len(text)} 字 评分 {_avg_score(crit)}", flush=True)
    agent.save()

    scores = [r["score"] for r in rows if r.get("score") is not None]
    return {
        "model": model, "rows": rows, "failures": failures,
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "best": max(scores) if scores else None,
        "worst": min(scores) if scores else None,
        "total_words": sum(r.get("words", 0) for r in rows),
    }


def render_report(results: list[dict], judge: str, out: Path) -> None:
    lines = [
        "# 多模型写作对照实验报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　|　"
        f"评委模型：{judge or '（各模型自评，存在自评偏差）'}",
        "",
        "## 总览",
        "",
        "| 模型 | 平均分 | 最高 | 最低 | 失败章数 | 总字数 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        ms = r["mean_score"] if r["mean_score"] is not None else "-"
        lines.append(f"| {r['model']} | {ms} | {r['best'] or '-'} | "
                     f"{r['worst'] or '-'} | {r['failures']} | {r['total_words']} |")
    lines += ["", "## 分章明细", ""]
    for r in results:
        lines.append(f"### {r['model']}")
        lines.append("")
        lines.append("| 章 | 标题 | 字数 | 评分 |")
        lines.append("|---|---|---|---|")
        for row in r["rows"]:
            if row.get("ok"):
                sc = row["score"] if row["score"] is not None else "-"
                lines.append(f"| {row['chapter']} | {row.get('title','-')} | "
                             f"{row.get('words',0)} | {sc} |")
            else:
                lines.append(f"| {row['chapter']} | ❌ {row.get('reason')} | - | - |")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多模型写作对照实验")
    ap.add_argument("--models", nargs="+", required=True, help="参赛模型名列表")
    ap.add_argument("--chapters", type=int, default=10)
    ap.add_argument("--words", type=int, default=2200, help="每章目标字数")
    ap.add_argument("--judge", default=None,
                    help="评委模型（建议第三方；缺省=各自自评，有偏差）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：仅第一个模型、1 章、600 字")
    args = ap.parse_args(argv)

    models = args.models[:1] if args.smoke else args.models
    chapters, words = (1, 600) if args.smoke else (args.chapters, args.words)
    judge = args.judge or models[0]

    stamp = datetime.now().strftime("%m%d-%H%M")
    tag = "smoke" if args.smoke else f"{chapters}x{len(models)}"
    exp_dir = EXP_ROOT / f"{stamp}-{tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for m in models:
        print(f"=== 模型 {m} ===", flush=True)
        results.append(run_model(m, chapters, words, judge, exp_dir))

    out = exp_dir / "report.md"
    render_report(results, judge, out)
    (exp_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
