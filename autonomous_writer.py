"""长时间自主成长后台：联网研究 → 演化新设定 → 续写一章 → 自评复盘 → 睡眠巩固，循环运行。

用法（睡觉前挂上即可，Ctrl+C 优雅退出）：
    python autonomous_writer.py --persona novelist --cycles 20
    python autonomous_writer.py --persona novelist          # 无限循环（Ctrl+C 停止）
    python autonomous_writer.py --min-interval 600 --max-interval 3600
    python autonomous_writer.py --no-critique               # 关闭自评（省 token）

每轮动作（写入 works/autonomous.log 与终端）：
    ① evolve(with_web=True)   —— 联网查资料 → 提出并入库新设定
    ② write_chapter()         —— 基于当前人设档案续写下一章（落盘 works/<书名>/）
    ③ critique_chapter()      —— 抓对标作家章节 → 五维自评 → 沉淀写作改进规则
    ④ sleep()                 —— 巩固（含学习器自动调参）
    ⑤ 随机休息 min~max 秒      —— 模拟"间歇性思考"，不狂刷 API 配额

设计映射：作家不是连续赶稿，而是"写一阵 → 反思复盘 → 对标好作家 → 沉淀改进 → 再写"。
自评沉淀的写作改进规则（kind="skill"）遗忘最慢，长期生效——agent 越写越强。
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.cli import enable_utf8  # noqa: E402
from memagent.io_utils import FileLock, LockTimeoutError  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402

STORE_PATH = Path(__file__).resolve().parent / "agent_memory.json"
LOG_PATH = Path(__file__).resolve().parent / "works" / "autonomous.log"

# 每轮是否先跑自主演化（联网查资料 + 新设定入库）。默认开（旧行为）；
# --no-evolve 关闭——大纲驱动的紧结构作品必须关：演化每轮注入新伏笔，
# 悬念只加不减从不兑付，是《错季锁星》"谜题通胀、越写越没劲"的根因之一。
_EVOLVE_ENABLED: bool = True


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 with BOM——PS 5.1+/Win 记事本自动按 UTF-8 读，中文不乱码；
    # Linux 侧 cat 只在文件最开头多一个无影响字符。无 BOM 的 UTF-8 会被
    # PS 误按 GBK 读导致中文乱码，这是你报问题的根因。
    first_write = not LOG_PATH.exists()
    with LOG_PATH.open("ab") as f:
        if first_write:
            f.write(b"\xef\xbb\xbf")
        f.write((line + "\n").encode("utf-8"))


def _replace_chapter(orig_path: Path, rewrite_path: Path, orig_no: int) -> None:
    """用重写稿替换原章：重写稿按新章号生成，标题里的「第N章」要改回原章号，
    否则文件名与正文编号错乱（《斩契》试写第2章实测踩坑：第2章.md 标题写第3章）。"""
    body = rewrite_path.read_text(encoding="utf-8")
    body = re.sub(r"(第)\d+(章)", rf"\g<1>{orig_no}\g<2>", body, count=1)
    orig_path.write_text(body, encoding="utf-8")
    try:
        rewrite_path.unlink()
    except Exception:
        pass


def _ledger_path(work_dir: Path) -> Path:
    """事实台账路径：作品根目录 facts.json。

    agent._work_dir() 返回的是 works/<书名>/chapters（章节目录），architecture.py
    的 _arch_dir 专门做了 parent 归一化——台账同样要放作品根目录。第 5/6 章实跑
    踩坑：漏了归一化 → 台账建在 chapters/facts.json（幽灵文件，首次为空 →
    canon 词全失效 → 术语守卫把「司契/命契」全报新造，重写约束被垃圾淹没）。
    """
    root = work_dir.parent if work_dir.name == "chapters" else work_dir
    return root / "facts.json"


def _prev_chapter_texts(work_dir: Path, before_no: int, k: int = 2) -> list[tuple[int, str]]:
    """取 before_no 之前的 k 章正文（章间重复检测的参照）。"""
    texts: list[tuple[int, str]] = []
    chap_dir = work_dir / "chapters"
    if not chap_dir.is_dir():
        return texts
    for no in range(before_no - 1, max(0, before_no - 1 - k), -1):
        f = chap_dir / f"第{no}章.md"
        if f.is_file():
            try:
                texts.append((no, f.read_text(encoding="utf-8")))
            except Exception:
                pass
    return texts


def _load_voice_profile(work_dir: Path) -> str | None:
    """加载作品目录下的 voice_profile.md（句法/节奏量化约束）。

    voice_profile 是 persona 的**量化补充**——persona 说"对话占比≤40%"，
    voice_profile 把它拆成可验证的指标（段长分布、极简回应词上限、感官词
    最低出现密度、禁止段落形态清单）。LLM 在"写小说"任务上默认会退化到
    "对话剧本+纯对话段堆叠"（《斩契》第8章实测：'对。'出现41次），
    必须有这份量化约束才能把它拉回来。

    若作品目录没有 voice_profile.md，返回 None（不注入任何句法约束，保留旧行为）。
    """
    arch_dir = work_dir / "architecture"
    for candidate in (work_dir / "voice_profile.md", arch_dir / "voice_profile.md"):
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception:
                pass
    return None


def one_cycle(agent: MemoryAgent, i: int, *, do_critique: bool = True) -> dict:
    log(f"=== 第 {i} 轮 ===")

    # 作品命名 + 架构文档：每轮开头确保齐全
    from memagent.architecture import (ensure_work_title,
                                        ensure_architecture,
                                        update_architecture as _up_arch)
    current_title = agent._work_title()
    new_title = ensure_work_title(agent, current_title)
    if new_title != current_title:
        log(f"作品命名: 已投票选定 → 《{new_title}》")

    work_dir = Path(agent._work_dir(new_title))
    arch = ensure_architecture(work_dir, new_title, agent)
    log(f"架构: outline={arch['outline']}  chars={arch['characters']}  world={arch['world']}")

    ev = {"ok": False, "reason": "evolve-off"}
    if _EVOLVE_ENABLED:
        ev = agent.evolve(with_web=True)
        if not ev.get("ok"):
            log(f"演化未执行：{ev.get('reason')}")
        else:
            log(f"演化: 主题「{ev['query']}」 联网 {ev['web_n']} 条 · 新增设定 {len(ev['added'])} 条")
            for s in ev["added"]:
                log(f"  • {s}")

    # ②'' 连续性支撑：事实台账 + 写前节拍表（memagent/continuity.py）。
    # 台账是结构化"世界状态表"——数字/年龄/生死/谁知道什么的唯一权威来源；
    # 节拍表把"角色进场时知道什么、从哪知道"的检查前移到动笔之前。
    from memagent.continuity import FactLedger, beat_sheet
    ledger = FactLedger(_ledger_path(work_dir))
    constraints: list[str] = []
    if ledger.sheet():
        constraints.append(f"事实台账（数字/年龄/生死/谁知道什么，必须与此一致，"
                           f"不得矛盾）：\n{ledger.sheet()}")
    try:
        from memagent.architecture import next_chapter_goal
        nxt = agent._next_chapter(new_title)
        beats = beat_sheet(agent.responder, new_title, nxt,
                           next_chapter_goal(agent, new_title, nxt, work_dir),
                           ledger.sheet(), agent._last_chapter_tail(new_title))
        if beats:
            constraints.append("本章节拍（括号里已给足信息来源，按此推进）：\n"
                               + "\n".join(beats))
            log(f"节拍表: {len(beats)} 拍")
    except Exception as e:
        log(f"节拍表跳过：{str(e)[:60]}")

    # 句法/节奏 voice profile：量化 persona 未覆盖的段落配比、段长分布、
    # 感官词/心理词最低密度、禁止段落形态——不注入则 LLM 会退化到纯对话堆叠
    voice = _load_voice_profile(work_dir)
    if voice:
        constraints.append(f"【句法节奏约束（voice profile，违反任何一条即重写）】\n{voice}")
        log("voice profile: 已加载（基于参考作品量化采样）")

    w = agent.write_chapter(constraints=constraints or None)
    if w.get("ok"):
        title_tag = f" · 「{w['chapter_title']}」" if w.get("chapter_title") else ""
        log(f"写作: 《{w['title']}》第 {w['chapter']} 章{title_tag} · {w['words']} 字 → {w['path']}")
    else:
        log(f"写作未完成：{w.get('reason')}")

    chapter_text = ""
    if do_critique and w.get("ok"):
        from memagent.critique import self_critique, persist_improvements

        try:
            chapter_text = Path(w["path"]).read_text(encoding="utf-8")
        except Exception as e:
            log(f"自评: 读章节失败 {e}")
            chapter_text = ""
        if chapter_text:
            crit = self_critique(
                chapter_text=chapter_text,
                chapter_no=w["chapter"],
                title=w["title"],
                responder=agent.responder,
                persona_sheet=agent.persona_sheet(),
                n_samples=2,
                timeout=60.0,
            )
            if crit is None:
                log("自评: 未执行（LLM 不可用）")
            else:
                scores = ", ".join(f"{k}={v:.1f}" for k, v in crit.scores.items()) or "无分数"
                log(f"自评: 五维 {{ {scores} }} | 网络章节 {crit.benchmark_note}")
                if crit.overall:
                    log(f"  综合: {crit.overall[:120]}")
                n = persist_improvements(agent, crit)
                log(f"  沉淀改进 {n} 条")
                for imp in crit.improvements[:3]:
                    log(f"    → {imp}")
                # 读者硬门槛：读者友好度（术语不劝退）与追读欲（爽点兑现/钩子具体）
                # 取低者，低于 6 分强制重写本章（最多 2 次）——两维任一崩坏都留不住读者
                # 策略：重写生成新章（章号+1），若达标则删除原章、重命名新章为原章号，保持连续性
                def _reader_gate(s: dict) -> float:
                    return min(s.get("读者友好度", 10.0), s.get("追读欲", 10.0))

                reader_score = _reader_gate(crit.scores)
                rewrite_attempts = 0
                max_rewrites = 2
                orig_path = Path(w["path"])
                orig_no = w["chapter"]
                while reader_score < 6.0 and rewrite_attempts < max_rewrites:
                    rewrite_attempts += 1
                    log(f"  读者友好度 {reader_score:.1f} < 6.0，强制重写第{orig_no}章（第 {rewrite_attempts}/{max_rewrites} 次）…")
                    result = agent.write_chapter(constraints=constraints or None)
                    if result.get("ok"):
                        rewrite_path = Path(result["path"])
                        chapter_text = rewrite_path.read_text(encoding="utf-8")
                        crit2 = self_critique(
                            chapter_text=chapter_text,
                            chapter_no=result["chapter"],
                            title=result["title"],
                            responder=agent.responder,
                            persona_sheet=agent.persona_sheet(),
                            n_samples=2,
                            timeout=60.0,
                        )
                        if crit2:
                            reader_score2 = _reader_gate(crit2.scores)
                            scores2 = ", ".join(f"{k}={v:.1f}" for k, v in crit2.scores.items()) or "无分数"
                            log(f"  重写自评: {{ {scores2} }}")
                            if crit2.overall:
                                log(f"    综合: {crit2.overall[:120]}")
                            n = persist_improvements(agent, crit2)
                            log(f"    沉淀改进 {n} 条")
                            # 达标：用新章内容替换原章，删除多余章
                            if reader_score2 >= 6.0:
                                _replace_chapter(orig_path, rewrite_path, orig_no)
                                # 更新 w 指向原章
                                w = dict(result); w["path"] = str(orig_path); w["chapter"] = orig_no
                                reader_score = reader_score2
                            else:
                                reader_score = reader_score2
                    else:
                        log(f"  重写失败：{result.get('reason')}")
                        break
                if reader_score >= 6.0:
                    log(f"  ✓ 读者友好度达标 {reader_score:.1f}（{rewrite_attempts} 次重写）")
                else:
                    log(f"  ⚠ 读者友好度仍未达标 {reader_score:.1f}，本章按原样保留")
        else:
            log("自评: 章节为空，跳过")

    # ③' 连续性审校：确定性检查（死人开口/年龄/数字矛盾）+ 知识状态因果链
    # （"角色说的每条信息，他从何得知"）——问题条目注入重写，最多 2 次；
    # 通过后把本章事实抽回台账（世界状态表随章更新）。
    if w.get("ok"):
        from memagent.continuity import review_chapter, extract_facts
        try:
            chapter_text = Path(w["path"]).read_text(encoding="utf-8")
        except Exception:
            chapter_text = ""
        cont = None
        if chapter_text:
            try:
                cont = review_chapter(agent.responder, ledger, chapter_text,
                                      w["chapter"], w["title"],
                                      prev_chapters=_prev_chapter_texts(work_dir, w["chapter"]))
            except Exception as e:
                log(f"连续性审校异常（跳过）：{str(e)[:60]}")
        if cont is not None:
            for d in cont["det_issues"]:
                log(f"  ⚠ 连续性·确定: {d}")
            for it in cont["llm_issues"]:
                log(f"  ⚠ 连续性·因果: 「{it['quote']}」{it['problem']}")
            attempts, max_fix = 0, 2
            while not cont["ok"] and attempts < max_fix:
                attempts += 1
                log(f"  连续性问题 {len(cont['det_issues']) + len(cont['llm_issues'])} 条，"
                    f"重写第{w['chapter']}章（第 {attempts}/{max_fix} 次）…")
                fixes = [f"修正：{it['quote']} → {it['problem']}"
                         + (f"（改法：{it['fix']}）" if it["fix"] else "")
                         for it in cont["llm_issues"]]
                fixes += [f"修正：{d}" for d in cont["det_issues"]]
                # 禁用名/污染词守卫的硬提示——重写阶段 LLM 最容易复发（第7/8章实测）
                banned = [b for b in (ledger.data.get("banned_names") or [])
                          if isinstance(b, str) and b]
                if banned:
                    fixes.append(
                        f"禁用词铁律：以下词一律不得出现在正文任何位置（含对话、旁白、"
                        f"心理描写、比喻）：{'/'.join(banned)}。如需指代已逝者的灵契虚像，"
                        f"统一用「灵契虚像」或角色原称。")
                # 术语白名单：把台账允许词列明，让 LLM 收敛到词集而非自由造词
                allowed_terms = ledger.data.get("terms") or []
                if allowed_terms:
                    fixes.append(
                        f"允许使用的命契体系术语（仅可从以下词表中取，禁止自造新的"
                        f"含「契/境/纹/虫」的核心术语）：{'/'.join(allowed_terms)}")
                result = agent.write_chapter(constraints=(constraints or []) + fixes)
                if not result.get("ok"):
                    log(f"  连续性重写失败：{result.get('reason')}")
                    break
                new_text = Path(result["path"]).read_text(encoding="utf-8")
                try:
                    cont2 = review_chapter(agent.responder, ledger, new_text,
                                           result["chapter"], result["title"],
                                           prev_chapters=_prev_chapter_texts(work_dir, w["chapter"]))
                except Exception:
                    break
                n_old = len(cont["det_issues"]) + len(cont["llm_issues"])
                n_new = len(cont2["det_issues"]) + len(cont2["llm_issues"])
                if cont2["ok"] or n_new < n_old:
                    _orig_p, _orig_no = Path(w["path"]), w["chapter"]
                    _replace_chapter(_orig_p, Path(result["path"]), _orig_no)
                    w = dict(result); w["path"] = str(_orig_p); w["chapter"] = _orig_no
                    chapter_text = new_text
                    cont = cont2
                    for d in cont2["det_issues"]:
                        log(f"    仍存·确定: {d}")
                    for it in cont2["llm_issues"]:
                        log(f"    仍存·因果: 「{it['quote']}」{it['problem']}")
                else:
                    # 重写没变好：弃新稿保旧稿
                    try:
                        Path(result["path"]).unlink()
                    except Exception:
                        pass
                    log("  重写未改善，保留原稿")
            if cont["ok"]:
                log("  ✓ 连续性审校通过")
            else:
                log(f"  ⚠ 连续性仍有 {len(cont['det_issues']) + len(cont['llm_issues'])} 条未清，本章按原样保留")
        # ③'' 文学审校（确定性）：句法节奏、对话密度、感官/心理/动作密度、
        # 极简回应词重复。这是连续性审校不会覆盖的另一半——"不错"vs"好看"。
        # 若命中则强制重写（最多 2 次），把问题条目注入 rewrite constraints。
        # 与连续性审校**独立运行**，互不影响；即使连续性已 ok，文学审校仍可触发重写。
        from memagent.literary import literary_checks as _literary_checks
        try:
            lit_issues = _literary_checks(chapter_text) if chapter_text else []
        except Exception as e:
            lit_issues = []
            log(f"文学审校异常（跳过）：{str(e)[:60]}")
        if lit_issues:
            log(f"文学审校: 命中 {len(lit_issues)} 条，强制重写第{w['chapter']}章")
            for d in lit_issues:
                log(f"  ⚠ 文学: {d}")
            lit_attempts, lit_max = 0, 2
            while lit_issues and lit_attempts < lit_max:
                lit_attempts += 1
                log(f"  文学重写第{w['chapter']}章（第 {lit_attempts}/{lit_max} 次）…")
                fixes = [f"【文学硬约束-违反即重写】{d}" for d in lit_issues]
                result = agent.write_chapter(constraints=(constraints or []) + fixes)
                if not result.get("ok"):
                    log(f"  文学重写失败：{result.get('reason')}")
                    break
                new_text = Path(result["path"]).read_text(encoding="utf-8")
                try:
                    lit_issues2 = _literary_checks(new_text)
                except Exception:
                    break
                # 改善即采纳（不要求全部清 0，与连续性重写一致）
                if len(lit_issues2) < len(lit_issues):
                    _orig_p, _orig_no = Path(w["path"]), w["chapter"]
                    _replace_chapter(_orig_p, Path(result["path"]), _orig_no)
                    w = dict(result); w["path"] = str(_orig_p); w["chapter"] = _orig_no
                    chapter_text = new_text
                    lit_issues = lit_issues2
                    for d in lit_issues:
                        log(f"    仍存·文学: {d}")
                else:
                    try:
                        Path(result["path"]).unlink()
                    except Exception:
                        pass
                    log("  文学重写未改善，保留原稿")
                    break
            if lit_issues:
                log(f"  ⚠ 文学仍有 {len(lit_issues)} 条未清，本章按原样保留")
            else:
                log("  ✓ 文学审校通过")
        # 台账回填：把最终章的事实抽进 facts.json（LLM 不可用/解析失败则跳过）
        try:
            facts = extract_facts(agent.responder, w["title"], w["chapter"], chapter_text)
            if facts:
                ledger.merge_chapter(w["chapter"], facts)
                ledger.save()
                log(f"  事实台账: 已回填第{w['chapter']}章"
                    f"（人物 {len(facts.get('characters') or [])} · 数字 {len(facts.get('numbers') or [])}）")
            else:
                log(f"  ⚠ 事实台账: 第{w['chapter']}章抽取未返回有效 JSON（跳过回填，台账不含本章）")
        except Exception as e:
            log(f"  事实台账回填失败（跳过）：{str(e)[:60]}")

    # 架构文档增量更新：把本章新出场人物 / 新地点 / 新伏笔 追加进架构
    if w.get("ok"):
        try:
            _up_arch(w["title"], chapter_text or Path(w["path"]).read_text(encoding="utf-8"),
                     w["chapter"], work_dir, agent)
        except Exception:
            pass

    sr = agent.sleep()
    log(f"睡眠: 回放 {sr.get('replayed_count', 0)} · 冷压缩 {sr.get('cold_compressed', 0)} · "
        f"演化入库 {len(sr.get('evolved', []))} · 类型迁移 {sr.get('migrations', 0)}")
    return {"write_ok": bool(w.get("ok")), "evolve_ok": bool(ev.get("ok"))}


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="memagent 自主成长后台")
    parser.add_argument("--persona", default=os.environ.get("OPENAI_PERSONA"),
                        help="人设（novelist/小说家 或自定义文本）")
    parser.add_argument("--persona-file", default=None,
                        help="从文件读人设文本（长人设免命令行转义；覆盖 --persona）")
    parser.add_argument("--store", default=None,
                        help="记忆库 JSON 路径（默认 agent_memory.json；新书用独立库避免旧书设定污染人设档案）")
    parser.add_argument("--no-evolve", action="store_true",
                        help="关闭每轮自主演化（大纲驱动的紧结构作品应关：防谜题通胀）")
    parser.add_argument("--cycles", type=int, default=10, help="循环轮数（默认 10；0=无限）")
    parser.add_argument("--min-interval", type=float, default=120.0,
                        help="轮间最小休息秒数（默认 120）")
    parser.add_argument("--max-interval", type=float, default=600.0,
                        help="轮间最大休息秒数（默认 600）")
    parser.add_argument("--no-critique", action="store_true",
                        help="关闭自评复盘（省 token；自评会联网抓对标章节）")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="连续写作失败达到该次数后熔断退出（默认 3）")
    args = parser.parse_args()

    persona = args.persona
    if args.persona_file:
        persona = Path(args.persona_file).read_text(encoding="utf-8").strip()
        if not persona:
            log(f"警告：--persona-file {args.persona_file} 为空，回退 --persona/环境变量")
            persona = args.persona

    global _EVOLVE_ENABLED
    _EVOLVE_ENABLED = not args.no_evolve

    store_path = Path(args.store) if args.store else STORE_PATH

    instance_lock = FileLock(str(store_path) + ".autonomous.lock", timeout=0.0)
    try:
        instance_lock.acquire()
    except LockTimeoutError:
        log("已有自主写作进程在运行，本次启动退出。")
        return 2

    store = MemoryStore(path=str(store_path)) if store_path.exists() else MemoryStore()
    store.path = str(store_path)
    agent = MemoryAgent(store=store, persona=persona,
                        cfg=AgentConfig(evolve_on_sleep=not args.no_evolve))
    if not persona:
        log("警告：未设置 persona（建议 --persona novelist 或 --persona-file）")

    log(f"自主成长后台启动：{'无限循环' if args.cycles <= 0 else str(args.cycles) + ' 轮'}，"
        f"轮间休息 {args.min_interval:.0f}~{args.max_interval:.0f}s，"
        f"自评={'开' if not args.no_critique else '关'}")
    log(f"记忆库: {store_path}")
    log(f"演化: {'开（每轮联网+新设定入库）' if _EVOLVE_ENABLED else '关（--no-evolve，大纲驱动）'}")

    i = 0
    consecutive_failures = 0
    try:
        while args.cycles <= 0 or i < args.cycles:
            i += 1
            result = {"write_ok": False}
            try:
                result = one_cycle(agent, i, do_critique=not args.no_critique)
            except Exception as e:
                import traceback
                log(f"第 {i} 轮异常: {e}")
                log("".join(traceback.format_exc(limit=8).splitlines(keepends=True)))
            agent.save()
            if result.get("write_ok"):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                log(f"连续写作失败 {consecutive_failures}/{args.max_consecutive_failures}")
                if consecutive_failures >= max(1, args.max_consecutive_failures):
                    log("触发失败熔断，停止后台循环以保护 API 额度和产物。")
                    break
            if args.cycles <= 0 or i < args.cycles:
                rest = random.uniform(args.min_interval, args.max_interval)
                log(f"休息 {rest:.0f}s …")
                time.sleep(rest)
    except KeyboardInterrupt:
        log("收到中断，保存后退出。")
    finally:
        try:
            agent.save()
            log(f"记忆已保存 → {store_path}")
        finally:
            instance_lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
