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

    ev = agent.evolve(with_web=True)
    if not ev.get("ok"):
        log(f"演化未执行：{ev.get('reason')}")
    else:
        log(f"演化: 主题「{ev['query']}」 联网 {ev['web_n']} 条 · 新增设定 {len(ev['added'])} 条")
        for s in ev["added"]:
            log(f"  • {s}")

    w = agent.write_chapter()
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
        else:
            log("自评: 章节为空，跳过")

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

    instance_lock = FileLock(str(STORE_PATH) + ".autonomous.lock", timeout=0.0)
    try:
        instance_lock.acquire()
    except LockTimeoutError:
        log("已有自主写作进程在运行，本次启动退出。")
        return 2

    store = MemoryStore(path=str(STORE_PATH)) if STORE_PATH.exists() else MemoryStore()
    store.path = str(STORE_PATH)
    agent = MemoryAgent(store=store, persona=args.persona,
                        cfg=AgentConfig(evolve_on_sleep=True))
    if not args.persona:
        log("警告：未设置 persona（建议 --persona novelist）")

    log(f"自主成长后台启动：{'无限循环' if args.cycles <= 0 else str(args.cycles) + ' 轮'}，"
        f"轮间休息 {args.min_interval:.0f}~{args.max_interval:.0f}s，"
        f"自评={'开' if not args.no_critique else '关'}")
    log(f"记忆库: {STORE_PATH}")

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
            log(f"记忆已保存 → {STORE_PATH}")
        finally:
            instance_lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
