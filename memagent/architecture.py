"""小说架构文档维护模块（Architecture）——三卷大纲 / 当前卷细纲 / 人物关系 / 世界地理。

四件事：
1. ensure_work_title()   —— 设定积累到一定条数后，让 LLM 投票定书名
2. ensure_architecture()  —— 保证 works/<书名>/ 下有大纲/细纲/人物关系/地理四份文档
3. next_chapter_goal()    —— 给下一章一个明确"剧情目标"，注入写作 prompt
4. update_architecture()  —— 写完一章后把新信息回写到细纲/人物关系

设计映射：当前每章只靠"上一章结尾续写"，剧情在横向漂移；加了细纲和大纲，
agent 就知道"我现在写的是三卷里的哪一段、下一章要达成什么"，从被动续写变成主动推进。
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path


_ARCH_DOCS = ["outline.md", "characters.md", "world.md"]


def _safe_work_name(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", title).strip("._") or "unnamed"


def _arch_dir(work_dir: Path) -> Path:
    # agent._work_dir() 已返回 works/<书名>/chapters，
    # 架构目录应放作品根目录 works/<书名>/architecture/
    root = work_dir.parent if work_dir.name == "chapters" else work_dir
    d = root / "architecture"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chapter_dir(work_dir: Path) -> Path:
    """Accept either a work root or its chapters directory."""
    return work_dir if work_dir.name == "chapters" else work_dir / "chapters"


def migrate_legacy_work(base_dir: Path, old_title: str, new_title: str) -> dict:
    """Move pre-title chapters without ever overwriting an existing work."""
    base_dir = base_dir.resolve()
    source = base_dir / _safe_work_name(old_title)
    target = base_dir / _safe_work_name(new_title)
    if old_title == new_title or not source.exists():
        return {"migrated": False, "reason": "no-source"}

    target_chapters = target / "chapters"
    if not target.exists() or not any(target_chapters.glob("第*章.md")):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            for child in source.iterdir():
                shutil.move(str(child), str(target / child.name))
            source.rmdir()
        else:
            shutil.move(str(source), str(target))
        return {"migrated": True, "destination": str(target), "archived": False}

    archive = target / "legacy" / old_title
    if archive.exists():
        archive = archive.with_name(f"{old_title}-{int(time.time())}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(archive))
    return {"migrated": True, "destination": str(archive), "archived": True}


def ensure_work_title(agent, title: str) -> str:
    """设定积累到 ≥ 6 条且作品名为'未命名作品'时，让 LLM 投 5 个候选名、挑最高票。

    把书名写进 setting 记忆（含《》标记，_work_title 正则可识别）。
    有现成书名则直接返回，不重复投票。
    """
    if title != "未命名作品":
        return title
    settings = [m for m in agent.store.all() if m.kind == "setting"]
    if len(settings) < 6:
        return title
    if agent.responder is None or not getattr(agent.responder, "available", False):
        return title

    # 检查是否已有投票结果
    for m in settings:
        if "书名投票" in m.content and "选定：" in m.content:
            mt = re.search(r"《([^》]+)》", m.content)
            if mt:
                # 替换旧书名
                old = re.search(r"《([^》]+)》", title)
                if old:
                    return mt.group(1)
                return mt.group(1)

    sheet = "\n".join(f"• {m.content}" for m in settings[:15])
    prompt = (
        f"你是一名小说编辑，正在为一部新连载作品确定书名。"
        f"以下是当前已建立的世界观与设定：\n\n{sheet}\n\n"
        f"请投出 5 个候选书名（每个 4-10 字，古风仙侠味，避免俗套），"
        f"格式严格如下，每行一个，不要多余解释：\n"
        f"候选1：《书名A》\n候选2：《书名B》\n候选3：《书名C》\n候选4：《书名D》\n候选5：《书名E》\n"
        f"选定：《你最推荐的那个书名》"
    )
    try:
        reply = agent.call_with_retry(prompt, min_len=1, timeout=60.0)
        mt = re.search(r"选定：《([^》]+)》", reply)
        if mt:
            chosen = mt.group(1)
            agent.remember_setting(f"书名投票：选定《{chosen}》（由 5 候选投票得出）", importance=0.99)
            # 同步替换所有引用"未命名作品"的进度记忆
            for m in agent.store.all():
                if m.kind == "setting" and "未命名作品" in m.content:
                    m.content = m.content.replace("未命名作品", chosen)
            migrate_legacy_work(Path(agent.cfg.chapter_save_dir), "未命名作品", chosen)
            return chosen
    except Exception:
        pass
    return title


def ensure_architecture(work_dir: Path, title: str, agent) -> dict:
    """保证架构文档齐全。缺哪份就生成哪份。返回 {outline, characters, world} 路径字典。"""
    d = _arch_dir(work_dir)

    # 已存在的文档
    existing = {name: (d / name) for name in _ARCH_DOCS if (d / name).is_file()}
    needs = [name for name in _ARCH_DOCS if name not in existing]

    if not needs and (d / "outline.md").is_file():
        return {"outline": str(d / "outline.md"),
                "characters": str(d / "characters.md"),
                "world": str(d / "world.md")}

    # 有 LLM 才生成，否则写占位骨架
    if agent.responder is None or not getattr(agent.responder, "available", False):
        _write_skeleton(d, title, needs)
        return {"outline": str(d / "outline.md"),
                "characters": str(d / "characters.md"),
                "world": str(d / "world.md")}

    sheet = "\n".join(f"• {m.content}" for m in [m for m in agent.store.all() if m.kind == "setting"][:20])

    # 抓已有章节片段做参考
    chap_dir = _chapter_dir(work_dir)
    chap_preview = ""
    if chap_dir.is_dir():
        parts = []
        for p in sorted(chap_dir.glob("*.md"), key=lambda x: x.stat().st_mtime)[:3]:
            txt = p.read_text(encoding="utf-8", errors="replace")
            parts.append(f"【{p.name}】{txt[:500]}")
        chap_preview = "\n\n".join(parts)

    prompt = (
        f"你是一名小说架构师，正在为作品《{title}》搭建写作骨架。"
        f"以下是已建立的设定和已写章节片段：\n\n【设定】\n{sheet}\n\n"
        f"【已写章节片段】\n{chap_preview}\n\n"
        f"请生成以下三份文档的内容，每份用 === 分隔：\n\n"
        f"=== outline.md ===\n"
        f"三卷大纲（每卷 1 段 100-200 字：核心冲突、主要人物、关键转折、卷末悬念）。"
        f"要具体，不要泛泛而谈。\n\n"
        f"=== characters.md ===\n"
        f"人物关系表：每个主要角色一行，格式「姓名 | 身份 | 性格 | 与主角关系 | 关键伏笔」。"
        f"至少 5 人，主角放首位。\n\n"
        f"=== world.md ===\n"
        f"世界地理与势力：列出主要地点（每个 1-2 句：位置、特点、势力归属），"
        f"以及主要势力（名称、立场、与主角关系）。"
    )
    try:
        reply = agent.call_with_retry(prompt, min_len=50, timeout=120.0)
        _parse_and_save(d, title, reply)
    except Exception:
        _write_skeleton(d, title, _ARCH_DOCS)

    return {"outline": str(d / "outline.md"),
            "characters": str(d / "characters.md"),
            "world": str(d / "world.md")}


def _write_skeleton(d: Path, title: str, names: list[str]):
    for name in names:
        p = d / name
        if not p.is_file():
            p.write_text(
                f"# {title} — {name.replace('.md', '').title()}\n\n（待架构生成）\n",
                encoding="utf-8",
            )


def _parse_and_save(d: Path, title: str, reply: str):
    sections = re.split(r"===\s*([\w.]+)\s*===", reply)
    # sections = ['', 'outline.md', 'content...', 'characters.md', 'content...', ...]
    for i in range(1, len(sections) - 1, 2):
        fname = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if fname in _ARCH_DOCS and content:
            p = d / fname
            header = f"# {title} — {fname.replace('.md', '').title()}\n\n"
            p.write_text(header + content, encoding="utf-8")


def next_chapter_goal(agent, title: str, chapter_no: int, work_dir: Path) -> str:
    """给下一章一个明确的剧情目标（1-2 句），注入写作 prompt。

    先从 outline.md + 已写章节判断当前处于哪一阶段；再用 LLM 生成一个具体目标。
    目标要包含：主角要做什么 / 会遇到什么 / 本章末尾该留下什么钩子。
    """
    if agent.responder is None or not getattr(agent.responder, "available", False):
        return ""
    d = _arch_dir(work_dir)
    outline = (d / "outline.md").read_text(encoding="utf-8", errors="replace") if (d / "outline.md").is_file() else ""
    chars = (d / "characters.md").read_text(encoding="utf-8", errors="replace") if (d / "characters.md").is_file() else ""
    world = (d / "world.md").read_text(encoding="utf-8", errors="replace") if (d / "world.md").is_file() else ""
    sheet = "\n".join(f"• {m.content}" for m in [m for m in agent.store.all() if m.kind == "setting"][:12])

    # 抓已写章节末段（判断剧情走到哪了）
    chap_dir = _chapter_dir(work_dir)
    last_chap = ""
    if chap_dir.is_dir():
        files = sorted(chap_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
        if files:
            txt = files[0].read_text(encoding="utf-8", errors="replace")
            last_chap = txt[-800:] if len(txt) > 800 else txt

    prompt = (
        f"你是一名小说编辑，正在为《{title}》第 {chapter_no} 章设计剧情目标。\n\n"
        f"【三卷大纲】\n{outline}\n\n【人物关系】\n{chars}\n\n【世界地理】\n{world}\n\n"
        f"【设定档案】\n{sheet}\n\n【最新章节末段】\n{last_chap}\n\n"
        f"请给出第 {chapter_no} 章的剧情目标（严格 1-3 句，要具体到'谁在哪里做什么、会遇到什么、"
        f"本章末尾该留下什么悬念钩子'）。不要列大纲，不要分点，直接一句话或几句话。\n\n"
        f"格式：剧情目标：..."
    )
    try:
        reply = agent.call_with_retry(prompt, min_len=1, timeout=60.0)
        m = re.search(r"剧情目标[：:]\s*(.+)", reply)
        return m.group(1).strip() if m else reply.strip()[:200]
    except Exception:
        return ""


def update_architecture(title: str, chapter_text: str, chapter_no: int,
                        work_dir: Path, agent) -> None:
    """写完一章后，把本章新出场人物 / 新地点 / 新伏笔 用 LLM 提炼，
    追加到 characters.md / world.md 末尾（增量更新，避免大重写）。"""
    if agent.responder is None or not getattr(agent.responder, "available", False):
        return
    d = _arch_dir(work_dir)
    prompt = (
        f"你是一名小说编辑，正在更新《{title}》第 {chapter_no} 章的架构文档。\n\n"
        f"【本章内容】\n{chapter_text[:4000]}\n\n"
        f"请提炼本章新增的人物、地点、势力信息，格式如下：\n\n"
        f"【新增人物】\n（若无则写'无'）\n"
        f"【新增地点/势力】\n（若无则写'无'）\n"
        f"【重要伏笔】\n（若无则写'无'）"
    )
    try:
        reply = agent.call_with_retry(prompt, min_len=1, timeout=60.0)
        # 简单追加到对应文件
        def _section(text: str, header: str) -> str:
            m = re.search(rf"【{re.escape(header)}】\s*\n(.*?)(?:\n【|\Z)", text, re.S)
            return m.group(1).strip() if m else "无"

        chars_new = _section(reply, "新增人物")
        world_new = _section(reply, "新增地点/势力")
        foreshadow = _section(reply, "重要伏笔")

        if chars_new != "无":
            p = d / "characters.md"
            with p.open("a", encoding="utf-8") as f:
                f.write(f"\n（第{chapter_no}章新增）{chars_new}\n")
        if world_new != "无":
            p = d / "world.md"
            with p.open("a", encoding="utf-8") as f:
                f.write(f"\n（第{chapter_no}章新增）{world_new}\n")
        if foreshadow not in ("无", ""):
            agent.remember_setting(f"第{chapter_no}章伏笔：{foreshadow}", importance=0.85)
        # 写章守卫：新增伏笔记忆可能把作品名挤出人设档案前 8，写完自动复检提升
        agent.ensure_title_in_sheet()
    except Exception:
        pass
