"""Audit, snapshot, and safely repair locally stored novel chapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from .cli import enable_utf8
from .io_utils import FileLock, atomic_write_json, atomic_write_text


_CHAPTER_FILE = re.compile(r"^第(?P<number>\d+)章\.md$")
_COMPLETE_ENDING = re.compile(r'[。！？!?….](?:["”’」』）】])?$')
_WRITE_LOG = re.compile(
    r"\[(?P<timestamp>[^]]+)]\s+写作:\s+《(?P<work>[^》]+)》第\s*"
    r"(?P<number>\d+)\s*章(?:\s*·\s*「(?P<title>[^」]+)」)?\s*·\s*"
    r"(?P<characters>\d+)\s*字"
)


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chapter_number(path: Path) -> int | None:
    match = _CHAPTER_FILE.match(path.name)
    return int(match.group("number")) if match else None


def _body(text: str) -> str:
    lines = text.lstrip("\ufeff").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _body_characters(text: str) -> int:
    return len(re.sub(r"\s+", "", _body(text)))


def _heading(text: str) -> str:
    first = text.lstrip("\ufeff").splitlines()[0] if text.strip() else ""
    return first[2:].strip() if first.startswith("# ") else ""


def _has_complete_ending(text: str) -> bool:
    return bool(_COMPLETE_ENDING.search(_body(text).rstrip()))


def _chapter_record(path: Path, root: Path, role: str) -> dict:
    text = path.read_text(encoding="utf-8")
    return {
        "number": _chapter_number(path),
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "heading": _heading(text),
        "body_characters": _body_characters(text),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "modified_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime)
        ),
    }


def audit_work(work_dir: str | Path, minimum_characters: int = 900) -> dict:
    root = Path(work_dir).resolve()
    canonical_paths = sorted(
        (p for p in (root / "chapters").glob("第*章.md") if _chapter_number(p)),
        key=lambda p: _chapter_number(p) or 0,
    )
    draft_paths = sorted(
        (p for p in (root / "legacy").glob("**/chapters/第*章.md") if _chapter_number(p)),
        key=lambda p: (str(p.parent), _chapter_number(p) or 0),
    )
    canonical = [_chapter_record(path, root, "canonical") for path in canonical_paths]
    drafts = [_chapter_record(path, root, "pre-title-draft") for path in draft_paths]
    numbers = [record["number"] for record in canonical]
    missing = (
        sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))
        if numbers else []
    )
    draft_numbers = {record["number"] for record in drafts}
    incomplete = []
    heading_issues = []
    for path, record in zip(canonical_paths, canonical):
        text = path.read_text(encoding="utf-8")
        if not _has_complete_ending(text):
            incomplete.append(record["number"])
        expected = f"《{root.name}》第{record['number']}章"
        if expected not in record["heading"]:
            heading_issues.append(record["number"])
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "work": root.name,
        "canonical_count": len(canonical),
        "canonical_range": [min(numbers), max(numbers)] if numbers else [],
        "missing_canonical_numbers": missing,
        "short_canonical_numbers": [
            record["number"]
            for record in canonical
            if record["body_characters"] < minimum_characters
        ],
        "incomplete_ending_numbers": incomplete,
        "heading_issue_numbers": heading_issues,
        "pre_title_draft_count": len(drafts),
        "overlapping_draft_numbers": sorted(set(numbers) & draft_numbers),
        "minimum_characters": minimum_characters,
        "canonical": canonical,
        "pre_title_drafts": drafts,
    }


def overwrite_incidents(
    work_dir: str | Path, log_path: str | Path | None = None
) -> list[dict]:
    root = Path(work_dir).resolve()
    log = Path(log_path) if log_path else root.parent / "autonomous.log"
    if not log.is_file():
        return []
    grouped: dict[int, dict[tuple[str, str, int], dict]] = {}
    for line in log.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = _WRITE_LOG.search(line)
        if not match or match.group("work") != root.name:
            continue
        number = int(match.group("number"))
        event = {
            "timestamp": match.group("timestamp"),
            "title": match.group("title") or "",
            "characters": int(match.group("characters")),
        }
        key = (event["timestamp"], event["title"], event["characters"])
        grouped.setdefault(number, {})[key] = event
    incidents = []
    for number, unique in sorted(grouped.items()):
        events = sorted(unique.values(), key=lambda item: item["timestamp"])
        if len(events) < 2:
            continue
        current = root / "chapters" / f"第{number}章.md"
        incidents.append(
            {
                "chapter": number,
                "writes": events,
                "recoverability": "metadata-only; overwritten bodies were not retained",
                "current_sha256": _sha256(current) if current.is_file() else None,
            }
        )
    return incidents


def _snapshot_work(root: Path, audit: dict) -> Path:
    snapshots = root / "protection" / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    final = snapshots / stamp
    suffix = 1
    while final.exists():
        final = snapshots / f"{stamp}_{suffix:02d}"
        suffix += 1
    pending = snapshots / f".{final.name}.pending"
    pending.mkdir()
    try:
        for name in ("chapters", "legacy", "architecture"):
            source = root / name
            if source.is_dir():
                shutil.copytree(source, pending / name)
        for name in ("WORK_INDEX.md", "chapter_manifest.json"):
            source = root / name
            if source.is_file():
                shutil.copy2(source, pending / name)
        atomic_write_json(pending / "snapshot_manifest.json", audit)
        pending.rename(final)
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return final


def _work_index(audit: dict, incidents: list[dict], snapshot: Path) -> str:
    start, end = audit["canonical_range"] if audit["canonical_range"] else (0, 0)
    shorts = "、".join(f"第{n}章" for n in audit["short_canonical_numbers"]) or "无"
    overlaps = "、".join(str(n) for n in audit["overlapping_draft_numbers"]) or "无"
    return f"""# 《{audit['work']}》作品索引

## 正式主线

- 唯一正式章节目录：`chapters/`
- 当前范围：第 {start} 章至第 {end} 章，共 {audit['canonical_count']} 章
- 缺失章号：{audit['missing_canonical_numbers'] or '无'}
- 低于 {audit['minimum_characters']} 字的章节：{shorts}
- 正文未完整收句：{audit['incomplete_ending_numbers'] or '无'}
- 标题章号异常：{audit['heading_issue_numbers'] or '无'}
- 后续续写必须从第 {end + 1} 章开始，不得复用已有章号

## 定名前异稿

`legacy/未命名作品/chapters/` 中的 {audit['pre_title_draft_count']} 章是定名前的早期异稿。
它们与正式主线的章号重叠（{overlaps}），且剧情时间线并非接续关系，因此不并入正式编号，
但作为创作源稿永久保留并纳入哈希清单。

## 覆盖记录与保护

- 写作日志识别出 {len(incidents)} 个发生过多次写入的章号；已保存的日志只能证明标题、时间和字数，不能还原被覆盖正文。
- 本次保护快照：`{snapshot.relative_to(snapshot.parents[2]).as_posix()}`
- 完整章节哈希：`chapter_manifest.json`
- 历史覆盖证据：`protection/overwrite-incidents.json`
- 章节修复前版本：`protection/revisions/`

维护工具：`python -m memagent.work_admin audit --work "{snapshot.parents[2]}"`
"""


def protect_work(
    work_dir: str | Path,
    minimum_characters: int = 900,
    log_path: str | Path | None = None,
) -> dict:
    root = Path(work_dir).resolve()
    with FileLock(root / ".writer.lock", timeout=10.0):
        audit = audit_work(root, minimum_characters)
        incidents = overwrite_incidents(root, log_path)
        snapshot = _snapshot_work(root, audit)
        atomic_write_json(
            root / "protection" / "overwrite-incidents.json",
            {"work": root.name, "incidents": incidents},
            backup=True,
        )
        atomic_write_json(root / "chapter_manifest.json", audit, backup=True)
        atomic_write_text(root / "WORK_INDEX.md", _work_index(audit, incidents, snapshot))
    return {"audit": audit, "incidents": incidents, "snapshot": str(snapshot)}


def _normalise_repair(work: str, number: int, generated: str) -> str:
    text = generated.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()
    body = _body(text)
    return f"# 《{work}》第{number}章\n\n{body}\n"


def generate_repair_candidate(
    work_dir: str | Path,
    number: int,
    target_characters: int = 1200,
) -> Path:
    from .responder import LLMResponder

    root = Path(work_dir).resolve()
    current_path = root / "chapters" / f"第{number}章.md"
    previous_path = root / "chapters" / f"第{number - 1}章.md"
    next_path = root / "chapters" / f"第{number + 1}章.md"
    if not current_path.is_file() or not previous_path.is_file() or not next_path.is_file():
        raise FileNotFoundError(f"repair requires previous/current/next chapters: {number}")
    previous = previous_path.read_text(encoding="utf-8")
    current = current_path.read_text(encoding="utf-8")
    following = next_path.read_text(encoding="utf-8")
    prompt = f"""修复《{root.name}》第{number}章的截断稿。请重写这一章的正文。

硬性要求：
1. 只输出正文，不要标题、说明、提纲或 Markdown 围栏。
2. 正文约 {target_characters} 个中文字符，至少 900 个非空白字符。
3. 保留截断稿已经出现的事件、意象、人物动作和设定，不推翻既有事实。
4. 从上一章结尾自然承接，并在结尾精确衔接下一章开头；不要照抄下一章整段。
5. 不总结、不跳时、不引入改变主线的新角色或新规则。
6. 保持原作冷峻、克制、具象的东方玄幻笔调，避免解释设定和堆砌形容词。

上一章（仅供衔接）：
{_body(previous)}

当前截断稿（必须保留其事实）：
{_body(current)}

下一章（仅供衔接）：
{_body(following)}
"""
    responder = LLMResponder(
        persona="novelist", thinking="auto", max_tokens=4096, timeout=180.0
    )
    generated = responder.respond(prompt, memories=None, timeout=180.0)
    repaired = _normalise_repair(root.name, number, generated)
    candidates = root / "protection" / "candidates"
    candidate = candidates / f"第{number}章_{_timestamp()}.md"
    atomic_write_text(candidate, repaired, overwrite=False)
    return candidate


def promote_repair(
    work_dir: str | Path,
    number: int,
    candidate: str | Path,
    minimum_characters: int = 900,
) -> dict:
    root = Path(work_dir).resolve()
    target = root / "chapters" / f"第{number}章.md"
    candidate_path = Path(candidate).resolve()
    text = candidate_path.read_text(encoding="utf-8")
    if _body_characters(text) < minimum_characters:
        raise ValueError(
            f"candidate chapter {number} has only {_body_characters(text)} characters"
        )
    expected = f"《{root.name}》第{number}章"
    if expected not in _heading(text):
        raise ValueError(f"candidate heading does not match {expected}")
    with FileLock(root / ".writer.lock", timeout=10.0):
        old_hash = _sha256(target)
        revisions = root / "protection" / "revisions" / f"第{number}章"
        revision = revisions / f"{_timestamp()}_{old_hash[:12]}.md"
        revision.parent.mkdir(parents=True, exist_ok=True)
        if not revision.exists():
            shutil.copy2(target, revision)
        atomic_write_text(target, text, backup=True)
        audit = audit_work(root, minimum_characters=minimum_characters)
        atomic_write_json(root / "chapter_manifest.json", audit, backup=True)
    return {
        "chapter": number,
        "characters": _body_characters(text),
        "old_sha256": old_hash,
        "new_sha256": _sha256(target),
        "revision": str(revision),
    }


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="小说章节审计、保护与截断修复")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="只读审计章节")
    audit_parser.add_argument("--work", required=True)
    audit_parser.add_argument("--minimum", type=int, default=900)
    protect_parser = sub.add_parser("protect", help="建立快照、清单与覆盖记录")
    protect_parser.add_argument("--work", required=True)
    protect_parser.add_argument("--minimum", type=int, default=900)
    protect_parser.add_argument("--log")
    repair_parser = sub.add_parser("repair", help="生成并安全应用一个短章修复稿")
    repair_parser.add_argument("--work", required=True)
    repair_parser.add_argument("--chapter", type=int, required=True)
    repair_parser.add_argument("--target", type=int, default=1200)
    repair_parser.add_argument("--minimum", type=int, default=900)
    promote_parser = sub.add_parser("promote", help="归档旧章并应用已审核候选稿")
    promote_parser.add_argument("--work", required=True)
    promote_parser.add_argument("--chapter", type=int, required=True)
    promote_parser.add_argument("--candidate", required=True)
    promote_parser.add_argument("--minimum", type=int, default=900)
    args = parser.parse_args(argv)

    if args.command == "audit":
        print(json.dumps(audit_work(args.work, args.minimum), ensure_ascii=False, indent=2))
        return 0
    if args.command == "protect":
        report = protect_work(args.work, args.minimum, args.log)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "promote":
        report = promote_repair(args.work, args.chapter, args.candidate, args.minimum)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    candidate = generate_repair_candidate(args.work, args.chapter, args.target)
    report = promote_repair(args.work, args.chapter, candidate, args.minimum)
    report["candidate"] = str(candidate)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
