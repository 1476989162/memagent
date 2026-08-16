"""Novel work auditing and preservation tests."""

import json
from pathlib import Path

from memagent.work_admin import audit_work, overwrite_incidents, protect_work, promote_repair


def _chapter(root: Path, number: int, body: str, *, legacy: bool = False) -> Path:
    if legacy:
        path = root / "legacy" / "未命名作品" / "chapters" / f"第{number}章.md"
        work = "未命名作品"
    else:
        path = root / "chapters" / f"第{number}章.md"
        work = root.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# 《{work}》第{number}章\n\n{body}\n", encoding="utf-8")
    return path


def test_audit_separates_canonical_and_pre_title_drafts(tmp_path):
    work = tmp_path / "新书"
    _chapter(work, 1, "甲" * 700 + "。")
    _chapter(work, 2, "乙" * 100)
    _chapter(work, 1, "旧" * 800, legacy=True)
    report = audit_work(work, minimum_characters=600)
    assert report["canonical_count"] == 2
    assert report["short_canonical_numbers"] == [2]
    assert report["incomplete_ending_numbers"] == [2]
    assert report["overlapping_draft_numbers"] == [1]


def test_protect_creates_snapshot_manifest_and_incident_evidence(tmp_path):
    work = tmp_path / "新书"
    _chapter(work, 1, "甲" * 700)
    log = tmp_path / "autonomous.log"
    log.write_text(
        "[2026-01-01 00:00:00] 写作: 《新书》第 1 章 · 「初稿」 · 900 字 → x\n"
        "[2026-01-01 00:01:00] 写作: 《新书》第 1 章 · 「覆稿」 · 700 字 → x\n",
        encoding="utf-8",
    )
    report = protect_work(work, log_path=log)
    snapshot = Path(report["snapshot"])
    assert (snapshot / "chapters" / "第1章.md").is_file()
    assert (work / "chapter_manifest.json").is_file()
    assert "定名前异稿" in (work / "WORK_INDEX.md").read_text(encoding="utf-8")
    incidents = overwrite_incidents(work, log)
    assert incidents[0]["chapter"] == 1
    assert len(incidents[0]["writes"]) == 2


def test_promote_repair_archives_original_and_refreshes_manifest(tmp_path):
    work = tmp_path / "新书"
    original = _chapter(work, 1, "短" * 100)
    candidate = tmp_path / "candidate.md"
    candidate.write_text("# 《新书》第1章\n\n" + "长" * 950 + "\n", encoding="utf-8")
    result = promote_repair(work, 1, candidate)
    assert result["characters"] == 950
    assert Path(result["revision"]).read_text(encoding="utf-8") == original.with_suffix(
        ".md.bak"
    ).read_text(encoding="utf-8")
    manifest = json.loads((work / "chapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["short_canonical_numbers"] == []
