"""Installed ``memagent`` and ``python -m memagent`` entry point."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from . import __version__
from .agent import AgentConfig, MemoryAgent
from .cli import enable_utf8
from .memory import MemoryStore, StoreCorruptionError


def _check(path: str) -> int:
    target = Path(path).resolve()
    try:
        store = MemoryStore(path=str(target))
    except StoreCorruptionError as exc:
        print(f"[FAIL] persistence: {exc}")
        return 1
    print(f"[OK] memagent {__version__} / Python {platform.python_version()}")
    print(f"[OK] persistence: {target} ({len(store)} memories)")
    print(f"[OK] writable parent: {os.access(target.parent, os.W_OK)}")
    backup = target.with_suffix(target.suffix + ".bak")
    print(f"[INFO] backup: {backup if backup.exists() else 'created after first update'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="Local-first layered memory agent")
    parser.add_argument("--persist", default="memories.json", help="memory JSON path")
    parser.add_argument("--persona", default=os.environ.get("OPENAI_PERSONA") or None)
    parser.add_argument("--check", action="store_true", help="validate runtime and persistence")
    parser.add_argument("--migrate-work", nargs=2, metavar=("OLD", "NEW"),
                        help="move pre-title work safely without overwriting chapters")
    parser.add_argument("--works-dir", default="works", help="work root for --migrate-work")
    parser.add_argument("--version", action="version", version=f"memagent {__version__}")
    args = parser.parse_args(argv)
    if args.check:
        return _check(args.persist)
    if args.migrate_work:
        from .architecture import migrate_legacy_work

        report = migrate_legacy_work(
            Path(args.works_dir).resolve(), args.migrate_work[0], args.migrate_work[1]
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0
    MemoryAgent(
        persist_path=args.persist,
        persona=args.persona,
        cfg=AgentConfig(evolve_on_sleep=bool(args.persona)),
    ).cli_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
