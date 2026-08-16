"""Validated, atomic backup and restore operations for memory stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .cli import enable_utf8
from .io_utils import FileLock, atomic_write_bytes, atomic_write_json, atomic_write_text
from .memory import Memory


class BackupError(RuntimeError):
    """A persistence backup cannot be validated or restored."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def verify_store_file(path: str | Path) -> dict:
    source = Path(path).resolve()
    if not source.is_file():
        raise BackupError(f"persistence file not found: {source}")
    payload = source.read_bytes()
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError(f"invalid persistence JSON: {source}") from exc
    records = data.get("memories", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise BackupError(f"invalid persistence schema: {source}")
    try:
        for record in records:
            if not isinstance(record, dict) or "id" not in record or "content" not in record:
                raise ValueError("missing memory identity or content")
            Memory.from_dict(record)
    except (TypeError, ValueError, KeyError) as exc:
        raise BackupError(f"invalid memory record in {source}") from exc
    return {
        "path": str(source),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "memories": len(records),
        "format": "object" if isinstance(data, dict) else "legacy-array",
    }


def create_backup(
    persist_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    source = Path(persist_path).resolve()
    output = Path(output_dir).resolve() if output_dir else source.parent / "backups"
    with FileLock(str(source) + ".lock", timeout=30.0):
        info = verify_store_file(source)
        payload = source.read_text(encoding="utf-8")
        output.mkdir(parents=True, exist_ok=True)
        backup = output / f"{source.stem}-{_timestamp()}-{info['sha256'][:12]}{source.suffix}"
        if not backup.exists():
            atomic_write_text(backup, payload, overwrite=False)
        manifest = {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "source": str(source),
            "backup": str(backup),
            **{key: value for key, value in info.items() if key != "path"},
        }
        atomic_write_json(backup.with_suffix(backup.suffix + ".manifest.json"), manifest)
    return manifest


def restore_backup(backup_path: str | Path, persist_path: str | Path) -> dict:
    backup = Path(backup_path).resolve()
    target = Path(persist_path).resolve()
    backup_info = verify_store_file(backup)
    payload_bytes = backup.read_bytes()
    if _sha256_bytes(payload_bytes) != backup_info["sha256"]:
        raise BackupError("backup changed during verification")
    payload = payload_bytes.decode("utf-8")
    previous = None
    with FileLock(str(target) + ".lock", timeout=30.0):
        if target.is_file():
            raw = target.read_bytes()
            digest = _sha256_bytes(raw)
            previous_dir = target.parent / "restore-backups"
            previous_dir.mkdir(parents=True, exist_ok=True)
            previous = previous_dir / f"{target.stem}-pre-restore-{_timestamp()}-{digest[:12]}{target.suffix}"
            if not previous.exists():
                atomic_write_bytes(previous, raw, overwrite=False)
        atomic_write_text(target, payload)
        restored = verify_store_file(target)
        if restored["sha256"] != backup_info["sha256"]:
            raise BackupError("restored persistence checksum does not match backup")
    return {
        "restored": str(target),
        "from": str(backup),
        "sha256": restored["sha256"],
        "memories": restored["memories"],
        "pre_restore_backup": str(previous) if previous else None,
    }


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="MemAgent persistence backup manager")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--file", required=True)
    create = sub.add_parser("create")
    create.add_argument("--persist", required=True)
    create.add_argument("--output")
    restore = sub.add_parser("restore")
    restore.add_argument("--from", dest="source", required=True)
    restore.add_argument("--persist", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_store_file(args.file)
    elif args.command == "create":
        result = create_backup(args.persist, args.output)
    else:
        result = restore_backup(args.source, args.persist)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
