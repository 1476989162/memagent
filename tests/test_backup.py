"""Persistence backup and atomic restore tests."""

import json
from pathlib import Path

import pytest

from memagent.backup import BackupError, create_backup, restore_backup, verify_store_file
from memagent.memory import MemoryStore


def test_backup_restore_roundtrip_preserves_pre_restore_version(tmp_path):
    target = tmp_path / "memory.json"
    store = MemoryStore(path=str(target))
    store.add("first")
    store.save()
    backup = create_backup(target, tmp_path / "backups")

    changed = MemoryStore(path=str(target))
    changed.add("second")
    changed.save()
    changed_bytes = target.read_bytes()

    result = restore_backup(backup["backup"], target)
    restored = MemoryStore(path=str(target))
    assert [memory.content for memory in restored.all()] == ["first"]
    assert result["memories"] == 1
    assert result["sha256"] == backup["sha256"]
    assert Path(result["pre_restore_backup"]).read_bytes() == changed_bytes


def test_verify_store_rejects_corrupt_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_bytes(b"{not-json")
    with pytest.raises(BackupError):
        verify_store_file(path)


def test_verify_store_rejects_invalid_record(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"memories": [{"id": "x"}]}), encoding="utf-8")
    with pytest.raises(BackupError):
        verify_store_file(path)
