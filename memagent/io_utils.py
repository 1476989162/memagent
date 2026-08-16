"""Reliable local file operations used by persistence and autonomous jobs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


class LockTimeoutError(TimeoutError):
    """Raised when a process cannot acquire a file lock within the deadline."""


class FileLock:
    """Small cross-platform exclusive lock backed by a one-byte lock file."""

    def __init__(self, path: str | Path, timeout: float = 10.0, poll: float = 0.05):
        self.path = Path(path)
        self.timeout = max(0.0, timeout)
        self.poll = max(0.01, poll)
        self._file = None

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+b")
        self._file.seek(0, os.SEEK_END)
        if self._file.tell() == 0:
            self._file.write(b"0")
            self._file.flush()

        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._lock_once()
                return self
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self._file.close()
                    self._file = None
                    raise LockTimeoutError(f"timed out acquiring lock: {self.path}")
                time.sleep(self.poll)

    def _lock_once(self) -> None:
        assert self._file is not None
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    backup: bool = False,
    overwrite: bool = True,
) -> Path:
    """Flush a same-directory temporary file, then publish it atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        if backup and target.is_file():
            backup_path = target.with_suffix(target.suffix + ".bak")
            backup_temp = backup_path.with_suffix(backup_path.suffix + ".tmp")
            shutil.copy2(target, backup_temp)
            os.replace(backup_temp, backup_path)

        if overwrite:
            os.replace(temp, target)
        else:
            try:
                os.link(temp, target)
            except FileExistsError:
                raise
            finally:
                temp.unlink(missing_ok=True)
        return target
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, data: Any, *, backup: bool = False) -> Path:
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    return atomic_write_text(path, payload, backup=backup)


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically publish exact bytes without text decoding or newline changes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp, target)
        else:
            try:
                os.link(temp, target)
            except FileExistsError:
                raise
            finally:
                temp.unlink(missing_ok=True)
        return target
    finally:
        temp.unlink(missing_ok=True)
