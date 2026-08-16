"""控制台兼容：Windows GBK 终端下强制 UTF-8 输出，避免 Unicode 崩溃。"""

from __future__ import annotations

import sys


def enable_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
