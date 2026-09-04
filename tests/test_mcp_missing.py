"""验证 mcp_server.py 缺 mcp 包时友好退出（不抛 traceback）。

不依赖真实 uninstall mcp——通过 sys.path 注入一个 shadow 目录让 import mcp 失败，
然后子进程跑 mcp_server，断言 exit_code=2 + stderr 含可操作的 pip install 提示。
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_mcp_server_graceful_exit_when_mcp_missing():
    """无 mcp 包时 mcp_server 应友好退出而非抛 traceback。"""
    # shadow dir：放一个空的 mcp 包，让 `import mcp.server.stdio` 失败
    with tempfile.TemporaryDirectory() as shadow:
        (Path(shadow) / "mcp").mkdir()
        (Path(shadow) / "mcp" / "__init__.py").write_text("")
        env = dict(os.environ)
        # shadow 路径放最前，优先于 site-packages
        env["PYTHONPATH"] = shadow + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "memagent.mcp_server"],
            capture_output=True, text=True, encoding="utf-8",
            timeout=10, env=env, cwd=str(ROOT),
        )
        # 关键断言：exit_code=2 是自定义友好退出，不是 1（traceback）
        assert proc.returncode == 2, (
            f"期望 exit_code=2（友好退出），实际 {proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}"
        )
        # stderr 应含可操作的 pip install 命令
        assert "pip install" in proc.stderr, (
            f"stderr 应含可操作的 pip install 命令，实际: {proc.stderr[:300]}"
        )
        assert "mcp" in proc.stderr.lower(), (
            f"stderr 应提示 mcp 相关，实际: {proc.stderr[:300]}"
        )
        # stdout 应为空（友好提示走 stderr）
        assert proc.stdout.strip() == "", (
            f"stdout 应为空，实际: {proc.stdout[:200]!r}"
        )


if __name__ == "__main__":
    test_mcp_server_graceful_exit_when_mcp_missing()
    print("✓ Bug 1 修复验证通过：mcp_server 无 mcp 包时友好退出")
