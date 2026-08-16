"""setup_agents.ps1 端到端验证：一键导出 AGENTS.md + 挂 post-commit 钩子 + 提交自动 --sync。

在沙盒 git 仓库中运行真实 PowerShell 脚本（跳过守卫：无 powershell / git 时跳过）。
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("powershell") is None or shutil.which("git") is None,
    reason="需要 powershell 与 git",
)


def _copy_project(sandbox: Path) -> None:
    """把运行所需文件复制进沙盒：memagent 包 + session_memory.py + 脚本。"""
    shutil.copytree(ROOT / "memagent", sandbox / "memagent")
    shutil.copy(ROOT / "session_memory.py", sandbox / "session_memory.py")
    shutil.copy(ROOT / "setup_agents.ps1", sandbox / "setup_agents.ps1")


def _run(cmd, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=cwd, timeout=timeout,
    )


def _git_init(sandbox: Path) -> None:
    _run(["git", "init", "-b", "main"], sandbox)
    _run(["git", "config", "user.email", "t@t.t"], sandbox)
    _run(["git", "config", "user.name", "test"], sandbox)


def test_setup_exports_and_hook_auto_syncs(tmp_path):
    """完整链路：运行脚本 → AGENTS.md 生成 + 钩子安装 → 首次提交 → 钩子自动 --sync
    （决策沉淀进 memories_session.json 并刷新 AGENTS.md）。"""
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _copy_project(sandbox)
    _git_init(sandbox)
    (sandbox / "hello.txt").write_text("hi", encoding="utf-8")
    _run(["git", "add", "."], sandbox)

    proc = _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", "setup_agents.ps1"], sandbox)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AGENTS.md 已生成" in proc.stdout
    assert (sandbox / "AGENTS.md").exists()
    assert (sandbox / "CLAUDE.md").exists()  # 双格式同步
    assert (sandbox / "AGENTS.md").read_text(encoding="utf-8") == (sandbox / "CLAUDE.md").read_text(encoding="utf-8")

    hook = sandbox / ".git" / "hooks" / "post-commit"
    assert hook.exists()
    content = hook.read_text(encoding="ascii")
    assert "memagent post-commit" in content
    assert "session_memory.py --sync" in content

    # 提交 → post-commit 钩子触发 --sync
    commit = _run(["git", "commit", "-m", "first commit"], sandbox)
    assert commit.returncode == 0, commit.stdout + commit.stderr
    assert "memagent: decision synced" in commit.stdout + commit.stderr
    assert (sandbox / "memories_session.json").exists()      # --sync 落盘
    md = (sandbox / "AGENTS.md").read_text(encoding="utf-8")
    assert "开发决策：first commit" in md                    # 新提交沉淀进 AGENTS.md

    # 重复提交不堆积（去重）：提交数 +1，但决策条数不重复翻倍
    (sandbox / "hello.txt").write_text("hi2", encoding="utf-8")
    _run(["git", "commit", "-am", "first commit"], sandbox)  # 同主题 → 去重强化
    md2 = (sandbox / "AGENTS.md").read_text(encoding="utf-8")
    assert md2.count("开发决策：first commit") == 1          # 不重复


def test_setup_non_git_repo_warns_but_exports(tmp_path):
    """非 git 目录：仍导出 AGENTS.md，钩子跳过并提示。"""
    sandbox = tmp_path / "nongit"
    sandbox.mkdir()
    _copy_project(sandbox)
    proc = _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", "setup_agents.ps1"], sandbox)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AGENTS.md 已生成" in proc.stdout
    assert "不是 git 仓库" in proc.stdout
    assert not (sandbox / ".git").exists()
    assert (sandbox / "AGENTS.md").exists()
    assert (sandbox / "CLAUDE.md").exists()  # 非 git 目录也双写
