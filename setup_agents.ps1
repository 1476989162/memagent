# setup_agents.ps1 — memagent 一键接入 Codex / Claude Code / Hermes
#
# 功能：
#   ① 导出 AGENTS.md（决策记忆库，Codex 等开工自动加载）
#   ② 安装 git post-commit 钩子 → 每次提交自动运行 session_memory.py --sync
#      （git log 提炼新提交 → 沉淀决策 → 刷新 AGENTS.md，去重设计可重复触发）
#   ③ 打印开工/收工说明
#
# 用法：在项目根目录（含 session_memory.py / memagent/ 处）运行
#     powershell -ExecutionPolicy Bypass -File setup_agents.ps1
# 重复运行安全：重新导出、钩子已存在时先备份再覆盖。

$ErrorActionPreference = "Stop"
# 让管道/重定向输出统一 UTF-8（中文 Windows 控制台默认 GBK，被外部读取时可能乱码）
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Write-Step([int]$n, [string]$msg) {
    Write-Host ""
    Write-Host "[$n/3] $msg" -ForegroundColor Cyan
}

# ---------- 0) Python 解释器探测 ----------
$py = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py -3" }
    else { Write-Host "未找到 Python（python / py -3），请先安装。" -ForegroundColor Red; exit 1 }
}
& $py -c "import memagent, session_memory" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "无法导入 memagent / session_memory —— 请在项目根目录（含 memagent/ 与 session_memory.py）运行本脚本。" -ForegroundColor Red
    exit 1
}
Write-Host "Python: $py"

# ---------- 1) 一键导出 AGENTS.md + CLAUDE.md（双格式，内容同步） ----------
Write-Step 1 "导出决策记忆 → AGENTS.md + CLAUDE.md（双格式同步）"
& $py session_memory.py --export-agents-md
if ($LASTEXITCODE -ne 0) { Write-Host "导出失败（退出码 $LASTEXITCODE）。" -ForegroundColor Red; exit 1 }
if (-not (Test-Path "AGENTS.md") -or -not (Test-Path "CLAUDE.md")) {
    Write-Host "AGENTS.md / CLAUDE.md 未生成。" -ForegroundColor Red; exit 1
}
Write-Host "AGENTS.md 已生成（+ CLAUDE.md，内容同步：Codex 读前者 / Claude Code 读后者）。" -ForegroundColor Green

# ---------- 2) 安装 git post-commit 钩子 ----------
Write-Step 2 "安装 git post-commit 钩子（提交后自动 --sync）"
# PS 5.1 陷阱：原生命令的 stderr 在 $ErrorActionPreference=Stop 下会变成终止错误
# （NativeCommandError）——探测 git 仓库前临时放宽，交给 $LASTEXITCODE 判断。
$gitDirRaw = ""
$oldEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$gitOut = & git rev-parse --git-dir 2>&1
$ErrorActionPreference = $oldEAP
if ($LASTEXITCODE -eq 0) { $gitDirRaw = ($gitOut | Out-String).Trim() }
if ($LASTEXITCODE -ne 0 -or -not $gitDirRaw) {
    Write-Host "当前目录不是 git 仓库 —— 跳过钩子安装（AGENTS.md 已导出；进入 git 仓库后重跑本脚本即可补装）。" -ForegroundColor Yellow
} else {
    $gitDir = $gitDirRaw
    if (-not [System.IO.Path]::IsPathRooted($gitDir)) { $gitDir = Join-Path $root $gitDir }
    $hookPath = Join-Path $gitDir "hooks\post-commit"
    if (Test-Path $hookPath) {
        $bak = "$hookPath.memagent.bak"
        Copy-Item $hookPath $bak -Force
        Write-Host "已备份原钩子 → $bak" -ForegroundColor Yellow
    }
    # 单引号 here-string：不插值；用占位符替换解释器命令。
    # 钩子注释只用 ASCII：Set-Content -Encoding ASCII 会把中文写成 ?，且 sh 不认 BOM。
    $hook = @'
#!/bin/sh
# memagent post-commit: auto sync decisions & refresh AGENTS.md (installed by setup_agents.ps1)
cd "$(git rev-parse --show-toplevel)" || exit 0
export PYTHONIOENCODING=utf-8
PY=__PYTHON__
"$PY" session_memory.py --sync >/dev/null 2>&1 \
  && echo "memagent: decision synced & AGENTS.md refreshed" \
  || echo "memagent: sync skipped (no new commits or error)"
exit 0
'@
    $hook = $hook.Replace("__PYTHON__", $py)
    Set-Content -Path $hookPath -Value $hook -Encoding ASCII -NoNewline
    Write-Host "钩子已安装: $hookPath" -ForegroundColor Green
}

# ---------- 3) 开工/收工说明 ----------
Write-Step 3 "开工说明"
Write-Host @"
下一次会话如何使用：

  开工前（可选）:
    python session_memory.py --start --topic "主题"      # 打印相关决策，粘贴给 agent
    python session_memory.py --inject-agents-md          # 维护 AGENTS.md 顶部动态区块

  收工时:
    python session_memory.py --record --note "本次关键决策"   # 手动沉淀
    git commit ...   # post-commit 钩子自动运行 --sync：提炼 + 沉淀 + 刷新 AGENTS.md

  验证:
    python session_memory.py --eval-agents-md AGENTS.md   # 8 个主题问题的加载覆盖检查
    python session_memory.py --sync --eval                # 收工一步验证：加载评估 + 唤醒链路连续性 + 唤醒信号统计 + τ 学习器健康检查

  Codex / Claude Code / Hermes 开工时自动加载 AGENTS.md / CLAUDE.md，即拥有跨会话决策记忆。
"@
Write-Host ""
Write-Host "完成 ✔" -ForegroundColor Green
