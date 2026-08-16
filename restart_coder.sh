#!/usr/bin/env bash
# Foxtable 后台 coder 重启脚本（防日志清空事故：2026-08-15 用 `>` 重启丢过轮1-165 日志）
# 用法：bash restart_coder.sh
# 要点：
#   1) 启动前把现有日志备份成 .bak（轮转保留一份）
#   2) 日志用 `>>` 追加写，绝不覆盖
#   3) 若检测到已有 autonomous_coder 进程则提示先停
set -u

LOG=works/foxtable_coder.log

# 检查是否已有实例在跑
if powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -match 'autonomous_coder'}" 2>/dev/null | grep -q "ProcessId"; then
  echo "!! 已有 autonomous_coder 进程在运行，先停掉再重启（PID 见下）："
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -match 'autonomous_coder'} | Select-Object ProcessId,CommandLine | Format-List"
  exit 2
fi

# 备份现有日志（只保留一份 .bak）
if [ -f "$LOG" ] && [ -s "$LOG" ]; then
  cp "$LOG" "$LOG.bak"
  echo "已备份: $LOG.bak ($(wc -c < "$LOG.bak") 字节)"
fi

# 启动（>> 追加写，保留历史日志）
# 休息时长 2026-08-15 实验定稿：60-120s 与 300-900s 均分/抖动率无差异（见
# docs/rest_experiment_20260815.md），但迭代速度快 ~3.8 倍——采用 60-120s。
# 回放窗口 2026-08-15 实验定稿：默认 1 秒窗口只回放本轮新沉淀，旧教训永不
# 再激活（轮123→126 复犯温床）；--replay-rounds 10 让最近 10 轮周期性再激活
# （见 docs/replay_window_experiment_20260815.md）。
nohup python autonomous_coder.py --cycles 0 --min-interval 60 --max-interval 120 --replay-rounds 10 >> "$LOG" 2>&1 &
echo "已启动 PID $!，日志追加写: $LOG"
sleep 2
tail -2 "$LOG"
