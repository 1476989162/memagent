"""多视图记忆仪表盘：单文件 HTML（内联 SVG + 原生 JS，零依赖，离线可用）。

五个联动视图：
- 主图：记忆强度曲线（线宽 ∝ 重要性，环标 = 检索事件，◇ = 唤醒点——
  红条 = 实测偏差 dev、青条 = 类型预期偏差 expected，双条都结束于实测点
  高度，红条长于青条即"唤醒比类型预期剧烈"），滚轮缩放/拖动平移；
  **点击 ◇ 或任一双条展开该唤醒事件**：更长的 dev/expected 双条（端点带
  数值标签）+ 信号方向箭头（红↓τ应下调/青↑应上调/灰✓已校准）+ 悬浮
  callout（比例条、比值与方向解释），并与类型面板联动（该记忆曲线高亮 +
  面板内唤醒菱形可反向点击展开同一事件）；
- 记忆地图：气泡图 x=检索次数、y=重要性、气泡大小=当前强度；
- 层级×类型分布条（点击层级名切换显示）；
- 最强记忆 Top5 列表；
- 类型对比视图：技能/语义/情景三列子图共享横轴 + 典型遗忘参考曲线，
  支持自定义时间窗（过去 N 天 / 未来 M 天），JS 按窗口自适应重绘。
- 记忆类型画像面板：配置列之外，**τ 两路信号健康检查合表**（每类型一行：
  干净段方向 / 唤醒方向箭头 + 一致性徽章 ✔一致 ✘冲突 △单源 —无信号，
  tooltip 带 n 与建议 τ↓/τ↑/需检查/需补观测/已校准——单一事实源
  agent.tau_learner_health，与 --export-signals 的 CSV 合表同源），再加
  **行动徽章列**（suggest_adjust：τ↓红 / τ↑青 / ⚠需检查橙 / 其余灰，
  tooltip 带语义与置信度）——点击徽章或下方信号漂移行同类型的条目，双向
  高亮联动（再点 / Esc 取消），行动与漂移趋势一眼对照；
- **一致性徽章 → 主图类型唤醒联动**：点击 `✘ 冲突`（及任一一致性徽章）→
  主图高亮该类型**全部唤醒点**（非该类型压暗），悬浮 callout 列出该类型每条
  唤醒事件（ratio + 方向），**与干净段方向相反的事件标橙并注「← 与干净段相反」**
  ——直接定位是哪几起事件造成两路冲突；事件行可点击展开单条双条，再点 / Esc 取消。
- **冲突类型 ⚠ 行 + 两路证据展开**：`health.warnings` 非空的类型，画像行左侧
  描橙边 + 底色高亮 + ⚠ 标记——点击 ⚠ / 整行展开隐藏的两路证据行（① 干净段
  evidence ② 唤醒 evidence → 排查建议，与告警 JSON 同源）；点击该类型一致性
  徽章联动主图时证据行同步展开，再点 / Esc 全部收起。证据行末尾列出**冲突成因
  事件明细**（记忆预览 + `[行 k]` CSV 行号（行号算法收敛于 agent.py 单一实现，
  与导出 events CSV / 终端打印同源）+ 比值 + 方向箭头），**点击任意事件定位
  主图对应唤醒点**（展开 dev vs expected 双条 + 信号方向 callout，showAwakening），
  callout 同时附**原始 CSV 行预览**（与导出 events CSV 同列：memory_id, mtype,
  ts, ts_relative_seconds, dev, expected, ratio, dt_seconds, retrievals_before，
  标注 `[行 k]`；六元组事件才有后两列）；**Shift 点击多选事件** → 聚合面板实时显示
  选中事件的方向分布、干净段方向、移除后剩余事件的中位比值与方向，判定
  「✔ 移除后两路一致——冲突消除 / ✘ 移除后仍冲突 / — 观测不足」——直接在仪表盘
  验证“去掉这批事件后两路信号是否一致”；聚合面板带**全选 / 选反向 / 清空**
  快捷按钮，方向占比用**三段色条**可视化（↑青 / ↓红 / ＝灰按占比等比例显示，
  取代纯文本百分比；图例保留计数、悬浮显示精确 `n/N (pct%)`）——**双条显示**：
  全体条 + **选中集分布条**（选中事件的方向占比，0 选中显示无数据），全体条上
  选中覆盖到的方向段叠加**蓝色描边**（inset 环 + 悬浮标注「选中 n 起」），
  选反向 / 全选 / 清空 / Shift 逐条同步变化；**每个色段可点击**（data-dir，悬浮
  提示「点击色段只圈出该方向事件」）——点击只圈出该方向的事件（清空重选，
  与「选反向」的 `dir ≠ clean` 判定互补），移除此方向后两路是否一致即刻可见——
  **选反向**
  一键圈定与干净段相反方向的事件（`dir ≠ clean`，与一致性徽章 callout 的
  clashes 标橙共用 `_evDir` 同一判定），冲突成因两步操作变一步；全选/清空/
  选反向/Shift 逐条共用同一刷新路径（`refreshConflictSel`）——warn-ev-row 高亮、
  聚合面板与 callout 的 CSV 行预览（已选/未选徽章）三处同步更新，面板内嵌  **选中集 CSV 行预览**逐条列出选中事件的 CSV 行（行号对应导出 events CSV），
  Esc 清空多选并重置面板基线；面板带**「导出聚合 / 复制」按钮**——把全部类型当前
  Shift 多选状态一键下载为 `aggregations.json`（`--aggregations-file` 直接读取，
  免手写 memory_id 列表）；**复制**按钮把同一 payload 写入剪贴板（clipboard API
  + `execCommand('copy')` 回退，覆盖无下载权限的嵌入式环境），仪表盘圈定 → CLI
  回放全程免手抄；**选反向圈定的选择集一键生成 `--exclude-events` 参数串**
  （`memory_id:序号,...`，面板实时显示 + 证据行顶部「复制 --exclude-events」按钮
  写剪贴板）——仪表盘圈定 → CLI 剔除重判 → JSON 落盘全程免手抄。
- **历史聚合结论回放**：`plot_interactive(..., aggregations=[{"mtype", "events":
  ['memory_id:序号', ...]}])`（与 `--aggregations` 同格式）→ `health.aggregations`
  嵌入仪表盘，画像面板下方新增「历史聚合结论」区——每条 verdict 徽章（✔ 冲突消除
  绿 / ✘ 仍冲突 红 / — 观测不足 灰）+ 事件子集摘要（已选方向分布 · 干净段 · 移除后
  中位与方向）+ resolved 自动附带剔除后证据包（consistency / suggest / warnings，
  与 `--export-signals` 同链路）；**点击该行在主图高亮对应事件子集**（◇/红条/青条
  精确匹配，再点 / Esc 取消），callout 逐条列出子集事件并可点开单事件双条。
任意面板点击一条记忆 → 全局高亮；Hot/Warm/Cold 按钮切换全局生效。
"""

from __future__ import annotations

import json
import time

from .agent import (
    awakening_signal_periods,
    awakening_signal_stats,
    tau_learner_health,
)
from .memory import MemType, Tier
from .profiles import type_profiles
from .visualize import (
    MTYPE_COLORS,
    TIER_COLORS,
    _awakening_events,
    _esc,
    _reference_series,
    _scaffold,
    access_events,
    default_horizon,
    floor_verification,
    forgetting_slope,
    fmt_duration,
    strength_at,
    strength_series,
    tau_summary,
)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>记忆多视图仪表盘</title>
<style>
  body { font-family: system-ui, "Microsoft YaHei", sans-serif; background:#f4f4f4; margin:0; padding:20px; color:#222; }
  .wrap { max-width: 1280px; margin: 0 auto; }
  .card { background:#fff; border-radius:10px; box-shadow:0 2px 14px rgba(0,0,0,.12); padding:14px 18px 18px; }
  .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
  .toolbar .label { font-weight:bold; font-size:13px; margin-right:2px; }
  .btn { padding:6px 13px; border:1px solid #c9c9c9; border-radius:6px; background:#fff; cursor:pointer; font-size:13px; color:#333; }
  .btn.active { background:#2f6fd6; color:#fff; border-color:#2f6fd6; }
  .btn[data-tier="hot"].active { background:#e34a2f; border-color:#e34a2f; }
  .btn[data-tier="cold"].active { background:#6b7078; border-color:#6b7078; }
  .btn.small { padding:5px 10px; }
  .stats { display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:#666; padding:8px 2px 10px; }
  .stats b { color:#222; }
  .grid { display:grid; grid-template-columns: 1fr 330px; gap:14px; align-items:start; }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  svg#chart { width:100%; height:auto; display:block; cursor:grab; user-select:none; touch-action:none; }
  svg#chart.panning { cursor:grabbing; }
  .mem-curve, .mem-trajectory, .mem-dot, .access-tick, .awake-mark, .awake-dev, .awake-exp, .awake-label { transition: opacity .15s; }
  .awake-mark, .awake-dev, .awake-exp, .awake-label { cursor: pointer; }
  .awake-mark:hover, .awake-dev:hover, .awake-exp:hover { opacity: 1 !important; }
  .awake-exp-bar, .awake-exp-label, .awake-exp-arrow { pointer-events: none; }
  #awakeCallout { position: fixed; right: 22px; top: 22px; z-index: 50; width: 300px; background:#fff; border:1px solid #d9d9d9; border-radius:10px; box-shadow:0 6px 24px rgba(0,0,0,.18); padding:12px 14px; display:none; font-size:13px; }
  #awakeCallout .act-title { font-weight:bold; margin-bottom:6px; }
  #awakeCallout .act-mem { color:#888; font-size:12px; margin-bottom:8px; }
  #awakeCallout .act-row { display:flex; align-items:center; gap:6px; margin:5px 0; font-size:12px; }
  #awakeCallout .act-bar { flex:1; height:12px; border-radius:3px; }
  #awakeCallout .act-val { width:64px; text-align:right; color:#666; font-variant-numeric: tabular-nums; }
  #awakeCallout .act-dir { margin-top:8px; padding:6px 8px; border-radius:6px; font-weight:bold; font-size:12.5px; }
  #awakeCallout .act-hint { color:#999; font-size:11.5px; margin-top:6px; line-height:1.6; }
  .panel { border:1px solid #e6e6e6; border-radius:8px; padding:10px 12px; margin-top:14px; background:#fcfcfd; }
  .ptitle { font-size:12px; font-weight:bold; color:#555; margin-bottom:8px; }
  svg#bubble { width:100%; height:auto; display:block; }
  .bubble { cursor:pointer; transition: opacity .15s; }
  .row { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:6px; cursor:pointer; font-size:13px; }
  .row:hover { background:#f0f4fb; }
  .row.sel { background:#e7effc; outline:1px solid #2f6fd6; }
  .row .dot { width:10px; height:10px; border-radius:50%; flex:none; }
  .row .txt { flex:1; color:#444; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .row .val { color:#888; font-variant-numeric: tabular-nums; }
  .tw-bar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:13px; color:#555; padding:6px 2px 10px; }
  .tw-bar input { width:64px; padding:4px 6px; border:1px solid #c9c9c9; border-radius:5px; font-size:13px; }
  .tw-note { color:#aaa; font-size:12px; }
  .ptable { width:100%; border-collapse:collapse; font-size:13px; margin-top:4px; }
  .ptable th, .ptable td { padding:5px 12px 5px 0; border-bottom:1px solid #eee; text-align:left; }
  .ptable th { color:#888; font-weight:bold; font-size:12px; }
  .ptable .pcode { color:#aaa; font-size:11px; font-weight:normal; }
  .act-badge { display:inline-block; padding:1px 8px; border-radius:10px; color:#fff; font-size:11.5px; font-weight:bold; cursor:pointer; user-select:none; transition: box-shadow .15s; }
  .act-badge:hover { box-shadow: 0 0 0 2px rgba(0,0,0,.15); }
  .act-badge.linked { box-shadow: 0 0 0 2px #1f6feb; }
  .drift-item { cursor:pointer; border-radius:5px; padding:2px 4px; transition: background .15s; }
  .drift-item.linked { background:#fff3bf; outline:1px solid #e8590c; }
  .cs-badge { cursor:pointer; }
  .ptable tr.warn-row td:first-child { border-left: 3px solid #e8590c; }
  .ptable tr.warn-row { background:#fff8f2; cursor:pointer; }
  .ptable tr.warn-row:hover { background:#ffefe0; }
  .ptable tr.warn-row td { cursor:pointer; }
  .ptable tr.warn-ev td { background:#fff8f2; border-bottom:1px dashed #f0d9c5; color:#5a4a3a; font-size:12px; line-height:1.8; }
  .ptable tr.warn-ev .wev-t { font-weight:bold; color:#c2410c; margin-bottom:4px; }
  .ptable .warn-mark { color:#e8590c; font-weight:bold; margin-right:4px; }
  .ptable .wev-evts { margin-top:6px; border-top:1px dashed #f0d9c5; padding-top:5px; }
  .ptable .warn-ev-row { display:flex; align-items:center; gap:6px; margin:3px 0; font-size:12px; cursor:pointer; padding:2px 5px; border-radius:4px; }
  .ptable .warn-ev-row:hover { background:#fff3e6; }
  .ptable .warn-ev-row.sel { background:#e7f0fd; outline:1.5px solid #1f6feb; }
  .ptable .warn-ev-row .wev-rowno { color:#a0522d; font-family:monospace; font-size:11px; }
  .ptable .wev-hint { color:#a0522d; font-size:11.5px; margin-bottom:3px; }
  .ptable .wev-aggr { margin-top:6px; padding:6px 8px; background:#fffdf8; border:1px dashed #e8c9a0; border-radius:6px; font-size:12px; line-height:1.8; color:#5a4a3a; }
  .ptable .wev-aggr .aggr-btns { float:right; display:flex; gap:4px; }
  .ptable .wev-aggr .aggr-btn { font-size:11px; padding:1px 8px; border:1px solid #d0b48a; background:#fff6ea; color:#8a5a1e; border-radius:10px; cursor:pointer; }
  .ptable .wev-aggr .aggr-btn:hover { background:#ffe8cc; }
  .ptable .wev-aggr .aggr-distbar { display:flex; height:9px; border-radius:5px; overflow:hidden; margin:4px 0 3px; background:#eee9e0; }
  .ptable .wev-aggr .aggr-seg { height:100%; }
  .ptable .wev-aggr .aggr-seg-up { background:#2a9d8f; }
  .ptable .wev-aggr .aggr-seg-down { background:#e34a2f; }
  .ptable .wev-aggr .aggr-seg-flat { background:#8a8f98; }
  .ptable .wev-aggr .aggr-seg.aggr-sel { box-shadow: inset 0 0 0 2px #1f6feb; }
  .ptable .wev-aggr .aggr-seg-btn { cursor:pointer; }
  .ptable .wev-aggr .aggr-seg-btn:hover { filter:brightness(1.12); }
  .ptable .wev-aggr .aggr-distlegend { font-size:11px; color:#7a6a58; }
  .ptable .wev-aggr .aggr-csvsel { margin-top:5px; border-top:1px dashed #e8c9a0; padding-top:4px; font-size:11.5px; }
  .ptable .wev-aggr .aggr-csvrow { margin:2px 0; color:#5a4a3a; }
  .ptable .wev-aggr .aggr-csvrow code { background:#f5efe6; padding:0 4px; border-radius:3px; font-size:11px; }
  .ptable .wev-aggr .aggr-excl { margin-top:5px; padding-top:4px; border-top:1px dashed #e8c9a0; font-size:11.5px; color:#44506a; }
  .ptable .wev-aggr .aggr-excl code { background:#eef2fb; padding:0 4px; border-radius:3px; font-size:11px; }
  .ptable .wev-copy-excl { font-size:11px; padding:1px 8px; border:1px solid #b7c4de; background:#f4f7fd; color:#2f6fd6; border-radius:10px; cursor:pointer; margin-left:6px; }
  .ptable .wev-copy-excl:hover { background:#e7f0fd; }
  #awakeCallout .csv-sel { margin-left:8px; font-weight:bold; }
  .agg-hist { margin-top:6px; padding:6px 8px; background:#fafbfd; border:1px dashed #c8d3e8; border-radius:6px; font-size:12px; line-height:1.7; color:#44506a; }
  .agg-hist-row { cursor:pointer; padding:2px 5px; border-radius:4px; }
  .agg-hist-row:hover { background:#eef2fb; }
  .agg-hist-row.linked { background:#e7f0fd; outline:1px solid #2f6fd6; }
  .agg-rec { color:#2b8a3e; font-size:11px; }
  #awakeCallout .act-rec { margin-top:5px; padding:4px 6px; background:#eefaf0; border-radius:4px; font-size:11.5px; color:#2b8a3e; }
  #awakeCallout .csv-row-box { margin-top:8px; padding:6px 8px; background:#f8fafc; border:1px solid #d0d7de; border-radius:6px; font-size:11.5px; }
  #awakeCallout .csv-row-box .csv-title { color:#57606a; margin-bottom:4px; }
  #awakeCallout .csv-row-box .csv-line { display:flex; flex-wrap:wrap; gap:4px 12px; }
  #awakeCallout .csv-row-box .csv-cell b { color:#8a8f98; font-weight:normal; font-size:10.5px; margin-right:3px; }
  #awakeCallout .csv-row-box .csv-cell code { color:#24292f; font-size:11px; background:#eef1f4; padding:0 4px; border-radius:3px; }
  svg#chart.type-linked-mode .awake-mark:not(.type-linked), svg#chart.type-linked-mode .awake-dev:not(.type-linked), svg#chart.type-linked-mode .awake-exp:not(.type-linked), svg#chart.type-linked-mode .awake-label:not(.type-linked) { opacity: .22; }
  .awake-mark.type-linked { opacity: 1 !important; filter: drop-shadow(0 0 3px rgba(227,74,47,.9)); }
  .awake-dev.type-linked, .awake-exp.type-linked { opacity: 1 !important; stroke-width: 3.6 !important; }
  .awake-label.type-linked { opacity: 1 !important; font-weight: bold; }
  .sig-drift { margin-top:6px; font-size:12px; color:#555; line-height:1.7; }
  .sig-drift .drift-item { margin-right:12px; white-space:nowrap; }
  .dist-row { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }
  .dist-label { width:52px; color:#555; cursor:pointer; }
  .dist-label:hover { text-decoration:underline; }
  .dist-bar { flex:1; height:16px; border-radius:4px; overflow:hidden; background:#f0f0f0; display:flex; }
  .dist-seg { height:100%; }
  .dist-num { width:34px; text-align:right; color:#888; font-variant-numeric: tabular-nums; }
  .mlegend { font-size:12px; color:#888; margin-top:6px; }
  .hint { font-size:12px; color:#888; margin-top:8px; line-height:1.7; }
  #details { margin-top:12px; border-top:1px solid #eee; padding-top:12px; font-size:13px; display:none; }
  #details .dtitle { font-weight:bold; margin-bottom:8px; }
  #details td { padding:3px 12px 3px 0; vertical-align:top; }
  #details .content { color:#444; max-width: 700px; white-space: pre-wrap; }
  .obs { color:#888; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
<div class="card">
  <div class="toolbar">
    <span class="label">层级显示：</span>
    <button class="btn active" data-tier="hot">Hot 工作记忆</button>
    <button class="btn active" data-tier="warm">Warm 长时记忆</button>
    <button class="btn active" data-tier="cold">Cold 深藏记忆</button>
    <span style="flex:1"></span>
    <button class="btn small" id="reset">重置视图</button>
  </div>
  <div class="stats" id="stats"></div>
  <div class="grid">
    <div>
      __SVG__
      <div class="hint">🖱 主图滚轮缩放 · 拖动平移（缩放后）· 点击任意面板中的曲线/气泡/列表行高亮单条记忆（空白/Esc 取消）· 线宽=重要性，环标=检索事件，◇=唤醒点（红条=实测 dev / 青条=类型预期 expected）</div>
    </div>
    <div>
      <div class="panel"><div class="ptitle">记忆地图 — 气泡大小 = <span id="bubbleModeLabel">触底倒计时（斜率）</span>（点击选中；点击气泡显示触底倒计时）
        <button class="btn small" data-bsize="strength">强度</button><button class="btn small active" data-bsize="slope">斜率</button></div><svg id="bubble"></svg><div class="mlegend" id="bubbleLegend"></div></div>
      <div class="panel"><div class="ptitle">最强记忆 Top5（点击选中）</div><div id="toplist"></div></div>
    </div>
  </div>
  <div class="panel">
    <div class="ptitle">层级 × 类型分布（点击层级名切换显示）</div>
    <div id="dist"></div>
    <div class="mlegend">类型颜色：<span style="color:#2f9e44">■</span> skill 技能 · <span style="color:#7048e8">■</span> semantic 语义 · <span style="color:#f08c00">■</span> episodic 情景</div>
  </div>
  <div class="panel">
    <div class="ptitle">类型对比 — 遗忘斜率（三列同宽 = 同时长，灰色虚线 = 该类型典型遗忘参考曲线；点击曲线/观测点联动全局高亮；时间窗：只看过去 N 天或预测未来 M 天）</div>
    <div class="tw-bar">
      <span class="label">时间窗：</span>
      过去 <input id="twPast" type="number" min="0" step="any" placeholder="全程"> 天
      · 未来 <input id="twFuture" type="number" min="0" step="any" placeholder="默认"> 天
      <button class="btn small" id="twApply">应用</button>
      <button class="btn small tw-preset" data-w0="-7" data-w1="0">近7天</button>
      <button class="btn small tw-preset" data-w0="-30" data-w1="0">近30天</button>
      <button class="btn small tw-preset" data-w0="0" data-w1="7">预测7天</button>
      <button class="btn small tw-preset" data-w0="0" data-w1="30">预测30天</button>
      <button class="btn small" id="twReset">全程</button>
      <span class="tw-note">单位=天（可小数）；过去段显示实际观测，未来段显示预测</span>
    </div>
    <svg id="typechart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1132 266" font-family="system-ui, sans-serif"><rect x="0" y="0" width="1132" height="266" fill="#ffffff"></rect></svg>
  </div>
  <div class="panel">
    <div class="ptitle">记忆类型画像 — 每类型完整的遗忘 / 可塑性 / 压缩配置（drift 高 = 回忆时越容易被情境改写）</div>
    <div id="profiles"></div>
  </div>
  <div id="details"></div>
</div>
</div>
<div id="awakeCallout"></div>
<script id="memdata" type="application/json">__DATA__</script>
<script>
"use strict";
const MEM = JSON.parse(document.getElementById('memdata').textContent);
const TIER = { hot: '#e34a2f', warm: '#2f6fd6', cold: '#8a8f98' };
const MTYPE = { skill: '#2f9e44', semantic: '#7048e8', episodic: '#f08c00' };
const svg = document.getElementById('chart');
const plot = document.getElementById('plot');
const VB_W = 1100, VB_H = 640, K_MIN = 1, K_MAX = 30;
let k = 1, tx = 0, ty = 0;

function apply() { plot.setAttribute('transform', 'translate(' + tx + ' ' + ty + ') scale(' + k + ')'); }
function svgPoint(e) {
  const r = svg.getBoundingClientRect();
  return { x: (e.clientX - r.left) * VB_W / r.width, y: (e.clientY - r.top) * VB_H / r.height };
}

// ---- 主图：缩放 / 平移 ----
svg.addEventListener('wheel', function (e) {
  e.preventDefault();
  const p = svgPoint(e);
  const k2 = Math.min(K_MAX, Math.max(K_MIN, k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
  const f = k2 / k;
  tx = p.x - (p.x - tx) * f;
  ty = p.y - (p.y - ty) * f;
  k = k2; apply();
}, { passive: false });

let dragging = null, moved = false;
svg.addEventListener('pointerdown', function (e) {
  if (e.target.closest('.mem-curve, .mem-trajectory, .mem-dot, .access-tick, .awake-mark, .awake-dev, .awake-exp, .awake-label')) return;
  dragging = { x: e.clientX, y: e.clientY, sx: tx, sy: ty };
  moved = false;
  svg.classList.add('panning');
  if (svg.setPointerCapture) svg.setPointerCapture(e.pointerId);
});
svg.addEventListener('pointermove', function (e) {
  if (!dragging) return;
  const dx = e.clientX - dragging.x, dy = e.clientY - dragging.y;
  if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
  const r = svg.getBoundingClientRect();
  tx = dragging.sx + dx * VB_W / r.width;
  ty = dragging.sy + dy * VB_H / r.height;
  apply();
});
svg.addEventListener('pointerup', function () { dragging = null; svg.classList.remove('panning'); });
svg.addEventListener('click', function (e) {
  if (moved) { moved = false; return; }
  const t = e.target.closest('[data-mem]');
  if (t) { select(t.dataset.mem); return; }
  if (e.target.closest('#plot')) deselect();
});
document.addEventListener('keydown', function (e) { if (e.key === 'Escape') deselect(); });

// ---- 全局选中 / 取消（联动所有视图）----
let selected = null;
let floorTimer = null;
let selecting = false;   // 防重入：select 里重绘类型面板 → renderTypeChart → select 递归
function currentWindow() {
  const p = parseFloat(document.getElementById('twPast').value);
  const f = parseFloat(document.getElementById('twFuture').value);
  let w0 = MEM.type_window.t0, w1 = MEM.type_window.t1;
  if (!isNaN(p) && p >= 0) w0 = MEM.now - p * 86400;
  if (!isNaN(f) && f >= 0) w1 = MEM.now + f * 86400;
  return [w0, w1];
}
function updateFloorCountdown(m) {
  const el = document.getElementById('floorcd');
  if (!el || !m) return;
  const ttf = m.slope.time_to_floor;
  if (ttf == null) { el.textContent = '不触底（模型预测）'; return; }
  if (ttf <= 0) { el.textContent = '已触底'; return; }
  const remain = ttf - (Date.now() / 1000 - MEM.now);
  if (remain <= 0) { el.textContent = '已触底'; return; }
  el.textContent = remain < 60 ? remain.toFixed(1) + '秒后触底' : '约 ' + fmtFloor(remain) + ' 后触底';
}
function select(id) {
  if (selecting) return;
  selecting = true;
  selected = id;
  document.querySelectorAll('.mem-curve').forEach(function (p) {
    const on = p.dataset.mem === id;
    p.style.opacity = on ? '1' : '0.07';
    p.style.strokeWidth = on ? '4' : p.dataset.baseW;
  });
  document.querySelectorAll('.mem-trajectory, .mem-dot, .access-tick, .awake-mark, .awake-dev, .awake-exp, .awake-label').forEach(function (p) {
    p.style.opacity = p.dataset.mem === id ? '1' : '0.08';
  });
  document.querySelectorAll('.bubble').forEach(function (c) {
    const on = c.dataset.mem === id;
    c.style.opacity = on ? '1' : '0.15';
    c.setAttribute('stroke', on ? '#222' : '#fff');
    c.setAttribute('stroke-width', on ? '2.5' : '1');
  });
  document.querySelectorAll('.row').forEach(function (r) {
    r.classList.toggle('sel', r.dataset.mem === id);
  });
  const selMem = MEM.memories.find(function (m) { return m.id === id; });
  showDetails(selMem);
  // 类型面板联动：先重绘类型对比视图（含选中记忆的唤醒点菱形），再对**新**
  // 子图元素施加高亮样式（先设样式再重绘会被 innerHTML 替换抹掉）
  const w = currentWindow();
  renderTypeChart(w[0], w[1]);
  document.querySelectorAll('.sub-curve').forEach(function (p) {
    const on = p.dataset.mem === id;
    p.style.opacity = on ? '1' : '0.07';
    p.style.strokeWidth = on ? '4' : p.dataset.baseW;
  });
  document.querySelectorAll('.sub-trajectory, .sub-dot').forEach(function (c) {
    c.style.opacity = c.dataset.mem === id ? '1' : '0.08';
  });
  // 重绘后重建展开元素（renderTypeChart 只重建类型面板，主图元素保留）
  if (awakeExpanded && awakeExpanded.memId === id) {
    const evIdx = awakeExpanded.evIdx;   // 先取序号再清（clearAwakening 会置空）
    clearAwakening();
    awakeExpanded = { memId: id, evIdx: evIdx };
    const ev = evOf(selMem, evIdx);
    if (ev) drawAwakeningExpansion(id, evIdx, ev);
  }
  // 触底倒计时：选中后每秒滴答（不触底/已触底则静止显示）
  if (floorTimer) clearInterval(floorTimer);
  updateFloorCountdown(selMem);
  floorTimer = setInterval(function () { updateFloorCountdown(selMem); }, 1000);
  selecting = false;
}
function drawAwakeningExpansion(memId, evIdx, ev) {
  // 主图：展开双条（比静态标注更长更粗 + 数值标签 + 信号方向箭头）
  const ts = ev.ts, dev = ev.dev, expected = ev.expected;
  const mark = document.querySelector('.awake-mark[data-evi="' + evIdx + '"][data-mem="' + memId + '"]');
  if (!mark) return;
  const actual = strengthAtHist(memOf(memId), ts);
  const px = parseFloat(mark.getAttribute('d').split(' ')[0].slice(1));
  const pyA = parseFloat(mark.getAttribute('d').split(' ')[1]);
  const plotH = 512; // 640-64-64
  const Y = function (s) { return 64 + (1 - s) * plotH; };
  const yDevPx = Y(Math.max(0.2, actual - dev)), yExpPx = Y(Math.max(0.2, actual - expected));
  // 连接线（菱形 → 展开标签区）
  const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  ln.setAttribute('class', 'awake-exp-bar');
  ln.setAttribute('x1', px); ln.setAttribute('y1', pyA);
  ln.setAttribute('x2', px + 34); ln.setAttribute('y2', pyA - 14);
  ln.setAttribute('stroke', '#bbb'); ln.setAttribute('stroke-width', '1');
  ln.setAttribute('stroke-dasharray', '2 2');
  plot.appendChild(ln);
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  const bars = [
    { y1: yDevPx, y2: pyA, color: '#e34a2f', label: 'dev ' + dev.toFixed(3) },
    { y1: yExpPx, y2: pyA, color: '#2a9d8f', label: '预期 ' + expected.toFixed(3) },
  ];
  bars.forEach(function (b, i) {
    const x = px + 34 + i * 66;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('class', 'awake-exp-bar');
    line.setAttribute('x1', x); line.setAttribute('y1', Math.min(b.y1, b.y2));
    line.setAttribute('x2', x); line.setAttribute('y2', Math.max(b.y1, b.y2));
    line.setAttribute('stroke', b.color); line.setAttribute('stroke-width', '5');
    g.appendChild(line);
    const lb = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    lb.setAttribute('class', 'awake-exp-label');
    lb.setAttribute('x', x); lb.setAttribute('y', Math.min(b.y1, b.y2) - 4);
    lb.setAttribute('font-size', '10'); lb.setAttribute('fill', b.color);
    lb.setAttribute('text-anchor', 'middle');
    lb.textContent = b.label;
    g.appendChild(lb);
  });
  plot.appendChild(g);
  // 信号方向箭头（红↓=τ应下调 / 青↑=应上调 / 灰✓=已校准）
  const d = dirInfo(ev.ratio);
  const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  arrow.setAttribute('class', 'awake-exp-arrow');
  const ax = px + 34 + 132, ay = (yDevPx + yExpPx) / 2;
  const tri = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  if (d.mark === '↓') tri.setAttribute('d', 'M' + ax + ' ' + (ay + 2) + ' l-6 -8 h12 z');
  else if (d.mark === '↑') tri.setAttribute('d', 'M' + ax + ' ' + (ay - 2) + ' l-6 8 h12 z');
  else tri.setAttribute('d', 'M' + (ax - 5) + ' ' + ay + ' l4 5 l7 -9 z');
  tri.setAttribute('fill', d.c);
  arrow.appendChild(tri);
  const dirTxt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  dirTxt.setAttribute('class', 'awake-exp-label');
  dirTxt.setAttribute('x', ax); dirTxt.setAttribute('y', ay + (d.mark === '↓' ? 22 : d.mark === '↑' ? -12 : 22));
  dirTxt.setAttribute('font-size', '10'); dirTxt.setAttribute('fill', d.c);
  dirTxt.setAttribute('text-anchor', 'middle');
  dirTxt.textContent = d.mark === '✓' ? '已校准' : (d.mark === '↓' ? 'τ 应下调' : 'τ 应上调');
  arrow.appendChild(dirTxt);
  plot.appendChild(arrow);
}
function memOf(id) { return MEM.memories.find(function (m) { return m.id === id; }); }
function deselect() {
  if (floorTimer) { clearInterval(floorTimer); floorTimer = null; }
  clearAwakening();
  selected = null;
  document.querySelectorAll('.mem-curve').forEach(function (p) { p.style.opacity = ''; p.style.strokeWidth = p.dataset.baseW; });
  document.querySelectorAll('.mem-trajectory, .mem-dot, .access-tick, .awake-mark, .awake-dev, .awake-exp, .awake-label').forEach(function (p) { p.style.opacity = ''; });
  document.querySelectorAll('.bubble').forEach(function (c) { c.style.opacity = ''; c.setAttribute('stroke', '#fff'); c.setAttribute('stroke-width', '1'); });
  document.querySelectorAll('.sub-curve').forEach(function (p) { p.style.opacity = ''; p.style.strokeWidth = p.dataset.baseW; });
  document.querySelectorAll('.sub-trajectory, .sub-dot').forEach(function (c) { c.style.opacity = ''; });
  document.querySelectorAll('.row').forEach(function (r) { r.classList.remove('sel'); });
  document.querySelectorAll('.act-badge.linked, .drift-item.linked').forEach(function (x) { x.classList.remove('linked'); });
  typeLinked = null;
  clearTypeLink();
  aggLinked = null;
  document.querySelectorAll('.agg-hist-row').forEach(function (r) { r.classList.remove('linked'); });
  document.querySelectorAll('tr.warn-ev').forEach(function (r) { r.style.display = 'none'; });
  document.querySelectorAll('tr.warn-row').forEach(function (r) { r.classList.remove('open'); });
  document.querySelectorAll('.warn-ev-row.sel').forEach(function (r) { r.classList.remove('sel'); });
  Object.keys(selConflictEvts).forEach(function (k) { delete selConflictEvts[k]; });
  document.querySelectorAll('.wev-aggr').forEach(function (box) { renderConflictAggregate(box.dataset.mt); });
  document.getElementById('details').style.display = 'none';
}

// ---- 唤醒点展开：dev vs expected 双条 + 信号方向（点击◇/双条触发）----
let awakeExpanded = null;   // { memId, evIdx }
function evOf(mem, idx) {
  return mem && mem.awakening_events ? mem.awakening_events[idx] : null;
}
function dirInfo(ratio) {
  if (ratio == null) return { c: '#95a5a6', mark: '—', text: '旧格式事件（无类型预期），无法判向' };
  if (ratio > 1.05) return { c: '#e34a2f', mark: '↓', text: '实测跳升深于类型预期 → 埋得比信念深 → τ 应下调（或可塑性配置偏小）' };
  if (ratio < 0.95) return { c: '#2a9d8f', mark: '↑', text: '实测跳升浅于类型预期 → 忘得比信念慢 → τ 应上调（或可塑性配置偏大）' };
  return { c: '#95a5a6', mark: '✓', text: '实测 ≈ 类型预期 → 已校准（信号衰减收敛）' };
}
function clearAwakening() {
  awakeExpanded = null;
  document.querySelectorAll('.awake-exp-bar, .awake-exp-label, .awake-exp-arrow').forEach(function (el) { el.remove(); });
  const c = document.getElementById('awakeCallout');
  if (c) c.style.display = 'none';
}
function csvRowPreviewHtml(mem, ev, csvRow) {
  // 与导出 events CSV 同列：memory_id, mtype, ts, ts_relative_seconds,
  // dev, expected, ratio, dt_seconds, retrievals_before（六元组才有后两列）
  const rel = Math.round((ev.ts - MEM.now) * 10) / 10;
  const cells = [
    ['memory_id', mem.id], ['mtype', ev.mtype], ['ts', Math.round(ev.ts * 10) / 10],
    ['ts_relative_seconds', rel], ['dev', ev.dev], ['expected', ev.expected],
    ['ratio', ev.ratio == null ? '' : ev.ratio],
    ['dt_seconds', ev.dt == null ? '' : ev.dt],
    ['retrievals_before', ev.n_cold == null ? '' : ev.n_cold],
  ];
  return '<div class="csv-row-box"><div class="csv-title">原始 CSV 行预览' +
    (csvRow ? ' <span class="wev-rowno">[行 ' + csvRow + ']</span>' : '') +
    ' · events CSV 列: memory_id, mtype, ts, ts_relative_seconds, dev, expected, ratio, dt_seconds, retrievals_before</div>' +
    '<div class="csv-line">' + cells.map(function (c) {
      return '<span class="csv-cell"><b>' + c[0] + '</b><code>' + esc(String(c[1])) + '</code></span>';
    }).join('') + '</div></div>';
}
function showAwakening(memId, evIdx, csvRow) {
  const mem = memOf(memId);
  const ev = evOf(mem, evIdx);
  if (!mem || !ev) return;
  awakeExpanded = { memId: memId, evIdx: evIdx };
  select(memId);                    // 联动：全局高亮（含类型面板该记忆曲线）+ 重建展开
  // 悬浮 callout：双条比例 + 信号方向 + 解释 + 原始 CSV 行预览
  const c = document.getElementById('awakeCallout');
  const maxV = Math.max(ev.dev, ev.expected, 1e-9);
  const d = dirInfo(ev.ratio);
  const tsTxt = fmtRel(ev.ts - MEM.now);
  c.innerHTML =
    '<div class="act-title">唤醒事件展开 <span style="font-weight:normal;color:#888">' + tsTxt + '</span></div>' +
    '<div class="act-mem">' + esc(mem.summary || mem.content).slice(0, 60) + ' · [' + esc(mem.mtype) + ']</div>' +
    '<div class="act-row"><span>实测 dev</span><div class="act-bar" style="background:#e34a2f;width:' + (ev.dev / maxV * 100).toFixed(0) + '%"></div><span class="act-val">' + ev.dev.toFixed(3) + '</span></div>' +
    '<div class="act-row"><span>预期 expected</span><div class="act-bar" style="background:#2a9d8f;width:' + (ev.expected / maxV * 100).toFixed(0) + '%"></div><span class="act-val">' + ev.expected.toFixed(3) + '</span></div>' +
    '<div class="act-dir" style="background:' + d.c + '1a;color:' + d.c + ';border:1px solid ' + d.c + '55">' +
    (ev.ratio == null ? '' : '比值 ' + ev.ratio.toFixed(3) + ' · ') + '信号方向 ' + d.mark + ' — ' + d.text + '</div>' +
    csvRowPreviewHtml(mem, ev, csvRow) +
    '<div class="act-hint">点击类型面板中该记忆曲线上的 ◇ 可联动查看同一事件；Esc / 空白 / 点击其他唤醒点收起</div>';
  c.style.display = 'block';
  // CSV 行预览同步选中状态：事件已在多选集时直接带「已选 ✓」徽章
  const sset = selConflictEvts[mem.mtype] || new Set();
  if (sset.size) {
    const box = c.querySelector('.csv-row-box');
    if (box) _setCsvSelBadge(box, sset.has(conflictEvKey(mem.id, evIdx)));
  }
}
function strengthAtHist(mem, ts) {
  const rec = mem.recorded || [];
  if (!rec.length) return 0.5;
  if (ts <= rec[0][0]) return rec[0][1];
  if (ts >= rec[rec.length - 1][0]) return rec[rec.length - 1][1];
  for (let i = 1; i < rec.length; i++) {
    if (rec[i][0] >= ts) {
      const a = rec[i - 1], b = rec[i];
      const f = (ts - a[0]) / Math.max(1e-9, b[0] - a[0]);
      return a[1] + (b[1] - a[1]) * f;
    }
  }
  return rec[rec.length - 1][1];
}
function bindAwakeningClicks() {
  document.querySelectorAll('.awake-mark, .awake-dev, .awake-exp, .awake-label').forEach(function (el) {
    el.addEventListener('click', function (e) {
      e.stopPropagation();
      const idx = parseInt(el.dataset.evi, 10);
      if (!isNaN(idx)) showAwakening(el.dataset.mem, idx);
    });
  });
}

// ---- 层级切换（全局）----
function applyTierVisibility() {
  const visible = {};
  document.querySelectorAll('.btn[data-tier]').forEach(function (b) { visible[b.dataset.tier] = b.classList.contains('active'); });
  document.querySelectorAll('.tier-group[data-tier]').forEach(function (g) { g.style.display = visible[g.dataset.tier] ? '' : 'none'; });
  document.querySelectorAll('.bubble').forEach(function (c) { c.style.display = visible[c.dataset.tier] ? '' : 'none'; });
  document.querySelectorAll('.row').forEach(function (r) { r.style.display = visible[r.dataset.tier] ? '' : 'none'; });
  document.querySelectorAll('.sub-curve, .sub-trajectory, .sub-dot').forEach(function (c) { c.style.display = visible[c.dataset.tier] ? '' : 'none'; });
  renderDist();
}
document.querySelectorAll('.btn[data-tier]').forEach(function (btn) {
  btn.addEventListener('click', function () { btn.classList.toggle('active'); applyTierVisibility(); });
});
// 类型对比视图：点击子图曲线/观测点 → 全局高亮；点击唤醒菱形 → 主图展开联动
(function () {
  const tc = document.getElementById('typechart');
  if (!tc) return;
  tc.addEventListener('click', function (e) {
    const aw = e.target.closest('.type-awake');
    if (aw) { showAwakening(aw.dataset.mem, parseInt(aw.dataset.evi, 10)); return; }
    const t = e.target.closest('[data-mem]');
    if (t) { select(t.dataset.mem); return; }
    if (e.target.closest('svg')) deselect();
  });
})();
document.getElementById('reset').addEventListener('click', function () { k = 1; tx = 0; ty = 0; apply(); deselect(); });

// ---- 统计条 ----
function renderStats() {
  const s = document.getElementById('stats');
  const total = MEM.memories.length;
  const byTier = {};
  ['hot', 'warm', 'cold'].forEach(function (t) { byTier[t] = MEM.memories.filter(function (m) { return m.tier === t; }).length; });
  const avgImp = (MEM.memories.reduce(function (a, m) { return a + m.importance; }, 0) / (total || 1)).toFixed(2);
  const evts = MEM.memories.reduce(function (a, m) { return a + (m.access_events || []).length; }, 0);
  const awks = MEM.memories.reduce(function (a, m) { return a + (m.awakening_events || []).length; }, 0);
  s.innerHTML =
    '<span>记忆 <b>' + total + '</b></span>' +
    '<span>Hot <b>' + byTier.hot + '</b></span>' +
    '<span>Warm <b>' + byTier.warm + '</b></span>' +
    '<span>Cold <b>' + byTier.cold + '</b></span>' +
    '<span>平均重要性 <b>' + avgImp + '</b></span>' +
    '<span>检索事件 <b>' + evts + '</b></span>' +
    (awks ? '<span>唤醒事件 <b>' + awks + '</b></span>' : '');
}

// ---- 记忆地图（气泡图）----
let bubbleSizeMode = 'slope';  // slope = 触底倒计时（遗忘斜率），strength = 当前强度
function bubbleSize(m, maxTtf) {
  if (bubbleSizeMode === 'strength') return m.strength;
  const ttf = m.slope.time_to_floor;
  if (ttf == null) return 1;   // 不触底 = 最持久 = 最大
  if (ttf <= 0) return 0;      // 已触底 = 最小
  return Math.min(1, ttf / (maxTtf || 1));
}
function renderBubble() {
  const el = document.getElementById('bubble');
  const W = 330, H = 250, ML = 34, MR = 10, MT = 16, MB = 26;
  const maxAcc = Math.max(1, MEM.memories.reduce(function (a, m) { return Math.max(a, m.access_count); }, 0));
  const ttfs = MEM.memories.map(function (m) { return m.slope.time_to_floor; })
    .filter(function (t) { return t != null && t > 0; });
  const maxTtf = Math.max.apply(null, ttfs.length ? ttfs : [1]);
  document.getElementById('bubbleModeLabel').textContent =
    bubbleSizeMode === 'strength' ? '当前强度' : '触底倒计时（斜率）';
  document.getElementById('bubbleLegend').textContent =
    bubbleSizeMode === 'strength'
      ? '气泡大小 = 当前强度（大=强 小=弱）'
      : '气泡大小 = 触底倒计时（大=持久 小=即将触底）；点击气泡显示触底倒计时';
  const X = function (a) { return ML + (a / maxAcc) * (W - ML - MR); };
  const Y = function (i) { return MT + (1 - i) * (H - MT - MB); };
  let out = '';
  for (let i = 0; i <= 1; i += 0.25) {
    out += '<line x1="' + ML + '" y1="' + Y(i).toFixed(1) + '" x2="' + (W - MR) + '" y2="' + Y(i).toFixed(1) + '" stroke="#eee"/>';
    out += '<text x="' + (ML - 4) + '" y="' + (Y(i) + 3).toFixed(1) + '" font-size="9" fill="#999" text-anchor="end">' + i.toFixed(2) + '</text>';
  }
  const step = Math.max(1, Math.round(maxAcc / 4));
  for (let a = 0; a <= maxAcc; a += step) {
    out += '<line x1="' + X(a).toFixed(1) + '" y1="' + (H - MB) + '" x2="' + X(a).toFixed(1) + '" y2="' + (H - MB + 4) + '" stroke="#999"/>';
    out += '<text x="' + X(a).toFixed(1) + '" y="' + (H - MB + 13) + '" font-size="9" fill="#999" text-anchor="middle">' + a + '</text>';
  }
  out += '<text x="' + (ML - 4) + '" y="' + (MT - 4) + '" font-size="10" fill="#888">重要性</text>';
  out += '<text x="' + (W - MR) + '" y="' + (H - 8) + '" font-size="10" fill="#888" text-anchor="end">检索次数</text>';
  MEM.memories.forEach(function (m) {
    const s = bubbleSize(m, maxTtf);
    const r = (5 + 15 * s).toFixed(1);
    const ttf = m.slope.time_to_floor;
    const ttfText = ttf == null ? '不触底' : (ttf <= 0 ? '已触底' : fmtFloor(ttf) + '倒计时');
    out += '<circle class="bubble" data-mem="' + m.id + '" data-tier="' + m.tier + '" cx="' + X(m.access_count).toFixed(1) +
      '" cy="' + Y(m.importance).toFixed(1) + '" r="' + r + '" fill="' + TIER[m.tier] + '" fill-opacity="0.75" stroke="#fff" stroke-width="1">' +
      '<title>' + esc(m.summary || m.content) + ' 强度' + m.strength.toFixed(2) + ' 检索' + m.access_count +
      '次 · 触底倒计时 ' + ttfText + '</title></circle>';
  });
  el.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  el.innerHTML = out;
  el.querySelectorAll('.bubble').forEach(function (c) { c.addEventListener('click', function () { select(c.dataset.mem); }); });
}
document.querySelectorAll('[data-bsize]').forEach(function (b) {
  b.addEventListener('click', function () {
    bubbleSizeMode = b.dataset.bsize;
    document.querySelectorAll('[data-bsize]').forEach(function (x) { x.classList.toggle('active', x === b); });
    renderBubble();
    if (selected) select(selected);   // 重绘后恢复高亮
  });
});

// ---- 层级 × 类型分布 ----
function renderDist() {
  const el = document.getElementById('dist');
  const visible = {};
  document.querySelectorAll('.btn[data-tier]').forEach(function (b) { visible[b.dataset.tier] = b.classList.contains('active'); });
  const mtypes = ['skill', 'semantic', 'episodic'];
  let out = '';
  ['hot', 'warm', 'cold'].forEach(function (t) {
    const mems = MEM.memories.filter(function (m) { return m.tier === t && visible[t]; });
    const total = mems.length;
    out += '<div class="dist-row"><span class="dist-label" data-tier="' + t + '">' + t + '</span><div class="dist-bar">';
    mtypes.forEach(function (mt) {
      const n = mems.filter(function (m) { return m.mtype === mt; }).length;
      if (n > 0) out += '<div class="dist-seg" style="width:' + (n / (total || 1) * 100).toFixed(1) + '%;background:' + MTYPE[mt] + '"></div>';
    });
    out += '</div><span class="dist-num">' + total + '</span></div>';
  });
  el.innerHTML = out;
  el.querySelectorAll('.dist-label').forEach(function (lab) {
    lab.addEventListener('click', function () {
      const b = document.querySelector('.btn[data-tier="' + lab.dataset.tier + '"]');
      b.classList.toggle('active');
      applyTierVisibility();
    });
  });
}

// ---- 记忆类型画像 ----
function renderProfiles() {
  const el = document.getElementById('profiles');
  if (!MEM.profiles || !MEM.profiles.length) return;
  const cols = [['label', '类型'], ['tau_text', 'τ 遗忘速度'], ['drift_factor', 'drift 可塑性'],
    ['importance_factor', 'importance 可塑性'], ['cold_after_text', '压缩阈值'], ['cold_after_tau', '埋藏时机'],
    ['signal_text', '唤醒信号'], ['clean_dir', '干净段'], ['aw_dir', '唤醒'], ['consistency', '一致性'],
    ['action', '行动']];
  const sigColor = { up: '#e34a2f', down: '#2a9d8f', flat: '#8a8f98', none: '#b9bdc6' };
  const DIR = { down: ['↓', '#e34a2f', '应下调'], up: ['↑', '#2a9d8f', '应上调'], flat: ['=', '#8a8f98', '已校准'] };
  const CS = { agree: ['✔ 一致', '#2b8a3e'], conflict: ['✘ 冲突', '#c92a2a'],
    one_sided: ['△ 单源', '#8a8f98'], no_data: ['— 无信号', '#b9bdc6'] };
  // 行动徽章（suggest_adjust）：τ↓红 / τ↑青 / 需检查橙 / 其余灰，点击联动信号漂移行
  const ACT = { 'τ↓': ['τ↓', '#e34a2f', 'τ 应下调（配置偏大）'], 'τ↑': ['τ↑', '#2a9d8f', 'τ 应上调（配置偏小）'],
    '需检查': ['⚠需检查', '#e8590c', '两路信号冲突，先排查再调参'],
    '需补观测': ['需补观测', '#8a8f98', '单源信号，积累另一路'],
    '已校准': ['✓已校准', '#2b8a3e', '两路一致且无偏差'],
    '无信号': ['—', '#b9bdc6', '无观测'] };
  let out = '<table class="ptable"><thead><tr>' + cols.map(function (c) { return '<th>' + c[1] + '</th>'; }).join('') + '</tr></thead><tbody>';
  // 冲突告警索引（MEM.health.warnings）：冲突类型行 ⚠ 高亮 + 可展开两路证据
  const WARNS = {};
  (MEM.health && MEM.health.warnings || []).forEach(function (w) { WARNS[w.mtype] = w; });
  MEM.profiles.forEach(function (p) {
    const sig = p.awakening_signal || { events: 0 };
    const color = sigColor[sig.events ? sig.dominant : 'none'];
    const warn = WARNS[p.mtype];
    // τ 两路信号健康检查（MEM.health.by_type）：干净段/唤醒方向 + 一致性徽章 + 行动
    const h = (MEM.health && MEM.health.by_type) ? (MEM.health.by_type[p.mtype] || {}) : {};
    const cl = h.clean || {}, aw = h.awakening || {};
    const dirCell = function (d, tip) {
      if (!d) return '<span style="color:#c9cdd4">—</span>';
      const m = DIR[d];
      return '<span style="color:' + m[1] + ';font-weight:bold" title="' + tip + '">' + m[0] + '</span>';
    };
    const cs = h.consistency || 'no_data';
    const csM = CS[cs];
    const csTip = '建议: ' + (h.suggest || '无信号') + '（' + (h.confidence || '—') +
      ' · 干净段 n=' + (cl.n || 0) + ' / 唤醒 n=' + (aw.n || 0) + '）';
    const sug = h.suggest || '无信号';
    const am = ACT[sug] || [sug, '#8a8f98', sug];
    const actTip = am[2] + ' · 置信度 ' + (h.confidence || '—') + '（点击联动信号漂移行）';
    out += '<tr' + (warn ? ' class="warn-row"' : '') + ' data-mt="' + p.mtype + '">' +
      '<td>' + (warn ? '<span class="warn-mark" title="点击展开两路证据">⚠</span>' : '') +
      '<b>' + p.label + '</b> <span class="pcode">' + p.mtype + '</span></td>' +
      '<td>' + p.tau_text + '</td>' +
      '<td>' + p.drift_factor.toFixed(2) + '</td>' +
      '<td>' + p.importance_factor.toFixed(2) + '</td>' +
      '<td>' + p.cold_after_text + '</td>' +
      '<td>' + p.cold_after_tau.toFixed(1) + '×τ</td>' +
      '<td><span class="sig-badge" style="color:' + color + '">' + (p.signal_text || '无观测') + '</span></td>' +
      '<td>' + dirCell(cl.direction, '干净段 n=' + (cl.n || 0) +
        (cl.tau_est != null ? ' · 实测τ=' + (cl.tau_est / 86400).toFixed(1) + '天' : '') +
        (cl.cfg_tau != null ? ' vs 配置 ' + (cl.cfg_tau / 86400).toFixed(1) + '天' : '')) + '</td>' +
      '<td>' + dirCell(aw.direction, '唤醒 n=' + (aw.n || 0) +
        (aw.ratio_med != null ? ' · 中位比值 ' + aw.ratio_med.toFixed(3) : '')) + '</td>' +
      '<td><span class="cs-badge" data-mt="' + p.mtype + '" style="color:' + csM[1] + ';font-weight:bold" title="' + csTip + '">' + csM[0] + '</span></td>' +
      '<td><span class="act-badge" data-mt="' + p.mtype + '" style="background:' + am[1] + '" title="' + actTip + '">' + am[0] + '</span></td></tr>';
    if (warn) {   // 隐藏的两路证据行：点 ⚠ / 行 / 一致性徽章展开
      let evHtml = '';
      if (warn.events && warn.events.length) {
        const ED = { down: ['↓', '#e34a2f', '应下调'], up: ['↑', '#2a9d8f', '应上调'],
          flat: ['=', '#8a8f98', '已校准'], legacy: ['—', '#95a5a6', '旧格式'] };
        evHtml = '<div class="wev-evts"><div class="wev-hint">冲突成因事件 · 点击定位主图并展示 CSV 行 · Shift 点击多选聚合' +
          ' <button class="wev-copy-excl" title="复制当前选择集为 --exclude-events 参数串（memory_id:序号,...，仪表盘圈定 → CLI 剔除重判 → JSON 落盘）">复制 --exclude-events</button></div>' +
          warn.events.map(function (ev, i) {
            const d = ED[ev.direction] || ED.legacy;
            const mem = memOf(ev.memory_id);
            const preview = (mem ? (mem.summary || mem.content) : ev.memory_id).slice(0, 12);
            const rowno = ev.row ? '<span class="wev-rowno">[行 ' + ev.row + ']</span>' : '';
            return '<div class="warn-ev-row" data-mt="' + p.mtype + '" data-mem="' +
              ev.memory_id + '" data-evi="' + ev.index + '" data-row="' + (ev.row || '') +
              '" title="点击定位主图对应唤醒点 + 展示原始 CSV 行（行号对应导出 events CSV）；Shift 点击多选聚合">' +
              '#' + (i + 1) + ' ' + esc(preview) + rowno +
              ' · ratio ' + (ev.ratio != null ? ev.ratio.toFixed(3) : '—') +
              ' <b style="color:' + d[1] + '">' + d[0] + ' ' + d[2] + '</b></div>';
          }).join('') +
          '<div class="wev-aggr" data-mt="' + p.mtype + '"></div></div>';
      }
      out += '<tr class="warn-ev" data-mt="' + p.mtype + '" style="display:none"><td colspan="' + cols.length + '">' +
        '<div class="wev-t">⚠ ' + p.label + '（' + p.mtype + '）两路信号冲突</div>' +
        '<div>① ' + esc(warn.clean_evidence) + '</div>' +
        '<div>② ' + esc(warn.awakening_evidence) + '</div>' +
        '<div>→ ' + esc(warn.suggestion) + '</div>' + evHtml + '</td></tr>';
    }
  });
  out += '</tbody></table>';
  // 信号漂移：最近 30 天 vs 更早的方向一致性对比（近↓早↑ 方向翻转 = ⚠）
  if (MEM.signal_drift && MEM.signal_drift.by_type) {
    const days = Math.round((MEM.signal_drift.recent_seconds || 2592000) / 86400);
    const brief = function (s) {
      if (!s || !s.events) return '—';
      const arrow = { up: '↑', down: '↓', flat: '=' };
      return arrow[s.dominant] + Math.round(s.consistency * 100) + '%';
    };
    const warn = { '方向翻转': true, '一致性变化': true };
    let drift = '<div class="sig-drift"><b>信号漂移（近' + days + '天 vs 更早）</b>';
    MEM.profiles.forEach(function (p) {
      const d = MEM.signal_drift.by_type[p.mtype];
      if (!d) return;
      const mark = warn[d.verdict] ? '⚠' : '✔';
      drift += ' <span class="drift-item" data-mt="' + p.mtype + '" title="点击联动行动徽章" style="color:' + (warn[d.verdict] ? '#c92a2a' : '#2b8a3e') + '">'
        + p.label + ' 近' + brief(d.recent) + ' 早' + brief(d.earlier) + ' ' + mark + d.verdict + '</span>';
    });
    drift += '</div>';
    out += drift;
  }
  // 历史聚合结论（health.aggregations）回放：verdict 徽章 + 点击主图高亮事件子集
  if (MEM.health && MEM.health.aggregations && MEM.health.aggregations.length) {
    const V = { resolved: ['✔ 冲突消除', '#2b8a3e'], still_conflict: ['✘ 仍冲突', '#c92a2a'],
      insufficient: ['— 观测不足', '#8a8f98'], unknown: ['— 未判定', '#8a8f98'] };
    let aggHtml = '<div class="agg-hist"><b>历史聚合结论（health.aggregations · 点击回放）</b>';
    MEM.health.aggregations.forEach(function (a, i) {
      const v = V[a.verdict] || V.unknown;
      const d = a.selected_dist || {};
      const remTxt = a.remaining_median_ratio != null
        ? '移除后 ' + a.remaining_n + ' 起中位 ' + a.remaining_median_ratio.toFixed(3) + ' → ' + (a.remaining_direction || '—')
        : '移除后 ' + a.remaining_n + ' 起（观测不足）';
      let rec = '';
      if (a.recomputed) {
        const rh = a.recomputed.health || {};
        const bt = (rh.by_type || {})[a.mtype] || {};
        rec = ' <span class="agg-rec" title="剔除后健康证据包（--exclude-events 同链路）">剔除后 ' +
          (bt.consistency || '—') + ' · ' + (bt.suggest || '—') + ' · warnings ' +
          ((rh.warnings || []).length) + '</span>';
      }
      aggHtml += '<div class="agg-hist-row" data-idx="' + i +
        '" title="点击在主图高亮该事件子集（再点 / Esc 取消）">' +
        '<span class="agg-verdict" style="color:' + v[1] + ';font-weight:bold">' + v[0] + '</span> ' +
        a.mtype + ' · 已选 ' + a.selected_n + ' 起（↑' + (d.up || 0) + ' · ↓' + (d.down || 0) + ' · =' + (d.flat || 0) + '）· 干净段 ' +
        (a.clean_direction || '—') + ' · ' + remTxt + rec + '</div>';
    });
    aggHtml += '</div>';
    out += aggHtml;
  }
  el.innerHTML = out;
  // 证据行顶部：复制当前选择集为 --exclude-events 参数串（仪表盘圈定 → CLI 剔除重判）
  el.querySelectorAll('.wev-copy-excl').forEach(function (b) {
    b.addEventListener('click', function (e) {
      e.stopPropagation();
      copyExcludeEventsParam(b);
    });
  });
  // 行动徽章 ↔ 信号漂移行联动：点击任一侧高亮同类型的另一侧（再点/Esc 取消）
  function toggleDriftLink(mt) {
    const badge = el.querySelector('.act-badge[data-mt="' + mt + '"]');
    const item = el.querySelector('.drift-item[data-mt="' + mt + '"]');
    const on = badge ? badge.classList.toggle('linked') : false;
    if (item) item.classList.toggle('linked', on);
    return on;
  }
  el.querySelectorAll('.act-badge').forEach(function (b) {
    b.addEventListener('click', function (e) {
      e.stopPropagation();
      toggleDriftLink(b.dataset.mt);
    });
  });
  el.querySelectorAll('.drift-item').forEach(function (it) {
    it.addEventListener('click', function (e) {
      e.stopPropagation();
      toggleDriftLink(it.dataset.mt);
    });
  });
  // 一致性徽章（✘ 冲突 等）→ 主图联动：高亮该类型全部唤醒点 + 事件明细定位
  el.querySelectorAll('.cs-badge').forEach(function (b) {
    b.addEventListener('click', function (e) {
      e.stopPropagation();
      linkTypeAwakenings(b.dataset.mt);
    });
  });
  // 历史聚合结论行：点击回放该事件子集在主图高亮（再点 / Esc 取消）
  el.querySelectorAll('.agg-hist-row').forEach(function (row) {
    row.addEventListener('click', function (e) {
      e.stopPropagation();
      linkAggregation(parseInt(row.dataset.idx, 10));
    });
  });
  // 冲突类型行：点击 ⚠ / 整行展开/收起两路证据行（与一致性徽章联动）
  el.querySelectorAll('tr.warn-row').forEach(function (tr) {
    tr.addEventListener('click', function (e) {
      e.stopPropagation();
      setWarnEv(tr.dataset.mt);
    });
  });
  // 证据行内冲突事件：Shift 点击多选聚合；普通点击定位主图 + 展示原始 CSV 行
  el.querySelectorAll('.warn-ev-row').forEach(function (row) {
    row.addEventListener('click', function (e) {
      e.stopPropagation();
      if (e.shiftKey) {
        toggleConflictSel(row.dataset.mt, conflictEvKey(row.dataset.mem, parseInt(row.dataset.evi, 10)));
        return;
      }
      showAwakening(row.dataset.mem, parseInt(row.dataset.evi, 10),
        row.dataset.row ? parseInt(row.dataset.row, 10) : null);
    });
  });
  // 聚合面板基线：0 选中（全体 vs 移除后=全体）
  el.querySelectorAll('.wev-aggr').forEach(function (box) { renderConflictAggregate(box.dataset.mt); });
}

// ---- 一致性徽章 → 主图类型唤醒联动（✘ 冲突 = 定位哪几起事件造成两路冲突）----
let typeLinked = null;   // 当前联动高亮的类型
// ---- 冲突事件 Shift 多选聚合：验证“去掉这批事件后两路是否一致” ----
let selConflictEvts = {};   // { mtype: Set(key) }，key = memory_id:index
function conflictEvKey(memId, evi) { return memId + ':' + evi; }
function _ratioMed(evs) {
  const rs = evs.map(function (e) { return e.ratio; }).filter(function (r) { return r != null; })
    .sort(function (a, b) { return a - b; });
  if (!rs.length) return null;
  return rs.length % 2 ? rs[(rs.length - 1) / 2]
    : (rs[rs.length / 2 - 1] + rs[rs.length / 2]) / 2;
}
// 中位比值的信号方向——与 health 检查同闸门（_ratio_dir：1.0 ± 0.005）
function _dirOfMed(r) {
  if (r == null) return null;
  if (Math.abs(r - 1) < 0.005) return 'flat';
  return r > 1 ? 'down' : 'up';
}
// 单条事件的比值方向——与 agent `_event_ratio_dir` 同闸门（1.05/0.95），
// 也是 showTypeCallout clashes 标橙与「选反向」共用的同一判定
function _evDir(ev) {
  return ev.ratio == null ? null : (ev.ratio > 1.05 ? 'down' : (ev.ratio < 0.95 ? 'up' : 'flat'));
}
// 方向分布三段色条：↑应上调（青）/ ↓应下调（红）/ =已校准（灰）按占比等比例
// 显示，取代纯文本百分比——长度一眼可读；精确计数与百分比保留在图例与
// title 悬浮里（悬浮显示完整 n/N (pct%)）。`selDist` = 选中集各方向计数（可选）
// ——选中覆盖到的方向段叠加蓝色描边（inset 环）并在悬浮里标注「选中 n 起」。
// `mt` = 类型（可选）——色段变为可点击（data-dir），点击只圈出该方向事件
// （与「选反向」的 dir ≠ clean 判定互补）。
function _distBar(dist, total, label, selDist, mt) {
  const segs = [
    ['up', 'aggr-seg-up', '↑应上调'],
    ['down', 'aggr-seg-down', '↓应下调'],
    ['flat', 'aggr-seg-flat', '=已校准'],
  ];
  if (!total) {
    return '<div class="aggr-distbar"><span class="aggr-seg" style="width:100%;background:#e3d9c8"></span></div>' +
      '<div class="aggr-distlegend">' + label + ': — 无数据</div>';
  }
  const pct = function (n) { return Math.round(n / total * 100); };
  const parts = [], tips = [];
  segs.forEach(function (s) {
    const n = dist[s[0]] || 0;
    const selN = selDist ? (selDist[s[0]] || 0) : 0;
    const tip = s[2] + ' ' + n + '/' + total + ' (' + pct(n) + '%)' +
      (selN ? ' · 选中 ' + selN + ' 起' : '');
    tips.push(tip);
    if (n) parts.push('<span class="aggr-seg ' + s[1] + (selN ? ' aggr-sel' : '') +
      (mt ? ' aggr-seg-btn' : '') + '" data-mt="' + (mt || '') + '" data-dir="' + s[0] +
      '" style="width:' + pct(n) + '%" title="' + tip + '"></span>');
  });
  return '<div class="aggr-distbar" title="' + tips.join(' · ') +
    (mt ? '（点击色段只圈出该方向事件）' : '') + '">' + parts.join('') + '</div>' +
    '<div class="aggr-distlegend">' + label + ': ↑' + (dist.up || 0) + ' · ↓' + (dist.down || 0) + ' · =' + (dist.flat || 0) + '</div>';
}
// 选中集 CSV 行预览块：面板操作（全选/清空/选反向/Shift 逐条）后随聚合面板
// 一起重渲染——与 warn-ev-row 的 .sel 高亮、callout 徽章共用 selConflictEvts
// 同一状态源，行号对应导出 events CSV。
function _selCsvBlock(all, sel, keyOf, mt, dirCn) {
  const selEvts = all.filter(function (ev) { return sel.has(keyOf(ev)); });
  if (!selEvts.length) return '';
  return '<div class="aggr-csvsel"><b>选中集 CSV 行预览（' + selEvts.length + ' 起）</b>' +
    selEvts.map(function (ev) {
      const d = dirCn[ev.direction] || dirCn.none;
      const tip = 'memory_id=' + ev.memory_id + ' · mtype=' + (ev.mtype || mt) +
        ' · ts=' + (ev.ts != null ? ev.ts : '—') +
        ' · dev=' + (ev.dev != null ? ev.dev : '—') +
        ' · expected=' + (ev.expected != null ? ev.expected : '—') +
        ' · ratio=' + (ev.ratio != null ? ev.ratio : '—') +
        (ev.row ? ' · 对应 events CSV 第 ' + ev.row + ' 行' : '');
      return '<div class="aggr-csvrow" title="' + tip + '">' +
        (ev.row ? '<span class="wev-rowno">[行 ' + ev.row + ']</span>' : '[—]') +
        ' <code>' + ev.memory_id + '</code> · dev ' +
        (ev.dev != null ? ev.dev.toFixed(4) : '—') + ' / expected ' +
        (ev.expected != null ? ev.expected.toFixed(4) : '—') + ' · ratio ' +
        (ev.ratio != null ? ev.ratio.toFixed(4) : '—') +
        ' <b style="color:' + d[1] + '">' + d[0] + ' ' + d[2] + '</b></div>';
    }).join('') + '</div>';
}
function renderConflictAggregate(mt) {
  const box = document.querySelector('.wev-aggr[data-mt="' + mt + '"]');
  if (!box) return;
  const warn = (MEM.health.warnings || []).find(function (w) { return w.mtype === mt; });
  const h = (MEM.health && MEM.health.by_type) ? (MEM.health.by_type[mt] || {}) : {};
  const clean = (h.clean || {}).direction;
  const all = (warn && warn.events) || [];
  if (!all.length) { box.innerHTML = ''; return; }
  const sel = selConflictEvts[mt] || new Set();
  const keyOf = function (ev) { return conflictEvKey(ev.memory_id, ev.index); };
  // 全体方向分布（有比值的事件）与选中分布
  const total = all.filter(function (ev) { return ev.ratio != null; });
  const distAll = { up: 0, down: 0, flat: 0 };
  total.forEach(function (ev) { distAll[ev.direction]++; });
  const dist = { up: 0, down: 0, flat: 0 };
  let nSel = 0;
  all.forEach(function (ev) {
    if (sel.has(keyOf(ev)) && ev.ratio != null) { dist[ev.direction]++; nSel++; }
  });
  const rem = all.filter(function (ev) { return !sel.has(keyOf(ev)); });
  const allMed = _ratioMed(all);
  const remMed = _ratioMed(rem);
  const dirCn = { down: ['↓', '#e34a2f', '应下调'], up: ['↑', '#2a9d8f', '应上调'],
    flat: ['=', '#8a8f98', '已校准'], none: ['—', '#999', '未判定'] };
  const cd = dirCn[clean] || dirCn.none;
  const allD = dirCn[_dirOfMed(allMed)] || dirCn.none;
  const rd = _dirOfMed(remMed);
  const remD = dirCn[rd] || dirCn.none;
  let verdict;
  if (remMed == null) verdict = '— 剩余观测不足，无法判定';
  else if (clean && rd === clean) verdict = '✔ 移除后两路一致——冲突消除';
  else if (clean && rd && rd !== clean) verdict = '✘ 移除后仍冲突';
  else verdict = '— 方向未判定';
  box.innerHTML =
    '<div class="aggr-btns"><button class="aggr-btn aggr-all">全选</button>' +
    '<button class="aggr-btn aggr-clash" title="选择与干净段相反方向的事件（冲突成因，与 clashes 标橙同一判定）">选反向</button>' +
    '<button class="aggr-btn aggr-clear">清空</button>' +
    '<button class="aggr-btn aggr-export" title="导出全部类型当前选择集为 JSON（--aggregations-file 直接读取，免手写 memory_id）">导出聚合</button>' +
    '<button class="aggr-btn aggr-copy" title="复制全部类型当前选择集 JSON 到剪贴板（--aggregations-file 直接粘贴）">复制</button></div>' +
    '<b>Shift 多选聚合（已选 ' + nSel + ' 起）</b> ↑' + dist.up + ' · ↓' + dist.down + ' · =' + dist.flat + '<br>' +
    _distBar(distAll, total.length, '方向分布（全体 ' + total.length + ' 起）', dist, mt) +
    _distBar(dist, nSel, '选中集分布（已选 ' + nSel + ' 起）', null, mt) +
    '干净段: <b style="color:' + cd[1] + '">' + cd[0] + ' ' + cd[2] + '</b> · ' +
    '全体 ' + all.length + ' 起中位 ' + (allMed != null ? allMed.toFixed(3) : '—') + ' ' + allD[0] + '<br>' +
    '移除后剩余 ' + rem.length + ' 起中位 ' + (remMed != null ? remMed.toFixed(3) : '—') +
    ' ' + remD[0] + ' ' + remD[2] + '<br>判定: ' + verdict +
    _selCsvBlock(all, sel, keyOf, mt, dirCn) +
    (excludeEventsParam() ? '<div class="aggr-excl">--exclude-events 参数: <code>' +
      excludeEventsParam() + '</code></div>' : '');
  const bAll = box.querySelector('.aggr-all');
  const bClr = box.querySelector('.aggr-clear');
  const bClash = box.querySelector('.aggr-clash');
  const bExp = box.querySelector('.aggr-export');
  const bCpy = box.querySelector('.aggr-copy');
  if (bAll) bAll.addEventListener('click', function (e) { e.stopPropagation(); selectAllConflict(mt); });
  if (bClr) bClr.addEventListener('click', function (e) { e.stopPropagation(); clearConflictSel(mt); });
  if (bClash) bClash.addEventListener('click', function (e) { e.stopPropagation(); selectClashConflict(mt); });
  if (bExp) bExp.addEventListener('click', function (e) { e.stopPropagation(); downloadAggregations(); });
  if (bCpy) bCpy.addEventListener('click', function (e) { e.stopPropagation(); copyAggregations(bCpy); });
  // 色段点击：只圈出该方向事件（与选反向 dir≠clean 互补）
  box.querySelectorAll('.aggr-seg-btn').forEach(function (seg) {
    seg.addEventListener('click', function (e) {
      e.stopPropagation();
      selectDirConflict(seg.dataset.mt, seg.dataset.dir);
    });
  });
}
function selectDirConflict(mt, dir) {
  // 点击色段：只圈出该方向的事件（清空重选）——方向圈选 vs 选反向（dir≠clean）互补
  const warn = (MEM.health.warnings || []).find(function (w) { return w.mtype === mt; });
  if (!selConflictEvts[mt]) selConflictEvts[mt] = new Set();
  const s = selConflictEvts[mt];
  s.clear();
  (warn && warn.events || []).forEach(function (ev) {
    if (_evDir(ev) === dir) s.add(conflictEvKey(ev.memory_id, ev.index));
  });
  refreshConflictSel(mt);
}
function aggregationsExportPayload() {
  // 一键导出选择集：全部类型当前 Shift 多选状态 → --aggregations-file 规格
  // [{"mtype", "events": ['memory_id:序号', ...]}, ...]（key 与 --aggregations 同语义）
  const specs = [];
  Object.keys(selConflictEvts).forEach(function (mt) {
    const s = selConflictEvts[mt];
    if (!s || !s.size) return;
    specs.push({ mtype: mt, events: Array.from(s).sort() });
  });
  return specs;
}
function downloadAggregations() {
  const specs = aggregationsExportPayload();
  if (!specs.length) {
    alert('当前没有选中事件——先用 Shift 多选 / 全选 / 选反向圈定选择集');
    return;
  }
  const blob = new Blob([JSON.stringify(specs, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'aggregations.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
}
function legacyCopy(text) {
  // 剪贴板 API 不可用（非安全上下文 / 嵌入式 webview）时回退到 execCommand('copy')
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); } catch (e) { /* 静默失败 */ }
  ta.remove();
}
function _flashBtn(btn, msg) {
  if (!btn) return;
  const old = btn.textContent;
  btn.textContent = msg;
  setTimeout(function () { btn.textContent = old; }, 1200);
}
function excludeEventsParam() {
  // 当前选择集 → --exclude-events 参数串（memory_id:序号,...，直接粘贴 CLI）
  const keys = [];
  Object.keys(selConflictEvts).forEach(function (mt) {
    const s = selConflictEvts[mt];
    if (!s || !s.size) return;
    s.forEach(function (k) { keys.push(k); });
  });
  return keys.sort().join(',');
}
function copyExcludeEventsParam(btn) {
  // 一键复制 --exclude-events 参数串（仪表盘圈定 → CLI 剔除重判 → JSON 落盘）
  const param = excludeEventsParam();
  if (!param) {
    alert('当前没有选中事件——先用 Shift 多选 / 全选 / 选反向圈定选择集');
    return;
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(param).then(function () { _flashBtn(btn, '已复制 ✓'); },
      function () { legacyCopy(param); _flashBtn(btn, '已复制 ✓'); });
  } else {
    legacyCopy(param);
    _flashBtn(btn, '已复制 ✓');
  }
}
function copyAggregations(btn) {
  // 复制选择集 JSON 到剪贴板（--aggregations-file 直接粘贴），覆盖无下载权限环境
  const specs = aggregationsExportPayload();
  if (!specs.length) {
    alert('当前没有选中事件——先用 Shift 多选 / 全选 / 选反向圈定选择集');
    return;
  }
  const text = JSON.stringify(specs, null, 2);
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function () { _flashBtn(btn, '已复制 ✓'); },
      function () { legacyCopy(text); _flashBtn(btn, '已复制 ✓'); });
  } else {
    legacyCopy(text);
    _flashBtn(btn, '已复制 ✓');
  }
}
function syncSelVisual(mt) {
  const s = selConflictEvts[mt] || new Set();
  document.querySelectorAll('.warn-ev-row[data-mt="' + mt + '"]').forEach(function (r) {
    r.classList.toggle('sel', s.has(conflictEvKey(r.dataset.mem, parseInt(r.dataset.evi, 10))));
  });
}
// 统一的选中状态刷新：面板操作（全选/清空/选反向）与 Shift 逐条切换共用——
// 同步更新 ① 全部 warn-ev-row 的 .sel 高亮 ② 聚合面板（含选中集 CSV 行预览）
// ③ 若 callout 正打开该类型某事件的 CSV 行预览，同步其已选/未选徽章。
function refreshConflictSel(mt) {
  syncSelVisual(mt);
  renderConflictAggregate(mt);
  syncCsvPreview(mt);
}
function _setCsvSelBadge(box, sel) {
  const title = box.querySelector('.csv-title');
  if (!title) return;
  const old = title.querySelector('.csv-sel');
  if (old) old.remove();
  const mark = document.createElement('span');
  mark.className = 'csv-sel';
  mark.style.cssText = 'color:' + (sel ? '#1f6feb' : '#8a8f98');
  mark.textContent = sel ? '已选 ✓' : '未选';
  title.appendChild(mark);
}
function syncCsvPreview(mt) {
  const c = document.getElementById('awakeCallout');
  if (!c || c.style.display === 'none' || !awakeExpanded) return;
  const mem = memOf(awakeExpanded.memId);
  if (!mem || mem.mtype !== mt) return;
  const box = c.querySelector('.csv-row-box');
  if (!box) return;
  const s = selConflictEvts[mt] || new Set();
  _setCsvSelBadge(box, s.has(conflictEvKey(awakeExpanded.memId, awakeExpanded.evIdx)));
}
function selectAllConflict(mt) {
  const warn = (MEM.health.warnings || []).find(function (w) { return w.mtype === mt; });
  if (!selConflictEvts[mt]) selConflictEvts[mt] = new Set();
  const s = selConflictEvts[mt];
  (warn && warn.events || []).forEach(function (ev) { s.add(conflictEvKey(ev.memory_id, ev.index)); });
  refreshConflictSel(mt);
}
function clearConflictSel(mt) {
  const s = selConflictEvts[mt];
  if (s) s.clear();
  refreshConflictSel(mt);
}
function selectClashConflict(mt) {
  // 「选反向」：一键圈定与干净段相反方向的事件——与 clashes 标橙同一判定
  // （_evDir：clean && dir && dir !== clean），冲突成因一步选中。
  const warn = (MEM.health.warnings || []).find(function (w) { return w.mtype === mt; });
  const h = (MEM.health && MEM.health.by_type) ? (MEM.health.by_type[mt] || {}) : {};
  const clean = (h.clean || {}).direction;
  if (!selConflictEvts[mt]) selConflictEvts[mt] = new Set();
  const s = selConflictEvts[mt];
  (warn && warn.events || []).forEach(function (ev) {
    const dir = _evDir(ev);
    if (clean && dir && dir !== clean) s.add(conflictEvKey(ev.memory_id, ev.index));
  });
  refreshConflictSel(mt);
}
function toggleConflictSel(mt, key) {
  if (!selConflictEvts[mt]) selConflictEvts[mt] = new Set();
  const s = selConflictEvts[mt];
  if (s.has(key)) s.delete(key); else s.add(key);
  refreshConflictSel(mt);
}

// 冲突类型两路证据行：toggle（无参）或强制展开/收起（true/false）。
// 由 ⚠ / 警告行点击与一致性徽章联动共用；无告警类型自动 no-op。
function setWarnEv(mt, force) {
  const ev = document.querySelector('tr.warn-ev[data-mt="' + mt + '"]');
  if (!ev) return;
  const on = (force === undefined) ? (ev.style.display !== 'table-row') : force;
  ev.style.display = on ? 'table-row' : 'none';
  const tr = document.querySelector('tr.warn-row[data-mt="' + mt + '"]');
  if (tr) tr.classList.toggle('open', on);
}
function clearTypeLink() {
  document.querySelectorAll('.type-linked').forEach(function (x) { x.classList.remove('type-linked'); });
  svg.classList.remove('type-linked-mode');
}
function showTypeCallout(mt, h) {
  const c = document.getElementById('awakeCallout');
  const cl = h.clean || {}, aw = h.awakening || {};
  const dirCn = { down: '应下调', up: '应上调', flat: '已校准' };
  const cleanDir = cl.direction;
  const csTxt = { agree: '✔ 两路一致', conflict: '✘ 两路冲突', one_sided: '△ 单源', no_data: '— 无信号' };
  let html = '<div class="act-title">' + (csTxt[h.consistency] || '') + ' · ' + mt +
    '（' + (aw.n || 0) + ' 条唤醒事件定位）</div>';
  html += '<div class="act-mem">干净段 ' + (cl.n || 0) + ' 条 → ' +
    (cleanDir ? dirCn[cleanDir] : '未判定') + ' · 唤醒 ' + (aw.n || 0) + ' 条 ratio 中位 ' +
    (aw.ratio_med != null ? aw.ratio_med.toFixed(3) : '—') + ' → ' +
    (aw.direction ? dirCn[aw.direction] : '未判定') + '</div>';
  let idx = 0;
  MEM.memories.forEach(function (mem) {
    if (mem.mtype !== mt) return;
    (mem.awakening_events || []).forEach(function (ev, i) {
      idx++;
      const dir = _evDir(ev);
      const arrow = dir === 'down' ? '↓' : (dir === 'up' ? '↑' : '=');
      const side = dir == null ? '旧格式' : dirCn[dir];
      const clashes = cleanDir && dir && dir !== cleanDir;   // 与干净段相反 → 冲突成因
      const bg = clashes ? '#fff0e6' : (dir === 'down' ? '#fdeaea' : (dir === 'up' ? '#e6f7f2' : '#f4f4f5'));
      html += '<div class="act-row" style="background:' + bg + ';border-radius:4px;padding:2px 5px;cursor:pointer;margin:2px 0" '
        + 'data-mem="' + mem.id + '" data-evi="' + i + '" title="点击展开该事件双条">' +
        '#' + idx + ' ' + (mem.summary || mem.content || '').slice(0, 14) + ' · ratio ' +
        (ev.ratio != null ? ev.ratio.toFixed(3) : '—') + ' <b>' + arrow + ' ' + side + '</b>' +
        (clashes ? ' ← 与干净段相反' : '') + '</div>';
    });
  });
  html += '<div class="act-hint">点击事件行展开双条 · 主图 ◇ 已高亮 · Esc 取消</div>';
  c.innerHTML = html;
  c.style.display = 'block';
  c.querySelectorAll('.act-row').forEach(function (row) {
    row.addEventListener('click', function () {
      showAwakening(row.dataset.mem, parseInt(row.dataset.evi, 10));
    });
  });
}
function linkTypeAwakenings(mt) {
  if (typeLinked === mt) { typeLinked = null; clearTypeLink(); hideCallout(); setWarnEv(mt, false); return; }
  typeLinked = mt;
  clearTypeLink();
  MEM.memories.forEach(function (mem) {
    if (mem.mtype !== mt) return;
    document.querySelectorAll('.awake-mark[data-mem="' + mem.id + '"], .awake-dev[data-mem="' + mem.id + '"], .awake-exp[data-mem="' + mem.id + '"], .awake-label[data-mem="' + mem.id + '"]').forEach(function (x) { x.classList.add('type-linked'); });
  });
  svg.classList.add('type-linked-mode');
  showTypeCallout(mt, (MEM.health && MEM.health.by_type) ? (MEM.health.by_type[mt] || {}) : {});
  setWarnEv(mt, true);   // 联动：冲突类型的证据行随徽章展开
}
function hideCallout() {
  const c = document.getElementById('awakeCallout');
  if (c) c.style.display = 'none';
}

// ---- 历史聚合结论回放：verdict 徽章 + 点击在主图高亮该事件子集 ----
let aggLinked = null;
function clearAggLink() {
  document.querySelectorAll('.agg-hist-row').forEach(function (r) { r.classList.remove('linked'); });
  document.querySelectorAll('.type-linked').forEach(function (x) { x.classList.remove('type-linked'); });
  if (svg) svg.classList.remove('type-linked-mode');
}
function showAggCallout(agg) {
  const c = document.getElementById('awakeCallout');
  const V = { resolved: ['✔ 冲突消除', '#2b8a3e'], still_conflict: ['✘ 仍冲突', '#c92a2a'],
    insufficient: ['— 观测不足', '#8a8f98'], unknown: ['— 未判定', '#8a8f98'] };
  const v = V[agg.verdict] || V.unknown;
  const d = agg.selected_dist || {};
  const dirCn = { down: '应下调', up: '应上调', flat: '已校准' };
  const rd = agg.remaining_direction ? dirCn[agg.remaining_direction] : '未判定';
  let html = '<div class="act-title">聚合结论回放 <span style="color:' + v[1] +
    ';font-weight:bold">' + v[0] + '</span> · ' + agg.mtype + '</div>';
  html += '<div class="act-mem">已选 ' + agg.selected_n + ' 起（↑' + (d.up || 0) + ' · ↓' +
    (d.down || 0) + ' · =' + (d.flat || 0) + '）· 干净段 ' + (agg.clean_direction || '—') +
    ' · 全体中位 ' + (agg.all_median_ratio != null ? agg.all_median_ratio.toFixed(3) : '—') +
    ' → ' + (agg.all_direction || '—') + ' · 移除后 ' + agg.remaining_n + ' 起中位 ' +
    (agg.remaining_median_ratio != null ? agg.remaining_median_ratio.toFixed(3) : '—') +
    ' → ' + rd + '</div>';
  const keys = new Set(agg.events || []);
  let idx = 0;
  MEM.memories.forEach(function (mem) {
    (mem.awakening_events || []).forEach(function (ev, i) {
      if (!keys.has(conflictEvKey(mem.id, i))) return;
      idx++;
      const dir = _evDir(ev);
      const arrow = dir === 'down' ? '↓' : (dir === 'up' ? '↑' : '=');
      const side = dir == null ? '旧格式' : dirCn[dir];
      html += '<div class="act-row" style="background:#f4f6fb;border-radius:4px;padding:2px 5px;cursor:pointer;margin:2px 0" '
        + 'data-mem="' + mem.id + '" data-evi="' + i + '" title="点击展开该事件双条">' +
        '#' + idx + ' ' + (mem.summary || mem.content || '').slice(0, 14) + ' · ratio ' +
        (ev.ratio != null ? ev.ratio.toFixed(3) : '—') + ' <b>' + arrow + ' ' + side + '</b></div>';
    });
  });
  const rc = agg.recomputed;
  if (rc && rc.health) {
    const bt = (rc.health.by_type || {})[agg.mtype] || {};
    html += '<div class="act-rec">剔除后证据包（exclude ' + (rc.excluded || []).length +
      ' 起）: consistency=' + (bt.consistency || '—') + ' · suggest=' + (bt.suggest || '—') +
      ' · warnings=' + (rc.health.warnings || []).length + ' · actions=' +
      (rc.health.actions || []).length + '</div>';
  }
  html += '<div class="act-hint">点击事件行展开双条 · 主图 ◇ 已高亮该子集 · Esc 取消</div>';
  c.innerHTML = html;
  c.style.display = 'block';
  c.querySelectorAll('.act-row').forEach(function (row) {
    row.addEventListener('click', function () {
      showAwakening(row.dataset.mem, parseInt(row.dataset.evi, 10));
    });
  });
}
function linkAggregation(idx) {
  const aggs = (MEM.health && MEM.health.aggregations) || [];
  const agg = aggs[idx];
  if (!agg) return;
  if (aggLinked === idx) { aggLinked = null; clearAggLink(); hideCallout(); return; }
  aggLinked = idx;
  clearAggLink();
  const keys = new Set(agg.events || []);
  MEM.memories.forEach(function (mem) {
    (mem.awakening_events || []).forEach(function (ev, i) {
      if (!keys.has(conflictEvKey(mem.id, i))) return;
      document.querySelectorAll('.awake-mark[data-mem="' + mem.id + '"][data-evi="' + i +
        '"], .awake-dev[data-mem="' + mem.id + '"][data-evi="' + i +
        '"], .awake-exp[data-mem="' + mem.id + '"][data-evi="' + i +
        '"], .awake-label[data-mem="' + mem.id + '"][data-evi="' + i +
        '"]').forEach(function (x) { x.classList.add('type-linked'); });
    });
  });
  if (svg) svg.classList.add('type-linked-mode');
  const row = document.querySelector('.agg-hist-row[data-idx="' + idx + '"]');
  if (row) row.classList.add('linked');
  showAggCallout(agg);
}

// ---- 类型对比视图：JS 按自定义时间窗渲染 ----
// 布局常量与 Python 端一致（列宽 352 / 间距 14 / 左留白 42 / 标题高 40 / 面板高 196）
const TC = { COL_W: 352, GAP: 14, ML: 42, MR: 6, TITLE_H: 40, PANEL_H: 196 };
const TC_W = TC.ML + 3 * TC.COL_W + 2 * TC.GAP + TC.MR;
const TC_H = TC.TITLE_H + TC.PANEL_H + 30;
// 强度公式（与 decay.py 的 ScorerConfig 一致）：w_recency=1.0 w_freq=0.6 w_importance=1.2 κ=5 下限 0.2
const DECAY = { wRec: 1.0, wFreq: 0.6, wImp: 1.2, kappa: 5, floor: 0.2, denom: 2.8 };
function strengthAt(mtype, lastAccess, n, imp, t) {
  const prof = MEM.profiles.find(function (p) { return p.mtype === mtype; });
  const tau = prof ? prof.tau_seconds : 604800;
  const dt = Math.max(0, t - lastAccess);
  const s = DECAY.wRec * Math.exp(-dt / tau) + DECAY.wFreq * (1 - Math.exp(-n / DECAY.kappa)) + DECAY.wImp * imp;
  return Math.min(1, Math.max(DECAY.floor, s / DECAY.denom));
}
function genSeries(fn, tStart, tEnd, n) {
  if (tEnd <= tStart) return [];
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = tStart + (tEnd - tStart) * i / (n - 1);
    out.push([t, fn(t)]);
  }
  return out;
}

function renderTypeChart(w0, w1) {
  const svg = document.getElementById('typechart');
  if (!svg) return;
  const span = Math.max(1e-9, w1 - w0);
  const plotW = TC.COL_W - TC.ML - TC.MR;
  const visible = {};
  document.querySelectorAll('.btn[data-tier]').forEach(function (b) { visible[b.dataset.tier] = b.classList.contains('active'); });
  const px = function (t, colX) { return colX + (t - w0) / span * plotW; };
  const py = function (s, y0) { return y0 + (1 - s) * TC.PANEL_H; };
  const mtypes = ['skill', 'semantic', 'episodic'];
  let out = '';
  out += '<text x="' + TC.ML + '" y="24" font-size="15" font-weight="bold" fill="#222">遗忘斜率对比：' +
    esc(MEM.tau_summary) + '（点击曲线联动全局高亮）</text>';
  mtypes.forEach(function (mt, i) {
    const colX = TC.ML + i * (TC.COL_W + TC.GAP);
    const y0 = TC.TITLE_H + 14;
    const mems = MEM.memories.filter(function (m) { return m.mtype === mt; });
    out += '<text x="' + colX + '" y="' + (y0 - 6) + '" font-size="13" font-weight="bold" fill="#333">' +
      esc(mt + '（' + mems.length + ' 条）') + '</text>';
    out += '<text x="' + (colX - 6) + '" y="' + (y0 + 10) + '" font-size="10" fill="#999" text-anchor="end">1.0</text>';
    out += '<text x="' + (colX - 6) + '" y="' + (y0 + TC.PANEL_H - 2) + '" font-size="10" fill="#999" text-anchor="end">0.0</text>';
    for (let s = 0; s <= 1.0001; s += 0.2) {
      out += '<line x1="' + colX + '" y1="' + py(s, y0).toFixed(1) + '" x2="' + (colX + plotW) +
        '" y2="' + py(s, y0).toFixed(1) + '" stroke="#ececec" stroke-width="1"/>';
    }
    const yf = py(0.2, y0);
    out += '<line x1="' + colX + '" y1="' + yf.toFixed(1) + '" x2="' + (colX + plotW) +
      '" y2="' + yf.toFixed(1) + '" stroke="#f0b840" stroke-width="1" stroke-dasharray="5 4"/>';
    // 参考曲线（该类型典型遗忘：重要0.1、零检索，从全程起点创建）——按窗口自适应取点
    const refStart = Math.max(w0, MEM.type_window.t0);
    if (w1 > refStart) {
      const ref = genSeries(function (t) { return strengthAt(mt, MEM.type_window.t0, 0, 0.1, t); }, refStart, w1, 120)
        .map(function (p) { return px(p[0], colX).toFixed(1) + ',' + py(p[1], y0).toFixed(1); }).join(' ');
      if (ref) out += '<polyline class="sub-ref" points="' + ref + '" fill="none" stroke="#999" stroke-width="1.4" stroke-dasharray="5 4"/>';
    }
    mems.forEach(function (m) {
      const baseW = (1.2 + m.importance * 2.6).toFixed(2);
      // 观测轨迹（灰色虚线，过去段）
      const rec = m.recorded.filter(function (r) { return r[0] >= w0 && r[0] <= w1; });
      if (rec.length >= 2) {
        const rpts = rec.map(function (r) { return px(r[0], colX).toFixed(1) + ',' + py(r[1], y0).toFixed(1); }).join(' ');
        out += '<polyline class="sub-trajectory" data-mem="' + m.id + '" data-tier="' + m.tier + '" points="' + rpts +
          '" fill="none" stroke="#555" stroke-width="1.2" stroke-dasharray="3 3" stroke-opacity="0.65"/>';
      }
      // 预测曲线（从最后访问起，按窗口自适应取点）
      const predStart = Math.max(w0, m.last_access);
      const pred = w1 > predStart
        ? genSeries(function (t) { return strengthAt(m.mtype, m.last_access, m.access_count, m.importance, t); }, predStart, w1, 120)
        : [];
      if (pred.length >= 2) {
        const pts = pred.map(function (p) { return px(p[0], colX).toFixed(1) + ',' + py(p[1], y0).toFixed(1); }).join(' ');
        out += '<polyline class="sub-curve" data-mem="' + m.id + '" data-tier="' + m.tier + '" data-base-w="' + baseW +
          '" points="' + pts + '" fill="none" stroke="' + TIER[m.tier] + '" stroke-width="' + baseW +
          '" stroke-opacity="0.85"><title>' + esc(m.summary || m.content) + ' 遗忘斜率 每τ ' +
          m.slope.slope_per_tau.toFixed(2) + '（' + m.slope.label + '）· 触底验证 ' +
          floorCheckShort(m) + '</title></polyline>';
      }
      // 观测点
      rec.forEach(function (r) {
        out += '<circle class="sub-dot" data-mem="' + m.id + '" data-tier="' + m.tier + '" cx="' +
          px(r[0], colX).toFixed(1) + '" cy="' + py(r[1], y0).toFixed(1) + '" r="2.4" fill="' + TIER[m.tier] +
          '" stroke="#fff" stroke-width="0.8"/>';
      });
      // 选中记忆的唤醒点（与主图联动）：窗口内的每个唤醒事件一个小菱形，
      // 点击 → 主图展开同一事件的 dev vs expected 双条 + 信号方向。
      if (selected === m.id && m.awakening_events) {
        m.awakening_events.forEach(function (ev, ei) {
          if (ev.ts < w0 || ev.ts > w1) return;
          const sx = px(ev.ts, colX), sy = py(strengthAtHist(m, ev.ts), y0);
          const ratio = ev.ratio;
          const dirC = ratio == null ? '#95a5a6' : (ratio > 1.05 ? '#e34a2f' : (ratio < 0.95 ? '#2a9d8f' : '#95a5a6'));
          out += '<path class="type-awake" data-mem="' + m.id + '" data-evi="' + ei + '" ' +
            'd="M' + sx.toFixed(1) + ' ' + sy.toFixed(1) + ' l3.5 -3.5 l3.5 3.5 l-3.5 3.5 z" ' +
            'fill="' + dirC + '" stroke="#fff" stroke-width="0.8">' +
            '<title>唤醒 #' + (ei + 1) + ' · 比值 ' + (ratio == null ? '—' : ratio.toFixed(3)) +
            '（点击联动主图展开）</title></path>';
        });
      }
    });
    if (!mems.length) {
      out += '<text x="' + (colX + plotW / 2) + '" y="' + (y0 + TC.PANEL_H / 2) + '" font-size="12" fill="#bbb" text-anchor="middle">无 ' +
        esc(mt) + ' 类记忆</text>';
    }
    // 列底时间轴
    out += '<text x="' + (colX + plotW / 2) + '" y="' + (y0 + TC.PANEL_H + 18) + '" font-size="11" fill="#888" text-anchor="middle">' +
      fmtRel(w0 - MEM.now) + '…' + fmtRel(w1 - MEM.now) + '</text>';
  });
  // 当前时刻竖线（窗口含 now 时）
  if (w0 <= MEM.now && MEM.now <= w1) {
    const nx = px(MEM.now, TC.ML).toFixed(1);
    out += '<line x1="' + nx + '" y1="' + TC.TITLE_H + '" x2="' + nx + '" y2="' + (TC.TITLE_H + TC.PANEL_H) +
      '" stroke="#333" stroke-width="1.2" stroke-dasharray="4 4"/>';
    out += '<text x="' + nx + '" y="' + (TC.TITLE_H - 6) + '" font-size="11" fill="#333" text-anchor="middle">现在</text>';
  }
  svg.setAttribute('viewBox', '0 0 ' + TC_W + ' ' + TC_H);
  svg.innerHTML = out;
  if (selected) select(selected);
  applyTierVisibility();
}

// 时间窗控件
function applyWindow() {
  const p = parseFloat(document.getElementById('twPast').value);
  const f = parseFloat(document.getElementById('twFuture').value);
  let w0 = MEM.type_window.t0, w1 = MEM.type_window.t1;
  if (!isNaN(p) && p >= 0) w0 = MEM.now - p * 86400;
  if (!isNaN(f) && f >= 0) w1 = MEM.now + f * 86400;
  renderTypeChart(w0, w1);
}
document.getElementById('twApply').addEventListener('click', applyWindow);
document.querySelectorAll('.tw-preset').forEach(function (b) {
  b.addEventListener('click', function () {
    renderTypeChart(MEM.now + parseFloat(b.dataset.w0) * 86400, MEM.now + parseFloat(b.dataset.w1) * 86400);
  });
});
document.getElementById('twReset').addEventListener('click', function () {
  document.getElementById('twPast').value = '';
  document.getElementById('twFuture').value = '';
  renderTypeChart(MEM.type_window.t0, MEM.type_window.t1);
});

// ---- 最强记忆 Top5 ----
function renderToplist() {
  const el = document.getElementById('toplist');
  const top = MEM.memories.slice().sort(function (a, b) { return b.strength - a.strength; }).slice(0, 5);
  el.innerHTML = top.map(function (m) {
    return '<div class="row" data-mem="' + m.id + '" data-tier="' + m.tier + '">' +
      '<span class="dot" style="background:' + TIER[m.tier] + '"></span>' +
      '<span class="txt">' + esc(m.summary || m.content).slice(0, 26) + '</span>' +
      '<span class="val">' + m.strength.toFixed(2) + '</span></div>';
  }).join('');
  el.querySelectorAll('.row').forEach(function (r) { r.addEventListener('click', function () { select(r.dataset.mem); }); });
}

// ---- 详情面板 ----
function fmtRel(dt) {
  const a = Math.abs(dt);
  if (a < 60) return (dt < 0 ? '-' : '+') + a.toFixed(0) + '秒';
  if (a < 3600) return (dt < 0 ? '-' : '+') + (a / 60).toFixed(0) + '分钟';
  return (dt < 0 ? '-' : '+') + (a / 3600).toFixed(1) + '小时';
}
function fmtFloor(sec) {
  if (sec == null) return '不触底';
  if (sec === 0) return '已触底';
  if (sec < 60) return Math.round(sec) + '秒';
  if (sec < 3600) return Math.round(sec / 60) + '分钟';
  return (sec / 3600).toFixed(1) + '小时';
}
function fmtFloorCheck(m) {
  const f = m.floor_check;
  if (!f || !f.floored) return '尚未实测触底（预测 ' + fmtFloor(m.slope.time_to_floor) + '）';
  return esc(f.label);
}
function floorCheckShort(m) {
  const f = m.floor_check;
  return f && f.floored ? esc(f.label) : '未实测';
}
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function showDetails(m) {
  if (!m) return;
  const d = document.getElementById('details');
  d.style.display = 'block';
  const obs = (m.recorded || []).slice(-6).reverse().map(function (o) {
    return '<span class="obs">' + fmtRel(o[0] - MEM.now) + ' 强度 ' + o[1].toFixed(2) + '</span>';
  }).join(' · ');
  d.innerHTML =
    '<div class="dtitle">' + esc(m.id) + ' <span style="color:#888;font-weight:normal">[' + esc(m.tier) + '] [' + esc(m.mtype) + ']</span></div>' +
    '<table><tr><td>内容</td><td class="content">' + esc(m.summary || m.content) + '</td></tr>' +
    (m.mtype_confidence != null ? '<tr><td>类型置信</td><td>' + m.mtype_confidence.toFixed(2) + '</td></tr>' : '') +
    '<tr><td>重要性</td><td>' + m.importance.toFixed(2) + '</td></tr>' +
    '<tr><td>检索次数</td><td>' + m.access_count + '</td></tr>' +
    '<tr><td>当前强度</td><td>' + m.strength.toFixed(2) + '</td></tr>' +
    '<tr><td>遗忘斜率</td><td>每τ ' + m.slope.slope_per_tau.toFixed(2) +
    ' · 触底 ' + fmtFloor(m.slope.time_to_floor) +
    '（参考 ' + fmtFloor(m.slope.ref_time_to_floor) + ' → ' + m.slope.label + '）</td></tr>' +
    '<tr><td>触底验证</td><td>' + fmtFloorCheck(m) + '</td></tr>' +
    '<tr><td>触底倒计时</td><td id="floorcd">…</td></tr>' +
    '<tr><td>再巩固修订</td><td>' + m.revision_count + '</td></tr>' +
    '<tr><td>语义化评分</td><td>' + m.semanticization_score.toFixed(2) +
    (m.migrations.length ? '（迁移 ' + m.migrations.length + ' 次）' : '') + '</td></tr>' +
    (m.checks.length ? '<tr><td>一致性校验</td><td>' + m.checks.length + ' 次，最近：' +
      esc(m.checks[m.checks.length - 1][2]) + '</td></tr>' : '') +
    '<tr><td>最近观测</td><td>' + (obs || '—') + '</td></tr></table>';
}

renderStats();
renderBubble();
renderDist();
renderToplist();
renderProfiles();
renderTypeChart(MEM.type_window.t0, MEM.type_window.t1);
bindAwakeningClicks();
apply();
</script>
</body>
</html>
"""


def _health_with_aggregations(agent, aggregations: list | None = None,
                              recent_seconds: float = 30 * 86400) -> dict:
    """health 合表（tau_learner_health 单一事实源）+ 聚合结论回放：`aggregations`
    = [{"mtype", "events": ['memory_id:序号', ...]}]（与 --aggregations 同格式，
    仪表盘 Shift 多选即此 key）——resolved 自动附带剔除后证据包（与导出 JSON
    同链路 aggregation_recompute），同一记忆库上仪表盘与 --export-signals 结论
    逐字一致。"""
    health = tau_learner_health(agent)
    if aggregations:
        from .agent import aggregation_for, aggregation_recompute

        aggs = []
        for spec in aggregations:
            agg = aggregation_for(agent, health, spec)
            if agg["verdict"] == "resolved":
                agg["recomputed"] = aggregation_recompute(agent, agg["events"],
                                                          recent_seconds)
            aggs.append(agg)
        health["aggregations"] = aggs
    return health


def render_interactive_html(
    agent,
    path: str = "memories_dashboard.html",
    horizon_seconds: float | None = None,
    now: float | None = None,
    samples: int = 200,
    aggregations: list | None = None,
) -> str:
    """把全部记忆渲染成多视图联动仪表盘（单文件 HTML），返回文件路径。
    `aggregations` = [{"mtype", "events": [...]}]——回放历史聚合结论。"""
    now = now if now is not None else time.time()
    horizon = horizon_seconds or default_horizon(agent)
    static, plot, ctx = _scaffold(agent, now, horizon, samples)
    static[0] = static[0].replace("<svg ", '<svg id="chart" ')

    parts = static
    parts.append('<g id="plot">')
    parts.extend(plot)
    # 类型参考曲线（底层虚线，无交互）：配置 τ 的"典型遗忘"预期斜率
    # （类型色：技能绿/语义紫/情景橙；主图缩放跟随 #plot 组）
    for mt in MemType:
        ref_pts = " ".join(
            f"{ctx.px(t):.1f},{ctx.py(s):.1f}"
            for t, s in _reference_series(agent, mt, ctx.t0, ctx.t1, samples)
        )
        tip = f"参考：{mt.value} 典型遗忘（τ={fmt_duration(agent.cfg.tau_for(mt))}）"
        parts.append(
            f'<polyline class="main-ref" points="{ref_pts}" fill="none" '
            f'stroke="{MTYPE_COLORS[mt]}" stroke-width="1.6" stroke-dasharray="6 4" '
            f'stroke-opacity="0.75"><title>{_esc(tip)}</title></polyline>'
        )
    for tier in Tier:
        members = [m for m in ctx.memories if m.tier is tier]
        if not members:
            continue
        parts.append(f'<g class="tier-group" data-tier="{tier.value}">')
        for mem in members:
            # 预测曲线：线宽 ∝ 重要性（data-base-w 供高亮恢复）
            base_w = 1.2 + mem.importance * 2.6
            pts = " ".join(
                f"{ctx.px(t):.1f},{ctx.py(s):.1f}"
                for t, s in strength_series(agent, mem, now, horizon, samples)
            )
            tip = f"{mem.id} [{mem.tier.value}] 强度历史 {len(mem.history)} 次 · {(mem.summary or mem.content)[:60]}"
            parts.append(
                f'<polyline id="curve-{mem.id}" class="mem-curve" data-mem="{mem.id}" '
                f'data-base-w="{base_w:.2f}" points="{pts}" fill="none" '
                f'stroke="{TIER_COLORS[tier]}" stroke-width="{base_w:.2f}" '
                f'stroke-opacity="0.85"><title>{_esc(tip)}</title></polyline>'
            )
            # 实际观测轨迹（灰色虚线）
            tpts = " ".join(
                f"{ctx.px(r[0]):.1f},{ctx.py(r[1]):.1f}"
                for r in mem.history if ctx.t0 <= r[0] <= ctx.t1
            )
            if len(mem.history) >= 2:
                parts.append(
                    f'<polyline class="mem-trajectory" data-mem="{mem.id}" points="{tpts}" '
                    f'fill="none" stroke="#555" stroke-width="1.3" stroke-dasharray="3 3" '
                    f'stroke-opacity="0.65"/>'
                )
            # 观测采样点
            for r in mem.history:
                ts, s = r[0], r[1]
                if ctx.t0 <= ts <= ctx.t1:
                    parts.append(
                        f'<circle class="mem-dot" data-mem="{mem.id}" cx="{ctx.px(ts):.1f}" '
                        f'cy="{ctx.py(s):.1f}" r="3.2" fill="{TIER_COLORS[tier]}" '
                        f'stroke="#fff" stroke-width="1"/>'
                    )
            # 检索事件环标
            for ts in access_events(mem):
                if ctx.t0 <= ts <= ctx.t1:
                    s = strength_at(mem.history, ts)
                    parts.append(
                        f'<circle class="access-tick" data-mem="{mem.id}" cx="{ctx.px(ts):.1f}" '
                        f'cy="{ctx.py(s):.1f}" r="4.6" fill="none" stroke="{TIER_COLORS[tier]}" '
                        f'stroke-width="1.5"><title>检索命中 {mem.id}</title></circle>'
                    )
            # 唤醒点标注：实测偏差 dev vs 类型预期 expected 双值。
            # 菱形 = 唤醒点（唤醒后实测强度）；红条 = 实测跳升（dev，真实 τ 延续预测
            # → 实测），青条 = 类型预期跳升（expected，模型 τ 延续预测 → 实测）——
            # 两条都结束于实测点高度，红条长于青条即"唤醒比类型预期剧烈"（比值>1）。
            # 每条带 data-ev 序号：点击菱形/双条 → 展开偏差双条 + 信号方向（JS 端
            # showAwakening），并与类型面板联动。
            for ev_i, ev in enumerate(_awakening_events(mem)):
                ts, dev, exp = ev["ts"], ev["dev"], ev["expected"]
                if not (ctx.t0 <= ts <= ctx.t1):
                    continue
                actual = strength_at(mem.history, ts)
                px_ts = ctx.px(ts)
                py_act = ctx.py(actual)
                # 两条预测线在唤醒点的高度 = 实测 − 各自偏差（都 ≥ 强度下限 0.2）
                y_dev = ctx.py(max(0.2, actual - dev))
                y_exp = ctx.py(max(0.2, actual - exp))
                ratio = f"（比值 {ev['ratio']:.2f}）" if ev["ratio"] is not None else ""
                parts.append(
                    f'<path class="awake-mark" data-mem="{mem.id}" data-evi="{ev_i}" '
                    f'd="M{px_ts:.1f} {py_act:.1f} l4 -4 l4 4 l-4 4 z" '
                    f'fill="{TIER_COLORS[tier]}" stroke="#fff" stroke-width="1">'
                    f'<title>唤醒 {mem.id} 第{ev_i + 1}次 · dev {dev} vs 预期 {exp} {ratio}（点击展开）</title></path>'
                )
                parts.append(
                    f'<line class="awake-dev" data-mem="{mem.id}" data-evi="{ev_i}" '
                    f'x1="{px_ts:.1f}" y1="{y_dev:.1f}" x2="{px_ts + 9:.1f}" y2="{y_dev:.1f}" '
                    f'stroke="#e34a2f" stroke-width="2.5"/>'
                )
                parts.append(
                    f'<line class="awake-exp" data-mem="{mem.id}" data-evi="{ev_i}" '
                    f'x1="{px_ts + 2:.1f}" y1="{y_exp:.1f}" x2="{px_ts + 11:.1f}" y2="{y_exp:.1f}" '
                    f'stroke="#2a9d8f" stroke-width="2.5"/>'
                )
                parts.append(
                    f'<text class="awake-label" data-mem="{mem.id}" data-evi="{ev_i}" '
                    f'x="{px_ts + 5:.1f}" y="{py_act - 7:.1f}" font-size="8.5" fill="#666">'
                    f'dev {dev:.3f}/预期 {exp:.3f}{ratio}</text>'
                )
        parts.append("</g>")
    parts.append("</g>")  # plot
    parts.append("</svg>")

    # 类型对比视图数据：记忆状态 + 默认窗口；JS 端用强度公式（decay.py 同款）
    # 按窗口自适应生成预测/参考曲线点——窄窗密、宽窗疏，任意窗口都平滑
    data = {
        "now": now,
        "tau_summary": tau_summary(agent.cfg),
        "profiles": [p.to_dict() for p in type_profiles(agent.cfg, awakening_signal_stats(agent))],
        # 信号漂移：最近 30 天 vs 更早的方向一致性（verdict 判定见 agent.py）
        "signal_drift": awakening_signal_periods(agent),
        # τ 两路信号健康检查：干净段 vs 唤醒的方向一致性 + 冲突成因事件
        # （含 CSV 行号——行号算法已收敛于 agent.py，导出/仪表盘/终端同源）
        # + 历史聚合结论回放（health.aggregations，含 verdict 徽章 + resolved
        # 自动附带剔除后证据包，与 --export-signals 同链路）
        "health": _health_with_aggregations(agent, aggregations),
        "type_window": {"t0": ctx.t0, "t1": ctx.t1, "now": now},
        "memories": [
            {
                "id": m.id,
                "tier": m.tier.value,
                "mtype": m.mtype.value,
                "mtype_confidence": m.mtype_confidence,
                "importance": m.importance,
                "access_count": m.access_count,
                "last_access": m.last_access,
                "revision_count": m.revision_count,
                "semanticization_score": round(agent._semanticization_score(m), 4),
                "migrations": m.migrations,
                "checks": m.checks,
                "strength": round(agent._strength(m), 4),
                "slope": forgetting_slope(agent, m, now),
                "floor_check": floor_verification(agent, m, now),
                "content": m.content,
                "summary": m.summary,
                "recorded": [[r[0], r[1]] for r in m.history],
                "access_events": access_events(m),
                "awakening_events": _awakening_events(m),
            }
            for m in sorted(agent.store.all(), key=lambda m: m.id)
        ],
    }
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = _HTML_TEMPLATE.replace("__SVG__", "\n".join(parts)).replace("__DATA__", data_json)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
