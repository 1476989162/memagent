"""指令文件生成与开工注入：AGENTS.md / CLAUDE.md 导出 + 主题注入块。

被两类入口共用（单一实现，避免两处漂移）：
- session_memory.py（CLI：--start / --inject-agents-md / --export-agents-md）
- memagent.mcp_server（MCP 工具 memagent_start / memagent_export）

CLI 专属的行为（终端打印、短主题加长提示、git log 提炼）留在调用方；
这里只提供纯函数：选记忆 → 组文本 → 写文件。
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING

from .agent import RELEVANT_TOTAL

if TYPE_CHECKING:  # 避免运行时循环导入，仅用于类型标注
    from .agent import MemoryAgent
    from .memory import Memory

MD_GROUP_TITLES = {
    "semantic": "项目事实与决策（语义类：稳定知识，被反复确认的越靠前）",
    "skill": "经验与方法（技能类：被反复验证的做法）",
    "episodic": "历史事件（情景类：随遗忘可能过时）",
}

# 展示边界预算：存储层永不截断，只在注入/导出时给单条记忆设上限。
# 超限追加唤醒指针——细节零丢失，模型按需用 memagent_recall 取全文；
# 与 Cold 层"摘要索引 + originals 深藏"同一分层哲学，搬到展示层。
INJECTION_MAX_CHARS = 200
_RECALL_POINTER = "…（长记忆，可 memagent_recall {id} 唤醒全文）"


def clip_content(text: str, mem_id: str,
                 max_chars: int = INJECTION_MAX_CHARS) -> str:
    """展示边界截断：超限给前 N 字 + 唤醒指针；max_chars<=0 视为不截断。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + _RECALL_POINTER.format(id=mem_id[:6])


def ranked_decisions(agent: "MemoryAgent", k: int) -> list[tuple["Memory", float]]:
    """按当前强度取记忆层最强的决策（排除对话流水）。"""
    ranked = sorted(
        (
            (m, agent._strength(m))
            for m in agent.store.all()
            if m.kind != "turn"
        ),
        key=lambda x: -x[1],
    )
    return ranked[:k]


def pick_decisions(agent: "MemoryAgent", topic: str | None = None,
                   k: int = 5) -> list[tuple["Memory", float]]:
    """选择要注入的决策：主题检索（rel 排序）或按强度 top-k。返回 [(记忆, 强度)]。

    短主题的子串优先重排已由核心 retrieve() 单点完成，此处不再重排。
    """
    if topic:
        hits = agent.retrieve(topic, k=k)
        return [(h.memory, h.strength) for h in hits if h.total > RELEVANT_TOTAL][:k]
    return ranked_decisions(agent, k)


def build_injection_md(agent: "MemoryAgent", topic: str | None = None, k: int = 5,
                       refresh_hint: str | None = None,
                       max_chars: int = INJECTION_MAX_CHARS) -> str:
    """生成注入块的 markdown 文本（供写文件 / 维护 AGENTS.md 区块）。

    refresh_hint：告知使用者如何刷新本区块的提示文本（CLI/MCP 各自传入
    合适的命令说明；省略则不写该行）。
    max_chars：单条记忆展示上限（0 = 不截断），超限给唤醒指针。
    """
    picked = pick_decisions(agent, topic, k)
    lines = [
        "## 本次注入的决策记忆（memagent 自动生成）",
        "",
        "以下为跨会话沉淀的开发决策，按相关性/强度注入；被反复检索的决策越来越强。",
        "",
    ]
    for i, (m, s) in enumerate(picked, 1):
        body = clip_content(m.content, m.id, max_chars)
        lines.append(f"{i}. [{m.mtype.value}] {body}（强度 {s:.2f} · 检索 {m.access_count} 次）")
    lines.append("")
    if refresh_hint:
        lines.append(f"> {refresh_hint}")
        lines.append("")
    return "\n".join(lines)


def export_agents_md_text(agent: "MemoryAgent", max_items: int = 50,
                          refresh_hint: str | None = None,
                          max_chars: int = INJECTION_MAX_CHARS) -> str:
    """把记忆层全部决策组建成 AGENTS.md 风格文档文本（按类型分组、带标注）。

    与注入区块的区别：注入是顶部 top-k 动态区块，这里导出完整决策库
    （全部条目），供全量加载指令文件的 agent 在会话开始时读取。
    """
    mems = sorted(
        (m for m in agent.store.all() if m.kind != "turn"),
        key=lambda m: (-m.importance, -m.access_count),
    )[:max_items]
    groups: dict[str, list] = {}
    for m in mems:
        groups.setdefault(m.mtype.value, []).append(m)
    lines = [
        "# 项目决策记忆（由 memagent 自动生成）",
        "",
        f"生成时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}；",
        f"来源：记忆层（共 {len(mems)} 条决策）。",
        "本文件供支持 AGENTS.md / CLAUDE.md 风格的 agent 在会话开始时加载。",
    ]
    if refresh_hint:
        lines.append(refresh_hint)
    lines.append("")
    for mtype in ("semantic", "skill", "episodic"):
        items = groups.get(mtype, [])
        if not items:
            continue
        lines.append(f"## {mtype} — {MD_GROUP_TITLES[mtype]}")
        lines.append("")
        for m in items:
            body = clip_content(m.content, m.id, max_chars)
            lines.append(
                f"- {body}（重要 {m.importance:.2f} · 检索 {m.access_count} 次）"
            )
        lines.append("")
    lines.append("<!-- 由 memagent 记忆层自动生成，重新运行 --export-agents-md 更新 -->")
    return "\n".join(lines)


def export_agents_md(agent: "MemoryAgent", path: str = "AGENTS.md", max_items: int = 50,
                     dual: bool = False, refresh_hint: str | None = None) -> str | tuple[str, str]:
    """导出完整决策库到 AGENTS.md 风格文档并落盘。

    dual=True 时用同一份内容同时写入 AGENTS.md 与 CLAUDE.md（逐字节一致，
    保持同步——Codex 加载前者、Claude Code 加载后者），返回两个路径；
    否则只写 path 并返回该路径。
    """
    text = export_agents_md_text(agent, max_items=max_items, refresh_hint=refresh_hint)
    if dual:
        targets = ("AGENTS.md", "CLAUDE.md")
        for t in targets:
            with open(t, "w", encoding="utf-8") as f:
                f.write(text)
        return targets
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
