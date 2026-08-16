"""按类型分流的再巩固内容钩子。

通用 content_updater 把所有类型都当"情境改写"处理——这对情景记忆合理，
但会污染技能记忆（程序性知识抵抗修改：回忆技能是核对，不是吸收情境）。
这里提供技能类专属的**一致性校验**钩子，配合 `content_updaters` 类型注册表使用。
"""

from __future__ import annotations

import time
from typing import Callable

from .memory import Memory

# 明确冲突的信号词（中英）：查询含这些词 → 判定与技能内容冲突
DEFAULT_CONTRADICTION_KEYWORDS = (
    "不会", "不是", "别", "不该", "不能", "无法", "做不到", "错了", "不对", "反了",
    "not", "never", "can't", "cannot", "don't", "wrong", "incorrect",
)


def consistency_checker(
    rewrite_on_conflict: Callable[[Memory, str], str | None] | None = None,
    overlap_threshold: float = 0.5,
    contradiction_keywords: tuple[str, ...] | None = None,
) -> Callable[[Memory, str, float], str | None]:
    """技能类回忆的一致性校验钩子（而非情境改写）。

    回忆技能记忆时核对"存储的技能内容"与"当前情境"是否一致，结论记入
    `mem.checks`（[时间, 查询, 结论, 相似度]），内容默认保持不变：

    - **consistent**：查询与技能内容重叠足够 → 一致，返回 None（不改写）；
    - **unknown**：重叠不足（证据不足）→ 不冒险，返回 None；
    - **conflict**：查询含明确否定/矛盾信号 → 留痕不改写（返回 None）；
      若提供了 `rewrite_on_conflict`（如 LLM 判定器），则由它返回修正后的内容
      （结论记 "corrected"，此时才真正改写技能内容）。

    返回 None 时 `_reconsolidate` 不会把这次回忆计为修订——技能记忆因此
    保持稳定，这正是"技能回忆是校验而非改写"的体现。
    """
    kws = contradiction_keywords or DEFAULT_CONTRADICTION_KEYWORDS

    from .embedding import cosine_similarity, embed_text  # 函数内导入，避免循环

    def hook(mem: Memory, query: str, lability: float) -> str | None:
        rel = cosine_similarity(embed_text(query), embed_text(mem.content))
        now = round(time.time(), 1)
        text = query.lower()
        if any(k in text for k in kws):
            if rewrite_on_conflict is not None:
                new = rewrite_on_conflict(mem, query)
                if new and new != mem.content:
                    mem.checks.append([now, query, "corrected", round(rel, 3)])
                    return new
            mem.checks.append([now, query, "conflict", round(rel, 3)])
            return None
        verdict = "consistent" if rel >= overlap_threshold else "unknown"
        mem.checks.append([now, query, verdict, round(rel, 3)])
        return None

    hook.is_checker = True  # 标记：校验型钩子——回忆只核对，不改写（_reconsolidate 据此短路）
    return hook
