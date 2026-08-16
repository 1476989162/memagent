"""从 Foxtable 官方技能库（E:\\foxtablecoder\\foxtable coder）提炼 API 知识注入记忆库。

背景：
  - 全库只有分级数据有 /坑 铁律（5 条），其余 43 个领域零铁律——LLM 在知识盲区
    凭想象写 API（轮130 HttpClient 构造签名被审查端瞎判、沉淀毒规则）；
  - 本脚本为每个 TASK_POOL 领域注入：
      1 条 [domain/API] 核心 API 速查（属性/方法精简版）
      N 条 [domain/坑] 官方「常见陷阱」（铁律级，无条件进注入窗口）
  - 同时清理已实锤的毒规则（HttpClient「无参构造」——官方签名是 New HttpClient(url)）。

幂等：内容已存在的记忆自动跳过，可重复运行。
运行前提：后台 autonomous_coder 进程已停止（FileLock 冲突）。

用法：
    python inject_api_skills.py --dry-run      # 只打印计划
    python inject_api_skills.py --domain HttpClient   # 只处理指定领域
    python inject_api_skills.py                # 全量注入
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import autonomous_coder as ac

FOX_SKILLS = Path(r"E:\foxtablecoder\foxtable coder")
FT_MEM = ac.FT_MEM_PATH

# 实锤错误的沉淀规则（审查端在知识盲区瞎编后入库的毒规则）
POISON_RULES = [
    ("HttpClient", "无参构造"),   # 官方签名 New HttpClient(url)，无参构造+Url 属性是错的
]


def _compress_table(text: str) -> str:
    """把 markdown 表格/列表压缩成紧凑文本。"""
    out: list[str] = []
    for ln in text.splitlines():
        ln = ln.rstrip()
        if not ln or ln.startswith("---") or re.match(r"^\s*\|?\s*[-| ]+\s*\|?\s*$", ln):
            continue
        ln = re.sub(r"^\s*\|", "", ln)
        ln = re.sub(r"\|\s*$", "", ln)
        if "|" in ln:
            cells = [re.sub(r"^`|`$", "", c.strip()) for c in ln.split("|")]
            out.append("：".join(c for c in cells if c))
        elif re.match(r"^\s*[-*]\s+", ln):
            out.append(re.sub(r"^\s*[-*]\s+", "", ln))
        else:
            out.append(ln.strip())
    return " ".join(out).strip()


def extract_api_and_pitfalls(domain: str) -> tuple[str, list[str]]:
    """从领域 SKILL.md 提取 (API速查文本, 陷阱列表)。"""
    path = FOX_SKILLS / domain / "SKILL.md"
    if not path.exists():
        return "", []
    txt = path.read_text(encoding="utf-8")
    # --- API 速查：优先「核心 API 速查/核心 API/核心知识卡片」section ---
    api_text = ""
    for sec in re.finditer(r"^## (.+)$", txt, re.M):
        title = sec.group(1)
        if not re.search(r"核心.*API|API.*速查|核心知识|核心 API", title):
            continue
        end = txt.find("\n## ", sec.end())
        block = txt[sec.end():end if end > 0 else len(txt)]
        compact = _compress_table(block)
        if len(compact) > 30:
            api_text = compact[:260]
            break
    # --- 陷阱：常见陷阱 / 重要警告 的编号或列表条目 ---
    pitfalls: list[str] = []
    for sec in re.finditer(r"^## (.+)$", txt, re.M):
        if not re.search(r"常见陷阱|重要警告|注意事项|易错|DataRow 替代", sec.group(1)):
            continue
        end = txt.find("\n## ", sec.end())
        block = txt[sec.end():end if end > 0 else len(txt)]
        # 标准格式：1. **陷阱名**：说明（标题与说明分离，避免非贪婪回溯错位）
        for m in re.finditer(r"^\s*\d+\.\s+\*\*(.+?)\*\*[：:]\s*(.+)$", block, re.M):
            head, body = m.group(1).strip(), m.group(2).strip()
            if head and "关联技能" not in head:
                pitfalls.append(f"{head}：{body[:120]}")
        # 兜底：无加粗标题的编号条目
        if not pitfalls:
            for m in re.finditer(r"^\s*\d+\.\s+(.+)$", block, re.M):
                head = m.group(1).strip()
                if head and "关联技能" not in head:
                    pitfalls.append(head[:120])
        # 兜底：段落内首行加粗即陷阱名（**陷阱名**：说明）
        if not pitfalls:
            for m in re.finditer(r"\*\*([^*]{2,40})\*\*[：:]\s*(.{5,180})", block):
                if "关联技能" not in m.group(1):
                    pitfalls.append(f"{m.group(1)}：{m.group(2)[:120]}")
    return api_text, pitfalls[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不写入")
    ap.add_argument("--domain", default=None, help="只处理指定领域")
    args = ap.parse_args()

    lock = ac.FileLock(str(FT_MEM) + ".autonomous.lock", timeout=0.0)
    try:
        lock.acquire()
    except ac.LockTimeoutError:
        print("后台 autonomous_coder 进程仍在运行（持锁），请先在休息间隙停止它再注入。")
        return 2

    mems = ac.load_ft_memory()
    domains = [args.domain] if args.domain else sorted(ac.TASK_POOL.keys())
    new_api = new_pit = skipped_api = skipped_pit = 0
    poisoned_removed = 0

    for d in domains:
        api_text, pitfalls = extract_api_and_pitfalls(d)
        if not api_text and not pitfalls:
            print(f"[{d}] 无可提取的 API/陷阱 section，跳过")
            continue
        now = time.time()
        # API 速查
        if api_text:
            content = f"[{d}/API] 核心API速查：{api_text}"
            if any(m.get("kind") == "skill" and m.get("content") == content for m in mems):
                skipped_api += 1
            else:
                new_api += 1
                print(f"[{d}] +API速查 ({len(content)} 字)")
                if not args.dry_run:
                    mems.append({"id": str(uuid.uuid4()), "kind": "skill", "mtype": "skill",
                                 "content": content, "importance": 0.90, "access_count": 2,
                                 "last_access": now, "tier": "warm", "created_at": now,
                                 "history": [[now, 1.0, now, 2, 0.90]]})
        # 陷阱 → /坑 铁律
        for p in pitfalls:
            content = f"[{d}/坑] 常见陷阱：{p}"
            if any(m.get("kind") == "skill" and m.get("content") == content for m in mems):
                skipped_pit += 1
                continue
            new_pit += 1
            print(f"[{d}] +坑: {p[:60]}")
            if not args.dry_run:
                mems.append({"id": str(uuid.uuid4()), "kind": "skill", "mtype": "skill",
                             "content": content, "importance": 1.0, "access_count": 2,
                             "last_access": now, "tier": "warm", "created_at": now,
                             "history": [[now, 1.0, now, 2, 1.0]]})

    # --- 清理毒规则 ---
    for dom, kw in POISON_RULES:
        keep = [m for m in mems
                if not (m.get("kind") == "skill" and dom in m.get("content", "")
                        and kw in m.get("content", ""))]
        removed = len(mems) - len(keep)
        if removed:
            poisoned_removed += removed
            print(f"[清理] 删除 {dom} 毒规则 {removed} 条（含「{kw}」——与官方签名冲突）")
        mems = keep

    print(f"\n计划: +API速查 {new_api} · +坑铁律 {new_pit} · 跳过 {skipped_api + skipped_pit} · 清毒规则 {poisoned_removed}")
    if args.dry_run:
        print("（dry-run，未写入）")
        return 0
    ac.save_ft_memory(mems)
    print(f"已写入 {FT_MEM}（总记忆 {len(mems)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
