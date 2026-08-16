#!/usr/bin/env python3
"""
阶段1：知识蒸馏——遍历 44 个 FoxTable 子技能，从每个 SKILL.md 提取
核心 API 规则 + 常见陷阱，沉淀为 MemType.SKILL 记忆。

安全边界：仅读取 SKILL.md 和 references 的纯技术内容，不涉及任何用户数据。
"""
import sys, json, os, re, time, uuid
from pathlib import Path

FOX_PATH = Path(r"E:\foxtablecoder\foxtable coder")
MEM_PATH = Path(r"E:\神经网络\agent_memory.json")

# 路由表（从根 SKILL.md 提取的 44 领域映射）
DOMAIN_MAP = {
    # 数据操作
    "DataTable": "数据表三级结构/增删改查/聚合计算",
    "Table": "显示层/筛选排序查找/加载树/冻结列",
    "动态加载与SQL": "分页/条件加载/SQL/参数化/事务",
    "事件编程": "自动计算/列公式/DrawCell/条件样式",
    "导入导出": "Excel/CSV/合并/Merger",
    "二进制列": "文件/图片存数据库",
    "其他类型": "表关联/数据源/用户信息",
    # 统计查询
    "统计与查询": "分组/交叉/同比/多表关联/Union",
    "分级数据": "BOM/树形/父子/分级树/MRP",
    # 界面控件
    "窗口设计": "窗口/录入界面/Form/控件布局",
    "菜单设计": "Ribbon/功能区/MenuBuilder",
    "ListView": "列表视图/大图标/多选",
    "TreeView": "目录树/筛选树/BuildTree",
    "条形码": "条码/二维码/QRCode",
    "生成图表": "折线/柱状/饼图/甘特图",
    # 报表
    "Excel报表": "Excel模板填充",
    "Word报表": "Word模板书签替换",
    "WordCreator": "纯代码生成Word",
    "PDFCreator": "纯代码生成PDF",
    "专业报表": "精确排版/套打/PrintDoc",
    "票据设计": "票据/发票打印",
    # 高级
    "编程基础": "语法/集合/Linq/委托",
    "JSON相关": "序列化/反序列化/操作",
    "HttpClient": "HTTP请求/响应",
    "网络相关": "网络通信/Socket",
    "本地WEB": "本地Web服务",
    "大模型API": "AI模型调用",
    "OpenQQ": "QQ/TIM消息",
    "工作流": "审批流/任务流",
    "权限管理": "角色/权限/菜单控制",
    "软件加密": "加密/授权",
    "异步编程": "Async/Await/Task",
    "高级开发指南-异步编程": "Async高级",
    "高级开发指南-WeUI框架": "WeUI移动端框架",
    "高级开发指南-HTML入门": "HTML/CSS/JS基础",
    "高级开发指南-Web数据源": "Web数据源",
    "高级开发指南-微信接口": "微信公众号/企业微信",
    "高级开发指南-客户端类": "客户端工具类",
    "高级开发指南-用Excel报表生成网页": "Excel报表→网页",
    "开发杂项-基础篇": "基础杂项",
    "开发杂项-进阶篇": "进阶杂项",
    "协同开发": "多用户协同",
    "自定义函数": "注册/调用自定义函数",
    "附录": "Excel/VBA对照",
}

def load_md(path: Path) -> str:
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except:
            continue
    return ""

def extract_rules(text: str, domain: str) -> list:
    """从 SKILL.md 提取核心规则：
    1. VB.NET 代码示例（带说明注释）
    2. 显式标记的"注意/重要/坑/陷阱/pitfall"段落
    3. 速查表中的关键签名
    """
    rules = []

    # 提取代码块 + 前一行注释/说明
    blocks = re.findall(r"```(vbnet|vb|visual basic)\s*\n(.*?)\n```", text, re.S | re.I)
    for _, code in blocks[:5]:  # 最多取 5 个
        lines = [l.strip() for l in code.strip().split("\n") if l.strip()]
        code_clean = " ".join(lines[:8])  # 取前 8 行做摘要
        if len(code_clean) > 30:
            # 找前面最近的中文说明
            before = text[:text.find(code)] if text.find(code) >= 0 else ""
            desc_match = re.search(r"[。：]\s*(.{10,60})", before[-200:])
            desc = desc_match.group(1).strip() if desc_match else "代码示例"
            rules.append(f"【{domain}·代码】{desc}：{code_clean[:120]}")

    # 提取"注意/重要/坑/警告/pitfall"段落
    for m in re.finditer(r"(?:注意|重要|重要提醒|坑|警告|Pitfall|陷阱|易错|常见错误)[：:].{5,200}", text, re.I):
        snippet = m.group(0).strip()
        if len(snippet) > 15:
            rules.append(f"【{domain}·陷阱】{snippet[:120]}")

    # 提取核心签名（DataTable().AddNew() 这类）
    sigs = re.findall(r"`([A-Za-z_]+\.[A-Za-z_]+\([^)]*\)|[A-Za-z_]+\([^)]*\))`", text)
    seen = set()
    for s in sigs[:8]:
        if s not in seen and len(s) > 8:
            seen.add(s)
            rules.append(f"【{domain}·API】{s}")

    return rules

def extract_pitfalls(text: str, domain: str) -> list:
    """专门提取陷阱/反模式"""
    pitfalls = []
    for kw in ["注意", "重要", "坑", "警告", "易错", "常见错误", "不能", "不要", "禁止"]:
        for m in re.finditer(rf"(?:{re.escape(kw)}[：: ]?)(.{15,150})", text):
            snippet = m.group(1).strip()
            if snippet not in [p for p in pitfalls]:
                pitfalls.append(f"【{domain}·反模式】{snippet[:120]}")
    return pitfalls[:3]  # 每技能最多 3 条陷阱


# --- 主流程 ---
print("=" * 60)
print("FoxTable 知识蒸馏 · 阶段1")
print("=" * 60)

d = json.loads(MEM_PATH.read_text(encoding="utf-8"))
mems = d.get("memories", [])
now = time.time()

# 清理旧的 FoxTable 蒸馏记忆（如果有重试）
mems = [m for m in mems if m.get("kind") == "skill" and not m.get("content","").startswith("【FoxTable")]

total_rules = 0
domain_report = []

for dirname, brief in sorted(DOMAIN_MAP.items()):
    skill_path = FOX_PATH / dirname / "SKILL.md"
    if not skill_path.exists():
        domain_report.append(f"[SKIP] {dirname} — 文件不存在")
        continue

    text = load_md(skill_path)
    if not text:
        domain_report.append(f"[EMPTY] {dirname}")
        continue

    rules = extract_rules(text, dirname)
    pitfalls = extract_pitfalls(text, dirname)
    all_items = rules + pitfalls

    # 去重
    seen = set()
    unique = []
    for r in all_items:
        key = r[:40]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    for r in unique:
        mem = {
            "id": str(uuid.uuid4()),
            "kind": "skill",
            "mtype": "skill",
            "content": r,
            "importance": 0.91,
            "access_count": 2,
            "last_access": now,
            "tier": "warm",
            "created_at": now,
            "history": [[now, 1.0, now, 2, 0.91]],
        }
        mems.append(mem)
        total_rules += 1

    domain_report.append(f"[OK] {dirname} ({brief}) — {len(unique)} 条")

d["memories"] = mems
MEM_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

print()
for line in domain_report:
    print(line)

print(f"\n=== 蒸馏完成 ===")
print(f"  处理领域: {sum(1 for l in domain_report if l.startswith('[OK]'))} 个")
print(f"  新增规则: {total_rules} 条")
print(f"  总技能记忆: {sum(1 for m in mems if m.get('kind')=='skill')} 条")