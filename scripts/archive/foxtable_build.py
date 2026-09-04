#!/usr/bin/env python3
"""
FoxTable 知识蒸馏 v2 —— 写入独立记忆文件，避免与小说 agent 冲突。

最终产物：
  E:\神经网络\foxtable_memory.json  — FoxTable 专家记忆（SKILL + ROUTER + PITFALL 三层）
"""
import sys, json, re, time, uuid
from pathlib import Path

FOX_PATH = Path(r"E:\foxtablecoder\foxtable coder")
FT_MEM = Path(r"E:\神经网络\foxtable_memory.json")

# === 路由表（44 领域 → 关键词映射） ===
ROUTER = {
    "DataTable": ["数据表","增删改","AddNew","Save","Find","Select","聚合","Compute"],
    "Table": ["显示层","筛选","排序","查找","冻结列","下拉选项","加载树"],
    "动态加载与SQL": ["分页","条件加载","SQL","参数化","事务","SQLCommand"],
    "事件编程": ["DrawCell","条件样式","高亮","自动计算","列公式","跨表引用"],
    "导入导出": ["Excel导入","CSV","Merger","合并数据"],
    "二进制列": ["二进制","图片存库","文件存库"],
    "其他类型": ["表关联","数据源","用户信息"],
    "统计与查询": ["分组统计","交叉统计","同比环比","Union合并","SQLJoin"],
    "分级数据": ["BOM","树形","父子关系","MRP","分级树"],
    "窗口设计": ["窗口","Form","控件布局","自动布局","录入界面"],
    "菜单设计": ["Ribbon","功能区","MenuBuilder","Gallery"],
    "ListView": ["列表视图","大图标","多选"],
    "TreeView": ["目录树","筛选树","BuildTree","加载树"],
    "条形码": ["条码","二维码","QRCode"],
    "生成图表": ["折线图","柱状图","饼图","甘特图","Chart"],
    "Excel报表": ["Excel模板","报表填充"],
    "Word报表": ["Word模板","书签替换"],
    "WordCreator": ["纯代码Word"],
    "PDFCreator": ["纯代码PDF"],
    "专业报表": ["套打","PrintDoc","精确排版"],
    "票据设计": ["票据","发票打印"],
    "编程基础": ["语法","集合","Linq","委托","Lambda"],
    "JSON相关": ["JSON","序列化","反序列化"],
    "HttpClient": ["HTTP","HttpWebRequest","WebClient"],
    "网络相关": ["网络","Socket"],
    "本地WEB": ["本地Web","HttpServer"],
    "大模型API": ["大模型","AI模型","LLM"],
    "OpenQQ": ["QQ","TIM"],
    "工作流": ["审批流","任务流","Workflow"],
    "权限管理": ["角色","权限","菜单控制"],
    "软件加密": ["加密","授权","License"],
    "异步编程": ["Async","Await","Task"],
    "高级开发指南-异步编程": ["Async高级","后台任务"],
    "高级开发指南-WeUI框架": ["WeUI","移动端","框架"],
    "高级开发指南-HTML入门": ["HTML","CSS","JS"],
    "高级开发指南-Web数据源": ["Web数据源"],
    "高级开发指南-微信接口": ["微信","公众号","企业微信"],
    "高级开发指南-客户端类": ["客户端","工具类"],
    "高级开发指南-用Excel报表生成网页": ["Excel报表网页"],
    "开发杂项-基础篇": ["杂项基础"],
    "开发杂项-进阶篇": ["杂项进阶"],
    "协同开发": ["协同","多用户"],
    "自定义函数": ["自定义函数","Register"],
    "附录": ["VBA对照"],
}

# 总控坑（从根 SKILL.md 提取的通用规则）
GLOBAL_RULES = [
    "【通用·类型】Foxtable文档中'字符型'=String，不是Char。列类型/变量类型/函数返回值说'字符型'一律按String处理。",
    "【通用·Lambda】Lambda调用必须用.Invoke()，直接写变量名(参数)会报错。例：Dim fn=Sub(x) MessageBox.Show(x) : fn.Invoke('hi')。",
    "【通用·Lambda】Lambda参数不能用Optional，所有参数必须提供。可选参数用重载或默认值赋值方式处理。",
    "【通用·If】单行If不能接ElseIf。If x Then 代码是单行If，不能跟ElseIf。用ElseIf时Then后必须换行，用多行If块。",
    "【通用·Sub】Foxtable事件代码/自定义函数/命令窗口代码都不需要定义Sub/Function。事件用e参数，自定义函数用Args(索引)，命令窗口用Return。仅全局代码才用完整VB.NET语法。",
    # === 跨领域通用陷阱（代码自检验必查项） ===
    "【通用·SQL注入】用Select/Fill/filter拼接字符型字段值时必须加单引号包裹，如：\"字段名 = '\" & CStr(dr(\"字段名\")) & \"'\"。数字型不需要引号。禁止直接把用户输入拼进SQL字符串，必须用参数化查询。",
    "【通用·日期校验】BeforeRowSave等校验场景禁止用CDate()直接转换用户输入，必须用DateTime.TryParse先验证，验证失败时设置e.ErrorText并e.Cancel=True，避免整行抛异常。",
    "【通用·资源释放】Dim 出的对象（DataRowView、DataTable、PDFCreator、Pen、Font、StringFormat等）用完后必须.Dispose()或用Using块包裹，避免内存泄漏。",
]

# PDFCreator 核心规则（手写的正确 API 速查，避免 agent 写出错误初始化写法）
PDFCREATOR_CORE = [
    "【PDFCreator·初始化】正确写法：Dim pdc As New PDFCreator() + Dim rect As RectangleF = pdc.PageRectangle() + rect.Inflate(-72, -72)。禁止用 'Dim doc As New PDFCreator' + 'Dim pdc As pdfPage = doc.CurrentPage' 这种错误写法。PDFCreator 对象本身就是绘制对象，直接用 pdc.DrawString/DrawLine 等方法。",
    "【PDFCreator·保存】正确保存：pdc.Save(文件名)。生成PDF完整流程：New PDFCreator → PageRectangle()→Inflate 设边距 → 绘制 → Save。不需要任何 doc.CurrentPage 之类的东西。",
    "【PDFCreator·文本】DrawString 位置参数只接受 RectangleF 或 PointF，不能传两个 Single。文本用矩形绘制：pdc.DrawString(文本, 字体, Brushes.Black, rect)；用点绘制：pdc.DrawString(文本, 字体, Brushes.Black, New PointF(x, y))。",
    "【PDFCreator·新页】多页用 pdc.NewPage() 或 pdc.Pages.Add()，两者等价。",
    "【PDFCreator·页面设置】纸张用 pdc.PaperKind = Drawing.Printing.PaperKind.A4；横向用 pdc.Landscape = True；自定义纸张用 pdc.PageSize = New Size(宽磅, 高磅)。",
]

def load_md(path: Path) -> str:
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except:
            continue
    return ""

def extract_rules(text: str, domain: str) -> list:
    rules = []
    # 代码示例
    blocks = re.findall(r"```(vbnet|vb|visual basic)\s*\n(.*?)\n```", text, re.S | re.I)
    for _, code in blocks[:5]:
        lines = [l.strip() for l in code.strip().split("\n") if l.strip()]
        code_clean = " ".join(lines[:6])
        if len(code_clean) > 30:
            rules.append(f"[{domain}/代码] {code_clean[:140]}")
    # 陷阱
    for kw in ["注意", "重要", "坑", "警告", "易错", "常见错误", "不能", "不要", "禁止", "不能写", "会报错"]:
        for m in re.finditer(rf"(?:{re.escape(kw)}[：: ]?)(.{15,180})", text):
            snippet = m.group(1).strip()
            if len(snippet) > 15 and not snippet.startswith(f"[{domain}"):
                rules.append(f"[{domain}/坑] {snippet[:140]}")
    # API签名
    sigs = re.findall(r"`([A-Za-z_]+\.[A-Za-z_]+\([^)]*\))`", text)
    seen = set()
    for s in sigs[:6]:
        if s not in seen and len(s) > 8:
            seen.add(s)
            rules.append(f"[{domain}/API] {s}")
    return rules

def build() -> dict:
    mems = []
    now = time.time()
    def add(content, kind="skill", mtype="skill", imp=0.91):
        mems.append({
            "id": str(uuid.uuid4()), "kind": kind, "mtype": mtype,
            "content": content, "importance": imp, "access_count": 2,
            "last_access": now, "tier": "warm", "created_at": now,
            "history": [[now, 1.0, now, 2, imp]],
        })

    # 1. 路由表
    for domain, kws in ROUTER.items():
        add(f"[路由] 领域「{domain}」触发词：{','.join(kws)}", kind="router", mtype="skill", imp=0.95)

    # 2. 通用铁律
    for rule in GLOBAL_RULES:
        add(rule, imp=0.98)

    # 3. PDFCreator 核心规则（手写，必入库）
    for rule in PDFCREATOR_CORE:
        add(rule, imp=0.98)

    # 3. 44 领域知识蒸馏
    total = 0
    for domain, kws in ROUTER.items():
        skill_path = FOX_PATH / domain / "SKILL.md"
        if not skill_path.exists():
            continue
        text = load_md(skill_path)
        if not text:
            continue
        rules = extract_rules(text, domain)
        # 去重
        seen = set()
        unique = []
        for r in rules:
            key = r[:30]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        for r in unique:
            add(r)
            total += 1
        print(f"  [{len(unique):2d}] {domain}")

    return {"memories": mems, "summary": {
        "router_entries": len(ROUTER),
        "global_rules": len(GLOBAL_RULES),
        "distilled_rules": total,
        "total": len(mems),
    }}

if __name__ == "__main__":
    print("=" * 60)
    print("FoxTable 专家记忆构建")
    print("=" * 60)
    result = build()
    FT_MEM.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 完成 ===")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print(f"  → {FT_MEM}")