"""增量追加 FoxTable 核心规则到 foxtable_memory.json，不覆盖已有自改进。"""
import json, uuid, time
now = time.time()

def add(content, imp=0.98):
    return {
        "id": str(uuid.uuid4()), "kind": "skill", "mtype": "skill",
        "content": content, "importance": imp, "access_count": 2,
        "last_access": now, "tier": "warm", "created_at": now,
        "history": [[now, 1.0, now, 2, imp]],
    }

d = json.load(open("foxtable_memory.json", encoding="utf-8"))
mems = d["memories"]

new_rules = [
    "【通用·SQL注入】用Select/Fill/filter拼接字符型字段值时必须加单引号包裹，数字型不需要引号。禁止直接把用户输入拼进SQL字符串，必须用参数化查询。",
    "【通用·日期校验】BeforeRowSave等校验场景禁止用CDate()直接转换用户输入，必须用DateTime.TryParse先验证，验证失败时设置e.ErrorText并e.Cancel=True，避免整行抛异常。",
    "【通用·资源释放】Dim 出的对象（DataRowView、DataTable、PDFCreator、Pen、Font、StringFormat等）用完后必须.Dispose()或用Using块包裹，避免内存泄漏。",
    "【PDFCreator·初始化】正确写法：Dim pdc As New PDFCreator() + pdc.PageRectangle() + rect.Inflate(-72, -72)。禁止用 Dim doc As New PDFCreator + doc.CurrentPage 这种错误写法。PDFCreator 对象本身就是绘制对象，直接用 pdc.DrawString/DrawLine 等方法。",
    "【PDFCreator·保存】正确保存：pdc.Save(文件名)。生成PDF完整流程：New PDFCreator → PageRectangle() → Inflate 设边距 → 绘制 → Save。",
    "【PDFCreator·文本】DrawString 位置参数只接受 RectangleF 或 PointF，不能传两个 Single。",
    "【PDFCreator·新页】多页用 pdc.NewPage() 或 pdc.Pages.Add()，两者等价。",
    "【PDFCreator·页面设置】纸张用 pdc.PaperKind = Drawing.Printing.PaperKind.A4；横向用 pdc.Landscape = True。",
]
for r in new_rules:
    mems.append(add(r))

d["memories"] = mems
json.dump(d, open("foxtable_memory.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

ft = [m["content"] for m in mems if "PDFCreator" in m["content"] and "·" in m["content"][:20]]
print(f"追加完成，总计 {len(mems)} 条，PDFCreator 核心规则 {len(ft)} 条")