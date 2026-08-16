"""FoxTable 自主进化 agent：抽领域 → 出题 → 生成 VB.NET 代码 → 自检验 → 沉淀改进。

用法：
    python autonomous_coder.py --cycles 10       # 跑 10 轮
    python autonomous_coder.py                    # 无限循环（Ctrl+C 停止）
    python autonomous_coder.py --min-interval 300 --max-interval 900

每轮动作：
    ① 从 44 个 FoxTable 领域随机抽 1 个（每 --cold-every 轮强制抽练习次数最少的
       冷门领域，优先补齐 DataTable/Excel报表 等零练习领域），从路由表里挑 1 个
       触发词组成"实战题目"
    ② LLM 根据记忆中的 API/代码/陷阱知识生成 VB.NET 代码
    ③ 生成后自动检测代码截断：若该领域已沉淀「禁止截断」类铁律而代码仍截断，
       记一条高优先级告警并强制下轮重练该领域（截断是 1.2 分事故的根因）
    ④ LLM 自检验码：语法正确性、是否符合 FoxTable 规范、有无踩坑
    ④ 把有效改进沉淀进 foxtable_memory.json 的 SKILL 记忆（access_count=2 + importance>=0.8）
    ⑤ 睡眠巩固 + 随机休息

设计目标：越长越准，自动避开已知陷阱，代码质量稳步上升。
"""
from __future__ import annotations

import argparse, difflib, json, os, random, re, shutil, sys, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.cli import enable_utf8  # noqa: E402
from memagent.io_utils import (FileLock, LockTimeoutError,
                               atomic_write_json, atomic_write_text)  # noqa: E402
from memagent.memory import MemType, Memory, Tier  # noqa: E402

FT_MEM_PATH = Path(__file__).resolve().parent / "foxtable_memory.json"
FOX_SKILLS_DIR = Path(r"E:\foxtablecoder\foxtable coder")
LOG_PATH = Path(__file__).resolve().parent / "works" / "foxtable_coder.log"
WORK_DIR = Path(__file__).resolve().parent / "works" / "foxtable"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# 长输出 max_tokens 上限：responder 默认 1024 会把 vbnet 代码块拦腰截断——
# 回复断在块内、``` 收不了栏 → 判「代码块未闭合」（660 轮里 286 轮截断、
# 279 轮恰好 500 字符兜底，轮654-661 连续 8 轮强制重练同一领域死循环的根因）。
CODEGEN_MAX_TOKENS = 4096   # 代码生成：完整 FoxTable 示例约需 2-3k token
CRITIQUE_MAX_TOKENS = 4096  # 自检验：五维分数 + 铁律遵守核对表（41 条）也要超 1024

# 44 领域实战题目池（每领域 2-4 个代表性问题，对应真实开发场景）
TASK_POOL = {
    "DataTable": [
        "对'订单'表按'产品'列分组，统计每个产品的总数量和总金额，显示到命令窗口",
        "批量将'订单'表中'折扣'列替换为0.15，条件是[数量]>600",
        "查找'订单'表中产品='PD01'且日期在2024年1月1日之后的所有行",
    ],
    "Table": [
        "对'订单'表按'金额'列降序排序，并冻结'产品'列",
        "用Select方法筛选出'客户'列包含'张'的所有行，并高亮显示",
        "在Table中设置'日期'列的下拉选项为最近12个月",
    ],
    "动态加载与SQL": [
        "用SQLCommand实现分页加载，每页50条，按'日期'降序",
        "用参数化查询防止SQL注入，查询产品名包含用户输入的值",
        "在事务中批量插入1000条订单，任一行失败则全部回滚",
    ],
    "事件编程": [
        "用DrawCell事件实现金额列大于10000时显示红色字体",
        "在列间公式中实现'小计'=[数量]*[单价]*(1-[折扣])",
        "用BeforeRowSave事件校验'发货日期'不能早于'订单日期'",
    ],
    "Excel报表": [
        "用Excel报表功能，基于'订单模板.xlsx'填充数据生成日报表",
        "在Excel报表中实现多表关联数据填充",
    ],
    "PDFCreator": [
        "用PDFCreator纯代码生成一份包含页眉页脚和表格的PDF报告",
        "在PDF中绘制一张带边框的数据汇总表",
    ],
    "TreeView": [
        "用BuildTree构建BOM结构树，显示父子物料层级关系",
        "在TreeView中实现按关键词筛选节点功能",
    ],
    "JSON相关": [
        "将DataTable转换为JSON字符串，包含父子表嵌套结构",
        "从API返回的JSON解析数据并写入'产品'表",
    ],
    "权限管理": [
        "实现菜单权限控制：普通用户看不到'删除'按钮",
        "用角色-菜单关联表实现动态菜单加载",
    ],
    "生成图表": [
        "用Chart控件绘制'订单'表的月度销售柱状图",
        "绘制带同比折线的组合图表",
    ],
    "导入导出": [
        "将'订单'表导出为CSV，含BOM头防止Excel中文乱码",
        "用Merger合并两个结构的DataTable",
    ],
    "二进制列": [
        "将用户选择的图片文件存入'照片'二进制列",
        "从二进制列读取图片并显示在图片框中",
    ],
    "统计与查询": [
        "用GroupTableBuilder实现'订单'表按月份和产品交叉统计",
        "用SQLJoinTableBuilder关联'订单'和'产品'表做多表统计",
    ],
    "分级数据": [
        "实现BOM展开，递归计算每个父产品的所有子物料数量",
        "在分级树中实现按层级筛选",
    ],
    "其他类型": [
        "用DataTableBuilder动态创建一个产品信息表并设置主键，再填充两行数据",
        "遍历Tables集合中所有表，打印每张表的行数和列名",
    ],
    "窗口设计": [
        "新建独立窗口设计客户信息录入表单，含文本框、下拉框与保存按钮，保存前校验必填项",
        "用代码动态创建窗口和控件，实现一个登录界面，回车键触发登录",
    ],
    "菜单设计": [
        "在Ribbon菜单中添加自定义分组和按钮，绑定Click事件弹出提示",
        "实现菜单按钮可用性控制：未登录时禁用'导出'菜单项",
    ],
    "ListView": [
        "用ListView显示订单列表，选中行后联动显示该订单的明细",
        "给ListView添加右键菜单，实现删除所选行并确认",
    ],
    "条形码": [
        "用条形码控件生成产品编号的Code128条形码并显示",
        "在报表中嵌入条形码，扫描后可按编码查询对应产品",
    ],
    "Word报表": [
        "用Word报表模板填充订单数据，生成送货单Word文档",
        "在Word报表中实现多行明细表格的自动填充",
    ],
    "WordCreator": [
        "用WordCreator纯代码创建包含标题和段落的Word文档",
        "用WordCreator在文档中插入表格并设置边框样式",
    ],
    "专业报表": [
        "用专业报表制作销售排行Top10报表，支持分组与合计",
        "设计一张带封面和目录的专业报表并导出为PDF",
    ],
    "票据设计": [
        "用票据设计制作一张带公司抬头的发票样式票据",
        "设计打印票据，纸张尺寸设为80mm热敏纸",
    ],
    "编程基础": [
        "用For循环遍历订单表，统计金额大于1000的订单数量",
        "定义自定义函数计算两个日期的间隔天数并返回整数",
    ],
    "HttpClient": [
        "用HttpClient从REST API获取数据并反序列化写入DataTable",
        "用HttpClient上传文件到服务器接口，并设置超时与错误处理",
    ],
    "网络相关": [
        "用WebClient下载远程文件到本地并显示下载进度",
        "检查网络连接状态，断网时给用户友好提示并重试",
    ],
    "本地WEB": [
        "启动本地HTTP服务，把订单数据以JSON接口暴露给局域网",
        "用本地WEB功能搭建一个简单的数据查询页面",
    ],
    "大模型API": [
        "调用OpenAI兼容接口，把订单摘要发送给大模型并显示回复",
        "封装函数调用大模型API做文本分类，处理429限流与超时",
    ],
    "OpenQQ": [
        "用OpenQQ监听群消息，收到指定关键词自动回复订单查询结果",
        "用OpenQQ发送私聊消息给指定QQ号",
    ],
    "工作流": [
        "用工作流功能定义审批流程：提交→经理审批→归档",
        "实现工作流节点超时自动提醒功能",
    ],
    "软件加密": [
        "用软件加密功能给应用添加注册码校验，未注册时限制功能",
        "把数据库连接字符串加密保存到配置文件并解密读取",
    ],
    "异步编程": [
        "用Task.Run异步加载大数据表，加载期间显示进度条且界面不卡死",
        "用Async/Await实现异步保存，保存完成后再提示用户",
    ],
    "高级开发指南-异步编程": [
        "在后台线程执行耗时计算，通过Invoke更新界面控件",
        "用异步方式批量导入Excel数据，避免界面冻结",
    ],
    "高级开发指南-WeUI框架": [
        "用WeUI样式实现移动端页面，展示订单列表卡片",
        "在WeUI页面中实现表单提交与Toast提示",
    ],
    "高级开发指南-HTML入门": [
        "生成HTML报表页面，用表格展示订单数据并支持按列排序",
        "把DataTable数据渲染成HTML表格字符串",
    ],
    "高级开发指南-Web数据源": [
        "从Web数据源读取JSON数据填充到表，处理字段缺失",
        "定时从Web接口同步数据到本地表",
    ],
    "高级开发指南-微信接口": [
        "调用微信公众号接口获取access_token并缓存复用",
        "把订单状态变更推送到微信模板消息",
    ],
    "高级开发指南-客户端类": [
        "用客户端类封装HTTP请求，统一处理错误与重试",
        "编写FTP客户端类实现文件上传与下载",
    ],
    "高级开发指南-用Excel报表生成网页": [
        "把Excel报表导出为网页版，支持在浏览器中查看",
        "用Excel报表生成HTML页面并嵌入查询条件表单",
    ],
    "开发杂项-基础篇": [
        "在应用启动时检查数据源连接，失败给出修复指引",
        "给系统添加操作日志：记录每个用户的登录与关键操作",
    ],
    "开发杂项-进阶篇": [
        "实现数据字典：把代码值翻译成中文显示（如1=已下单）",
        "给系统添加全局异常处理，未捕获异常写入日志文件",
    ],
    "协同开发": [
        "为团队共享的数据表添加字段说明注释，便于协作维护",
        "把常用功能封装成独立类库供多个项目引用",
    ],
    "自定义函数": [
        "用自定义函数实现金额大写转换（1234.5 → 壹仟贰佰叁拾肆元伍角）",
        "定义自定义函数把DataTable转成CSV字符串，处理引号与换行转义",
    ],
    "附录": [
        "综合练习：实现一个带查询、导出、图表三步的完整订单分析小工具",
        "综合练习：把两个表的关联统计结果同时输出到命令窗口和Excel",
    ],
}

# 抽题时如果领域不在 TASK_POOL，用通用题目兜底
DEFAULT_TASK = "请根据该领域的API，写一段FoxTable VB.NET代码演示核心用法"


def load_log_path() -> Path:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return LOG_PATH


def log(msg: str) -> None:
    """只写 FoxTable 专用日志，不碰 autonomous.log"""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    first_write = not LOG_PATH.exists()
    with LOG_PATH.open("ab") as f:
        if first_write:
            f.write(b"\xef\xbb\xbf")
        f.write((line + "\n").encode("utf-8"))


def load_ft_memory() -> list:
    if not FT_MEM_PATH.exists():
        return []
    d = json.loads(FT_MEM_PATH.read_text(encoding="utf-8"))
    return d.get("memories", [])


def save_ft_memory(mems: list) -> None:
    atomic_write_json(
        FT_MEM_PATH,
        {"memories": mems, "updated_at": datetime.now().isoformat()},
        backup=True,
    )


def next_cycle_number() -> int:
    numbers = []
    for path in WORK_DIR.glob("cycle_*.md"):
        match = __import__("re").match(r"cycle_(\d+)_", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


# 冷门领域保底：零练习候选里优先补齐的顺序（可自行调整）
COLD_PRIORITY = ["DataTable", "Excel报表"]


def practice_counts() -> dict:
    """各领域练习次数（含失败轮）：从日志的抽题行统计。

    用日志而非 cycle 文件：失败轮（LLM 空回复等）没有产出文件，
    但也算一次练习，应该计入，否则强制轮会反复抽同一个失败领域。
    """
    counts = {d: 0 for d in TASK_POOL}
    if not LOG_PATH.exists():
        return counts
    for ln in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"抽题: 领域「(.+?)」", ln)
        if m and m.group(1) in counts:
            counts[m.group(1)] += 1
    return counts


def least_practiced_domain() -> str:
    """练习次数最少的领域；零练习候选里优先 COLD_PRIORITY 顺序。"""
    counts = practice_counts()
    min_c = min(counts.values())
    candidates = [d for d, c in counts.items() if c == min_c]
    for pri in COLD_PRIORITY:
        if pri in candidates:
            return pri
    return random.choice(candidates)


COVERAGE_DOC = Path(__file__).resolve().parent / "docs" / "foxtable-domain-coverage.md"


def generate_coverage_doc(coverage_every: int = 20) -> str:
    """生成领域覆盖分析文档到 docs/foxtable-domain-coverage.md，返回内容。

    数据来源：
      - foxtable_coder.log → 每轮领域/分数/失败/沉淀
      - foxtable_memory.json → 各领域沉淀规则数（改进+坑+范例）
    每 --coverage-every 轮由 main 自动重写；也可手动 --write-coverage。
    """
    # --- 解析日志 ---
    log_text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    inject_ok = len(re.findall(r"注入覆盖:", log_text))
    inject_alerts = len(re.findall(r"注入截断告警", log_text))
    trunc_alerts = len(re.findall(r"高优先级截断告警", log_text))
    forced_runs = len(re.findall(r"强制重练·上轮截断告警", log_text))
    cycles: list[dict] = []
    cur: dict | None = None
    for ln in log_text.splitlines():
        # 头部兼容两种格式：普通「=== 第 N 轮 ===」与专项训练「=== 分级数据专项 第 N 轮 ===」
        m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] === (?:(\S+?) )?第 (\d+) 轮 ===", ln)
        if m:
            cur = {"t": m.group(1), "n": int(m.group(3)), "tag": (m.group(2) or "轮"),
                   "domain": "", "code": 0, "cyc": None,
                   "scores": {}, "distilled": 0, "fail": ""}
            cycles.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"抽题: 领域「(.+?)」", ln);  m and cur.__setitem__("domain", m.group(1))
        m = re.search(r"生成代码: (\d+) 字符", ln); m and cur.__setitem__("code", int(m.group(1)))
        m = re.search(r"cycle_(\d+)_[^\s]*\.md", ln); m and cur.__setitem__("cyc", int(m.group(1)))
        m = re.search(r"生成代码失败: (.*)", ln);   m and cur.__setitem__("fail", m.group(1).strip())
        m = re.search(r"自检验异常: (.*)", ln);     m and cur.__setitem__("fail", m.group(1).strip())
        m = re.search(r"自检验: 五维 \{(.*?)\}\n", ln + "\n")
        if m:
            for dim in _DIM_NAMES:
                s = re.search(dim + r"=(\d+\.?\d*)", m.group(1))
                if s:
                    cur["scores"][dim] = float(s.group(1))
        m = re.search(r"沉淀 (\d+) 条", ln); m and cur.__setitem__("distilled", int(m.group(1)))

    def label(c: dict) -> str:
        """轮次显示名：普通轮用轮号，专项轮用 cycle 文件号（更唯一）"""
        if c["tag"] == "轮":
            return f"{c['n']}"
        return f"专项{c['cyc'] or '?'}"

    # --- 领域统计 ---
    stats = {d: {"practices": 0, "fails": 0, "scored": [], "rounds": [], "distilled": 0}
             for d in TASK_POOL}
    for c in cycles:
        if c["domain"] not in stats:
            continue
        s = stats[c["domain"]]
        s["practices"] += 1
        s["rounds"].append(label(c))
        if c["fail"]:
            s["fails"] += 1
        if c["scores"]:
            s["scored"].append(sum(c["scores"].values()) / 5)
    for m in load_ft_memory():
        md = re.match(r"\[(.+?)/(改进|坑|范例)\]", m.get("content", ""))
        if md and md.group(1) in stats:
            stats[md.group(1)]["distilled"] += 1

    total = len(cycles)
    practiced = sum(1 for s in stats.values() if s["practices"] > 0)
    never = [d for d, s in stats.items() if s["practices"] == 0]
    fails = sum(1 for c in cycles if c["fail"])
    all_scored = [a for s in stats.values() for a in s["scored"]]
    avg = sum(all_scored) / len(all_scored) if all_scored else 0
    dist_total = sum(s["distilled"] for s in stats.values())

    L: list[str] = []
    A = L.append
    A("# Foxtable 编码领域覆盖分析")
    A("")
    A(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"更新策略：每 {coverage_every} 轮自动重写（autonomous_coder 主循环内触发）")
    A("")
    A("## 总览")
    A("")
    A(f"| 指标 | 值 |")
    A(f"|---|---|")
    A(f"| 总轮次 | {total} |")
    A(f"| 领域覆盖 | {practiced}/44 |")
    A(f"| 从未练习 | {len(never)} |")
    A(f"| 打分轮均分 | {avg:.2f}（n={len(all_scored)}） |")
    A(f"| 失败轮 | {fails}（{fails/total*100:.0f}%） |")
    A(f"| 累计沉淀规则 | {dist_total} 条 |")
    A(f"| 注入覆盖核对 | {inject_ok} 轮 · 注入截断告警 {inject_alerts} 次 · "
      f"代码截断告警 {trunc_alerts} 次 · 强制重练 {forced_runs} 轮（/坑 铁律全量进入生成窗口） |")
    A("")
    A("## 领域明细（按练习次数降序）")
    A("")
    A("| 领域 | 练习 | 失败 | 打分轮 | 均分 | 沉淀规则 | 最近轮次 |")
    A("|---|---|---|---|---|---|---|")
    for d, s in sorted(stats.items(), key=lambda kv: (-kv[1]["practices"], kv[0])):
        av = sum(s["scored"]) / len(s["scored"]) if s["scored"] else "—"
        av_s = f"{av:.1f}" if av != "—" else "—"
        rounds = "、".join(str(r) for r in s["rounds"][-6:]) or "—"
        A(f"| {d} | {s['practices']} | {s['fails']} | {len(s['scored'])} | {av_s} | "
          f"{s['distilled']} | {rounds} |")
    A("")
    A("## 从未练习领域（冷门保底目标）")
    A("")
    if never:
        A("，".join(never))
    else:
        A("全部 44 领域都至少练过一次。")
    A("")
    weakest = [(d, sum(s['scored'])/len(s['scored'])) for d, s in stats.items() if s['scored']]
    weakest.sort(key=lambda kv: kv[1])
    A("## 最弱领域（有打分记录中均分最低）")
    A("")
    if weakest:
        A("| 领域 | 均分 |")
        A("|---|---|")
        for d, av in weakest[:5]:
            A(f"| {d} | {av:.1f} |")
    else:
        A("暂无打分记录。")
    A("")
    A("## 最近 20 轮明细")
    A("")
    A("| 轮 | 时间 | 领域 | 代码 | 均分 | 沉淀 | 状态 |")
    A("|---|---|---|---|---|---|---|")
    for c in cycles[-20:]:
        sc = c["scores"]
        av = sum(sc.values()) / len(sc) if sc else "—"
        av_s = f"{av:.1f}" if av != "—" else "—"
        st = "✗失败" if c["fail"] else ("✓" if sc else "△无分")
        A(f"| {label(c)} | {c['t'][5:16]} | {c['domain']} | {c['code']} | {av_s} | "
          f"{c['distilled']} | {st} |")
    A("")
    A("## 数据来源")
    A("")
    A("- 轮次/分数/失败：`works/foxtable_coder.log`（抽题/自检验/沉淀行；`专项N`=分级数据专项训练轮）")
    A("- 沉淀规则数：`foxtable_memory.json`（改进+坑+范例条目按领域统计）")
    A("- 注入覆盖：每轮生成前核对 `/坑` 铁律是否全量进入 prompt（`注入覆盖:` 日志行），截断即告警自动修复")
    A("- 冷门保底：每 N 轮强制抽练习最少的领域，`COLD_PRIORITY=[DataTable, Excel报表]`")

    content = "\n".join(L) + "\n"
    COVERAGE_DOC.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(COVERAGE_DOC, content, overwrite=True)
    return content


def pick_task(force_cold: bool = False, force_domain: str | None = None) -> tuple:
    """从 TASK_POOL 抽 (domain, task)。

    优先级：force_domain（上轮截断告警的强制重练）> force_cold 冷门保底 > 44 领域随机。
    """
    if force_domain:
        domain = force_domain
    elif force_cold:
        domain = least_practiced_domain()
    else:
        domain = random.choice(list(TASK_POOL.keys()))
    task = random.choice(TASK_POOL[domain])
    return domain, task


def _mem_rank(m: dict) -> int:
    """注入排序：/坑 铁律 > /改进 > 代码/API 样例"""
    c = m.get("content", "")
    if "/坑" in c:
        return 0
    if "/改进" in c:
        return 1
    return 2


def select_injection_window(domain_memories: list, window: int = 15) -> list:
    """选择注入窗口（接收完整记忆 dicts）：/坑 铁律无条件全量进入，
    其次 /改进（importance 高 + 新近沉淀优先），最后代码/API 样例。

    修复1（轮112 事故）：/坑 排第 16 被 [:15] 截掉 → 铁律全量进窗口；
    修复2（轮123 事故）：新沉淀的 /改进 排在记忆库末尾，JSON相关 20 条里
    第 18 位的 .DataTable 教训永远进不了 prompt（规则入库≠注入）→
    改进按"新近优先"排序，且窗口随记忆条数自适应扩容（最多 2 倍），
    新教训不再被末尾截断。

    修复3（轮 2026-08-15 断点研究）：回放活性（access_count）此前完全不参与
    排序——铁律进 prompt 了但 LLM 分不清「被回放 15 次的老铁律」与普通记忆，
    遵守层断点（Spearman(窗口活性占比,遵守)=+0.085 脱钩）。现让 access_count
    作为第二排序键：/坑 内部与 /改进 都按 importance 优先、access_count 次之，
    被反复回放强化的规则排前面，成为显式硬约束。
    """
    def rank_key(m):
        return (m.get("importance", 0), m.get("access_count") or 0)
    pitfalls = sorted([m for m in domain_memories if "/坑" in m.get("content", "")],
                      key=rank_key, reverse=True)
    improves = sorted([m for m in domain_memories if "/改进" in m.get("content", "")],
                      key=lambda m: (m.get("importance", 0), m.get("access_count") or 0,
                                     m.get("created_at", 0)),
                      reverse=True)
    samples = sorted([m for m in domain_memories
                      if "/坑" not in m.get("content", "")
                      and "/改进" not in m.get("content", "")],
                     key=rank_key, reverse=True)
    w = max(window, min(len(domain_memories), window * 2))
    rest_n = max(w - len(pitfalls), 0)
    return pitfalls + improves[:rest_n] + samples[:max(rest_n - len(improves), 0)]


def verify_injection(prompt: str, domain: str, mems: list) -> dict:
    """端到端核对：最终生成 prompt 文本里是否包含该领域全部 /坑 铁律。

    返回 {"total": X, "missing": Y, "missing_items": [...]}。
    missing > 0 说明注入窗口被截断或排序回归——调用方应告警并自动修复重拼。
    """
    pitfalls = [m["content"] for m in mems
                if m.get("kind") == "skill" and domain in m.get("content", "")
                and "/坑" in m.get("content", "")]
    missing = [p for p in pitfalls if p not in prompt]
    return {"total": len(pitfalls), "missing": len(missing), "missing_items": missing}


def truncation_heuristic(code: str) -> str:
    """粗略判断生成代码是否中途截断：只看最后一行是否以收尾结构结束。

    返回：完整 / 疑似截断 / 未以收尾结构结束（需人工确认） / 空代码。
    截断的常见形态（轮7/轮112 事故）：代码断在语句中途——声明未完
    （`Dim filterLevel As`）、属性访问未完（`If tbl.G`）、裸 `End`、
    调用未完（`parentObj(`）、字符串未闭合（奇数个引号）。
    注意：对**未闭合代码块**（reply 里只有 ```vbnet 没有收尾 ```）不要用本函数——
    那说明 LLM 输出本身被截断，应直接判「截断（代码块未闭合）」（见 one_cycle）。
    """
    code = (code or "").rstrip()
    if not code:
        return "空代码"
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if not lines:
        return "空代码"
    last = lines[-1]
    ok_endings = ("End Sub", "End Function", "End If", "End While", "End Select",
                  "End Try", "End Class", "End With", "End Using",
                  "Next", "Loop", "}", "Return", ")")
    # 注意：")' 结尾 = 完整函数/方法调用收尾（如 MessageBox.Show(msg)），
    # 与截断形态 `\w+\($`（以 ( 结尾的未完成调用，轮155 事故）不冲突。
    if last.endswith(ok_endings):
        return "完整"
    # 截断的常见形态：最后一行断在语句中途（全部锚定行尾，避免误报完整声明）
    if re.search(r"^(Dim\s+\w+\s+(As|=)$|If\s+\w+\.\w+$|End$|\w+\($)", last):
        return "疑似截断"
    # 字符串未闭合（奇数个双引号）→ 行内即断
    if last.count('"') % 2 == 1:
        return "疑似截断"
    return "未以收尾结构结束（需人工确认）"


def has_antitruncation_rule(domain: str, mems: list) -> bool:
    """该领域是否已沉淀「禁止截断」类铁律（关键词：截断）。"""
    return any(m.get("kind") == "skill" and domain in m.get("content", "")
               and "截断" in m.get("content", "")
               for m in mems)


def extract_code(reply: str) -> tuple[str, str]:
    """从回复提取 vbnet 代码与截断状态：(code, status)。

    - 闭合块（```vbnet ... ```）：块内容跑 truncation_heuristic；
    - 块未闭合（回复断在块内，LLM 输出被 max_tokens 截断的形态）：取开栏之后的
      **全部截断代码**——自检验/沉淀能看到实际写出的部分，而不是旧版
      reply[:500] 兜底（设计思路等正文 + 代码开头混在一起，660 轮里 279 轮
      截断都只拿前 500 字符去自检验，评分与教训失真）；
    - 完全没有 ```vbnet 开栏：退回前 500 字符（纯正文回复的旧兜底）。
    """
    code_match = re.search(r"```vbnet\s*\n(.*?)\n```", reply, re.S)
    if code_match is not None:
        code = code_match.group(1)
        return code, truncation_heuristic(code)
    opened = re.search(r"```vbnet\s*\n(.*)$", reply, re.S)
    code = opened.group(1).rstrip() if opened else reply[:500]
    return code, "截断（代码块未闭合）"


# 上轮截断告警 → 下轮强制重练的领域（进程内单例，跨 one_cycle 传递）
_FORCED_RETRAIN_DOMAIN: str | None = None

# 强制重练连续上限：同一领域连续 N 轮截断后停止强制（系统性地截断靠重练
# 修不好——典型根因是任务规模超出模型单轮输出上限，重练只是烧配额；
# 轮654-661「专业报表」连续 8 轮强制重练全部截断即此形态）。
FORCED_RETRAIN_MAX_STREAK = 3
_forced_streak_domain: str | None = None
_forced_streak_count: int = 0

# 回放窗口轮数（进程内单例，main 启动时设置）：>0 时最近 N 轮沉淀的规则
# 也在 sleep 前被刷 last_access 周期性再激活（研究：默认窗口仅 1 秒，
# 旧教训从不被回放；扩窗让旧教训也能周期性强化，见 2026-08-15 实验）。
_REPLAY_ROUNDS: int = 0


def _force_retrain(domain: str) -> None:
    global _FORCED_RETRAIN_DOMAIN
    _FORCED_RETRAIN_DOMAIN = domain


def _take_forced_domain() -> str | None:
    """取出并清空强制重练领域（每轮消费一次）。"""
    global _FORCED_RETRAIN_DOMAIN
    d = _FORCED_RETRAIN_DOMAIN
    _FORCED_RETRAIN_DOMAIN = None
    return d


def _note_forced_truncation(domain: str) -> bool:
    """登记一次「强制重练后仍截断」，返回是否还应该继续强制该领域。

    同一领域连续截断计数达到 FORCED_RETRAIN_MAX_STREAK 后返回 False——
    停止强制重练转普通抽题，避免死循环烧配额；换一个领域截断时计数重新起算。
    """
    global _forced_streak_domain, _forced_streak_count
    if domain == _forced_streak_domain:
        _forced_streak_count += 1
    else:
        _forced_streak_domain, _forced_streak_count = domain, 1
    return _forced_streak_count < FORCED_RETRAIN_MAX_STREAK


def _reset_forced_retrain_state() -> None:
    """清空强制重练的全部进程内状态（测试隔离用）。"""
    global _FORCED_RETRAIN_DOMAIN, _forced_streak_domain, _forced_streak_count
    _FORCED_RETRAIN_DOMAIN = None
    _forced_streak_domain = None
    _forced_streak_count = 0


# 领域专属「提交前自检清单」：把铁律从「知识段落」升级为「行为约束」。
# 背景（2026-08-15）：轮157 HttpClient 验证轮把 7 条 /坑 铁律写进了「注意事项」但
# 代码主体没落实（同步 GetData 卡 UI、没设 SkipError、无参构造）——「注入≠遵守」。
# 生成端在写代码前必须逐项对照清单自查，不满足就修改，而不是写完复述规则。
DOMAIN_CHECKLISTS: dict[str, list[str]] = {
    "HttpClient": [
        "构造用官方签名 `Dim hc As New HttpClient(url)` 直接传 URL（狐表自研类，不是无参构造 + 手动赋 Url）",
        "需要代码手动处理错误时设 `hc.SkipError = True`（否则默认弹内置错误窗），再用 `hc.StatusCode` 判断成功/失败",
        "长耗时请求用异步：代码段首必须写 `'''Async` 标记，`Await hc.GetDataAsync()` / `Await hc.GetFileAsync(path)`，不要同步 `GetData()` 卡 UI",
        "同一实例只执行一次请求；第二次必须 `hc.Clone()` 克隆新实例（HttpClient 不能复用）",
        "上传文件用 `hc.Files.Add(字段名, 路径)`；发 JSON 设 `hc.ContentType = \"application/json\"` + `hc.Content`",
        "不要用 `Using`/`Dispose` 包裹狐表 HttpClient（狐表自研类非 IDisposable，`Using` 会编译失败）；直接 `Dim hc As New HttpClient(url)` 即可",
        "狐表 HttpClient 只有 GetData/GetDataAsync/GetFile/GetFileAsync/Clone 方法，没有 PostData；Method 自动判断，发 POST 也用 GetData",
    ],
    "生成图表": [
        "狐表 Chart 是 C1Chart 封装：`Chart.DataSource` 是 String 类型，绑定用字符串表名（如 `\"统计表\"`），不是 DataTable 对象",
        "没有 `ChartAreas` 集合，次 Y 轴是直接属性：`Chart.AxisY2.Enabled = True`（不要写 ChartAreas(0).AxisY2）",
        "没有 `AxisType`/`YAxisType` 属性；双图表（柱+线）标准做法：`Chart.ChartType2` + `Chart.SeriesList2.Clear()/Add()` + `Chart.AxisY2`，不要试图给单个系列设次坐标轴类型",
        "先 `Series.Length = N` 再赋值 `Series.X(i)/Y(i)`；编码 + 字符标签用 `AxisX.SetValueLabel` + `AnnoWithLabels = True`；绑定方式（DataField）X 轴自动显示字符，不需要 SetValueLabel",
        "绑定方式用 `Series.X.DataField/Series.Y.DataField` 时，列名用 Name 不是 Caption（分组统计表尤其注意）",
        "动态建表用 `DataTables.Add(dt)` 传 DataTable 对象（不要假定 `Add(表名, SQL)` 重载存在）；删除内存表用 `DataTables.Remove(表名)`",
    ],
}

# 通用清单（所有领域）：防截断 + 高频铁律
GENERIC_CHECKLIST = [
    "代码必须完整收尾，不得中途截断（vbnet 代码块必须闭合，最后一行以 End Sub/End If/End Function 等收尾结构结束）",
    "字符型字段值拼进 SQL/filter 必须用单引号包裹：`\"'\" & 值 & \"'\"`",
    "用户输入转日期必须用 DateTime.TryParse，不要 CDate() 直接转",
]


def build_prompt(mems: list, domain: str, task: str) -> str:
    """拼装 prompt：领域路由 + 相关技能记忆 + 铁律 + 题目 + 提交前自检清单"""
    # 收集该领域相关记忆（完整 dict，窗口函数内部排序）
    domain_memories = [m for m in mems
                       if m.get("kind") == "skill" and domain in m.get("content", "")]
    # 注入窗口：/坑 铁律无条件全量进入 + /改进 新近优先 + 自适应扩容
    # （见 select_injection_window：修复轮112 坑被截 / 轮123 新教训被截）
    domain_memories = select_injection_window(domain_memories, window=15)
    # 路由
    router = [m["content"] for m in mems if m.get("kind") == "router" and domain in m.get("content", "")]
    # 铁律
    globals_rules = [m["content"] for m in mems if m.get("content", "").startswith("【通用")]

    # 被回放 ≥3 次的规则加「高优先级·反复验证」标注，与普通记忆分层
    # （断点研究 2026-08-15：回放活性此前不影响 prompt 呈现 → LLM 分不清硬约束）
    def fmt(m):
        mark = "【高优先级·反复验证】" if (m.get("access_count") or 0) >= 3 else ""
        return f"  • {mark}{m['content']}" if mark else f"  • {m['content']}"
    skill_block = "\n".join(fmt(m) for m in domain_memories) if domain_memories else "  （该领域暂无蒸馏记忆，将尝试从通用知识推理）"
    route_block = router[0] if router else ""
    global_block = "\n".join(f"  {c}" for c in globals_rules)
    checklist = DOMAIN_CHECKLISTS.get(domain, []) + GENERIC_CHECKLIST
    check_block = "\n".join(f"  □ {c}" for c in checklist)

    return f"""你是一个 FoxTable 低代码平台资深开发工程师。请根据以下知识，为题目编写可直接运行的 VB.NET 代码。

【领域路由】{route_block}

【已沉淀的 FoxTable 铁律（务必遵守）】
{global_block}

【该领域已沉淀的技能记忆】
{skill_block}

【题目】{task}

【输出要求】
1. 先写 2-3 句设计思路
2. 写完整可运行的 VB.NET 代码（用 ```vbnet 代码块包围）
3. 代码中关键行加中文注释
4. 最后列 2-3 条注意事项或易踩的坑

【提交前自检清单（写代码前逐项对照，不满足就修改代码，不要只是在注意事项里复述）】
{check_block}

注意：lambda 调用必须用 .Invoke()；单行 If 不能接 ElseIf；不需要定义 Sub/Function；'字符型'=String。"""


def build_critique_prompt(domain: str, task: str, code: str, mems: list,
                          return_selected: bool = False):
    """自检验码 prompt —— 必须严格审查，不许放水

    return_selected=True 时返回 (prompt, selected)：selected 是带编号的注入
    规则列表（铁律1/2/...），供 parse_compliance 把遵守证据写回对应记忆。
    """
    # 审查端同样走统一注入窗口（/坑 全量 + /改进 新近优先 + 自适应扩容）。
    # 修复：原实现只注入「坑 + 含·的代码模板」，改进教训根本进不了审查 prompt，
    # 审查员在知识盲区会瞎编 API 签名（轮130 HttpClient 误判、沉淀毒规则）
    domain_mems = [m for m in mems
                   if m.get("kind") == "skill" and domain in m.get("content", "")]
    selected = select_injection_window(domain_mems, window=10)

    # 给每条注入规则编号（铁律1/2/...），审查回复里逐一标注是否被代码遵守。
    # 背景（2026-08-15）：遵守率此前只有五维「铁律遵守」代理分；本字段让
    # 审查员对每条规则给出代码级实证（遵守/未遵守），沉淀为 recent_obeyed 证据。
    pitfall_block = "\n".join(
        f"  • 铁律{i}: {m['content']}" for i, m in enumerate(selected, 1)
    ) if selected else "  （暂无该领域已知陷阱记录）"

    # 狐表特殊语法速查：防审查端用标准 VB.NET/.NET 规则误判狐表自研语法。
    # 背景（2026-08-15）：轮130 把官方 New HttpClient(url) 判错、轮179 把 '''Async 标记
    # 误判成「需 Async Sub」/ 要求 Imports/Dispose——全是通用 .NET 知识套狐表语境。
    fox_ctx = ("\n".join([
        "- `'''Async` 是狐表异步标记（段首注释），配合 `Await` 使用，**不需要** `Async Sub` 关键字",
        "- 狐表内置 Newtonsoft.Json：`JObject`/`JArray`/`JsonConvert` 默认可用，**不需要** Imports",
        "- 狐表自研 `HttpClient`（`New HttpClient(url)`）不是 System.Net.Http.HttpClient，非 IDisposable，**不需要** Using/Dispose",
        "- 狐表 HttpClient 方法只有 GetData/GetDataAsync/GetFile/GetFileAsync/Clone，**没有 PostData**；Method 自动判断，发 POST 也用 GetData",
        "- 狐表 Chart 控件是 C1Chart 封装：`Chart.DataSource` 是 **String 表名**（不是 DataTable）；**没有 ChartAreas 集合**（次轴用直接属性 `AxisY2`）；**没有 AxisType/YAxisType**（双图表用 `ChartType2` + `SeriesList2`）；先设 `Series.Length` 再赋值 X/Y",
        "- Chart 绑定方式（`Series.X.DataField = \"月份\"`）X 轴自动显示字符内容，**不需要** SetValueLabel/AnnoWithLabels（那是编码方式 `Series.X(i)` 的做法）",
        "- 狐表动态建表主流用法是 `DataTables.Add(dt)`（传 DataTable 对象）；给已有表设查询用 `DataTables(\"表\").SQLCommand = \"SQL\"`；**不要假定** `DataTables.Add(表名, SQL)` 重载存在（未见官方用法）",
        "- 绑定到 `Tables().DataSource` 的 DataTable 由狐表管理生命周期，**不要求**手动 Dispose",
        "- 狐表表的列用 `Tables(\"表名\")` 访问；外部数据表用 `Dim dt As New DataTable()`（标准 .NET 类）",
    ]))

    prompt = f"""你是 FoxTable 代码审查员。请**严格**审查下面这段代码，不要放水，发现错误必须明确指出。

【狐表特殊语法速查（审查时注意，勿用标准 VB.NET/.NET 规则误判）】
{fox_ctx}

【该领域已知铁律/陷阱（逐一对照代码检查）】
{pitfall_block}

【题目】{task}

【待审查代码】
{code}

## 审查清单（必须逐项检查并在评语中说明）
1. **字符型字段拼接 SQL/filter 时是否加了单引号？** 字符型字段值拼进字符串必须用 "'" & 值 & "'" 包裹。
2. **是否有 CDate()/CDate() 直接转换用户输入？** 必须用 DateTime.TryParse。
3. **是否用了错误 API 签名？** 逐行检查每个 API 调用是否符合该领域规范。
4. **资源对象（Font/Pen/StringFormat/DataTable 等）是否释放？** 必须 .Dispose() 或 Using。
5. **代码能否直接复制运行？** 有没有缺变量声明、缺对象初始化。

## 五个维度打分（每项 1-10）
1. 语法正确性：是否符合 VB.NET / FoxTable 语法
2. API 规范性：API 调用是否正确，有无写错方法签名
3. 铁律遵守：是否违反通用铁律或审查清单中的任何一条
4. 实战可用性：代码是否可直接复制运行，有无缺漏
5. 最佳实践：是否符合该领域最佳写法

## 铁律遵守核对表（必须逐条标注，这是代码级遵守证据）
对上方【该领域已知铁律/陷阱】中的每条 铁律N，检查代码是否实际遵守：
遵守情况：
  - 铁律1: 遵守（代码里 xxx 体现了）
  - 铁律2: 未遵守（代码里 xxx 违反了）
  - 铁律N: 未涉及（本轮代码没用到这条）
只允许：遵守 / 未遵守 / 未涉及 三种结论，必须逐条给出。

## 输出格式（严格按此格式，不要多写）
综合评语：（2-3 句，必须明确指出所有发现的具体错误）
审查清单结果：
  - 字符型加引号：通过/不通过（说明）
  - 日期校验：通过/不通过（说明）
  - API 签名：通过/不通过（说明）
  - 资源释放：通过/不通过（说明）
  - 可运行性：通过/不通过（说明）
铁律遵守核对表：
  - 铁律1: 遵守/未遵守/未涉及
  - 铁律2: 遵守/未遵守/未涉及
语法正确性：X/10
API 规范性：X/10
铁律遵守：X/10
实战可用性：X/10
最佳实践：X/10
改进建议：
1. 具体建议1
2. 具体建议2
3. 具体建议3"""
    if return_selected:
        return prompt, selected
    return prompt


_DIM_NAMES = ["语法正确性", "API 规范性", "铁律遵守", "实战可用性", "最佳实践"]

# 评语兜底提取：强缺陷信号（存在/缺少/未做/会导致…）命中即提取，优先级最高
_STRONG_DEFECT_RE = re.compile(
    r"(存在|缺少|缺失|未对|没有对|未做|没有做|未显式|会(导致|造成|抛|崩)|必须|不能|应该|应当|需要|建议)"
)
# 一般缺陷信号词（命中任一即视为"评语明确指出具体问题"）
_DEFECT_RE = re.compile(
    r"(错误|问题|漏|未|没有|不要|不该|"
    r"导致|崩溃|异常|风险|隐患|违规|违反|危险|不符|无效|失败|报错|坑|注意)"
)
# 纯正面表述（"代码整体正确，未发现严重错误"这类不是缺陷，不应提取）
_POSITIVE_ONLY_RE = re.compile(
    r"^(代码|整体|基本|写法|实现|逻辑|API|资源|结构)?"
    r"(整体|基本|结构|逻辑)?"
    r"(正确|无误|没问题|可行|干净|清晰|规范|合理|到位|完整|思路正确|结构清晰|"
    r"符合(规范|要求|最佳实践)|未违反|"
    r"未发现(严重|明显)?(错误|问题)|没有发现(严重|明显)?(错误|问题)|"
    r"无(严重|明显)?(错误|问题))"
)


def _fallback_improvements(overall: str, max_items: int = 3) -> list[str]:
    """综合评语兑底提取：编号建议缺失时，从评语中抽出明确指出缺陷的句子。

    评语被要求"必须明确指出所有发现的具体错误"，所以含缺陷信号的句子
    可直接沉淀为防坑铁律（如「publishDate 缺失会导致崩溃」本身即一条
    可执行的改进规则），避免轮17式知识丢失。
    """
    if not overall:
        return []
    segs: list[str] = []
    for s in re.split(r"[。；;\n]", overall):
        s = s.strip()
        if not s:
            continue
        # 长句内部按转折连词拆开——缺陷多在"但/不过"之后
        for part in re.split(r"(?<=，)(但|不过|然而|可是|只是)", s):
            part = part.strip().lstrip("，。；;但不过然而可是只是")
            if part:
                segs.append(part)
    out: list[str] = []
    for s in segs:
        if len(s) < 6:
            continue
        if _STRONG_DEFECT_RE.search(s):
            # 强缺陷信号（存在/缺少/未做/会导致…）即使夹在正面句中也提取
            out.append(s[:80])
        elif _POSITIVE_ONLY_RE.match(s):
            # 纯正面表述（"代码整体正确，未发现严重错误"）跳过
            continue
        elif _DEFECT_RE.search(s):
            out.append(s[:80])
        else:
            continue
        if len(out) >= max_items:
            break
    return out


def parse_compliance(reply: str, selected: list) -> list[dict]:
    """从审查回复解析「铁律遵守核对表」→ [{rule, content, status}]。

    status ∈ {"遵守", "未遵守", "未涉及"}。selected 是 build_critique_prompt 里
    编号的注入规则列表（铁律1/2/...）。解析不匹配的规则跳过（容错）。
    这是五维「铁律遵守」代理分的**代码级实证**：遵守/未遵守直接对应
    规则 id，可写回记忆的 recent_obeyed 计数器。
    """
    out = []
    # 找「铁律遵守核对表」段（可能叫 遵守情况/核对表）
    m = re.search(r"(?:铁律遵守核对表|遵守情况|核对表)\s*[：:]?(.*?)(?:语法正确性|API 规范性|铁律遵守：|\Z)",
                  reply, re.S)
    seg = m.group(1) if m else reply
    for i, rule in enumerate(selected, 1):
        # 匹配 "铁律N: 遵守/未遵守/未涉及"
        mm = re.search(rf"铁律{i}\s*[:：]\s*(遵守|未遵守|未涉及)", seg)
        if mm:
            out.append({"rule": i, "content": rule.get("content", ""),
                        "status": mm.group(1)})
    return out


def _apply_compliance_evidence(mems: list, compliance: list[dict]) -> int:
    """把审查核对结果写回对应记忆的 recent_obeyed 计数器。

    遵守 → recent_obeyed.obeyed +1；未遵守 → .violated +1；未涉及不计数。
    按 content 精确匹配（注入窗口里的规则即记忆本身，内容唯一）。
    返回被更新的记忆条数。这是遵守率的代码级实证来源。
    """
    updated = 0
    for c in compliance:
        if c["status"] == "未涉及":
            continue
        content = c.get("content", "")
        if not content:
            continue
        for m in mems:
            if m.get("kind") == "skill" and m.get("content") == content:
                ev = m.get("recent_obeyed") or {"obeyed": 0, "violated": 0}
                key = "obeyed" if c["status"] == "遵守" else "violated"
                ev[key] = ev.get(key, 0) + 1
                m["recent_obeyed"] = ev
                updated += 1
                break
    return updated


def parse_scores(reply: str) -> dict:
    """从 LLM 回复中解析五维分数与改进建议。

    改进建议优先取编号列表；编号列表缺失时从综合评语兑底提取
    （评语被要求必须明确指出具体错误），不让轮17式知识丢失。
    """
    import re
    scores = {}
    for dim in _DIM_NAMES:
        # 兼容 维度：8/10、维度: 8分、维度＝8 等写法
        m = re.search(rf"{dim}\s*[：:＝=]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?", reply)
        if m:
            scores[dim] = float(m.group(1))
    # 改进建议：编号列表优先
    imp = []
    for m in re.finditer(r"\d+\.\s*(.{15,200})", reply):
        s = m.group(1).strip()
        if s and "改进" not in s[:5]:
            imp.append(s)
    imp = imp[:5]
    # 综合评语
    overall = ""
    m = re.search(r"综合评语[：:]\s*(.{20,300})", reply)
    if m:
        overall = m.group(1).strip()
    # 兑底：编号列表为空但评语里有明确缺陷 → 从评语提取
    if not imp:
        imp = _fallback_improvements(overall)
    return {"scores": scores, "improvements": imp, "overall": overall}


def _core_lesson(content: str) -> str:
    """提取教训主体：去掉 [域/类型] 前缀和 题目「...」： 前缀。"""
    c = re.sub(r"^\[[^/\]]+/[^/\]]+\]\s*", "", content or "")
    c = re.sub(r"^题目「[^」]*」[：:]?\s*", "", c)
    return c.strip()


def _find_similar(mems: list, domain: str, core: str, thr: float = 0.55):
    """找同领域最相似的历史教训（相似度用 SequenceMatcher）。

    返回 (sim, memory) 或 None。用于复犯检测：同一教训第二次出现时
    自动升级为 /坑 铁律；已被 /坑 覆盖时跳过重复沉淀。
    """
    import difflib
    best = None
    for m in mems:
        if m.get("kind") != "skill":
            continue
        c = m.get("content", "")
        if domain not in c:
            continue
        r = difflib.SequenceMatcher(None, core, _core_lesson(c)).ratio()
        if r > thr and (best is None or r > best[0]):
            best = (r, m)
    return best


def persist_improvements(mems: list, domain: str, task: str, code: str, imp_list: list, scores: dict) -> int:
    """把改进沉淀为 skill 记忆；同一教训复犯时自动升级为 /坑 铁律。

    升级规则：新建议与同领域已有教训相似度 > 0.55（第二次犯）→
    把旧教训升级为 /坑（importance 提到 1.2，无条件进注入窗口）；
    若已存在 /坑 铁律覆盖 → 跳过本次重复沉淀（解决轮112式"沉淀→再沉淀"掩盖）。
    """
    import uuid, time as _t
    now = _t.time()
    added = 0
    for imp in imp_list:
        core = _core_lesson(imp)[:150]
        if not core or len(core) < 6:
            continue
        sim = _find_similar(mems, domain, core)
        if sim:
            r, old = sim
            if "/坑" in old.get("content", ""):
                # 铁律已入库，重复沉淀会稀释反而掩盖问题（轮112 现象）
                log(f"  跳过重复沉淀（「{domain}」已有 /坑 铁律覆盖，sim={r:.2f}）")
                continue
            # 复犯 → 升级为 /坑 铁律（无条件进窗口，importance 提升）
            old["content"] = f"[{domain}/坑] {_core_lesson(old.get('content', ''))[:120]}"
            old["importance"] = max(old.get("importance", 0.85), 1.2)
            old["tier"] = "warm"
            added += 1
            log(f"  ⬆ 复犯升级为 /坑 铁律（sim={r:.2f}）: {old['content'][:70]}")
            continue
        content = f"[{domain}/改进] 题目「{task[:30]}」：{imp[:150]}"
        mems.append({
            "id": str(uuid.uuid4()), "kind": "skill", "mtype": "skill",
            "content": content, "importance": 0.85, "access_count": 2,
            "last_access": now, "tier": "warm", "created_at": now,
            "history": [[now, 1.0, now, 2, 0.85]],
        })
        added += 1
    # 如果五维均>=8，把整段代码也沉淀为"优秀范例"
    if scores and all(v >= 8 for v in scores.values()):
        code_snippet = code[:300].replace("\n", " ")
        mems.append({
            "id": str(uuid.uuid4()), "kind": "skill", "mtype": "skill",
            "content": f"[{domain}/范例] 五维均>=8的优秀代码：{code_snippet}",
            "importance": 0.90, "access_count": 2,
            "last_access": now, "tier": "warm", "created_at": now,
            "history": [[now, 1.0, now, 2, 0.90]],
        })
        added += 1
    return added


def sync_ft_memory_to_store(store, mems: list, replay_rounds: int = 0,
                            round_now: float | None = None) -> int:
    """把 coder 记忆 dicts 同步进 MemoryAgent 的 store（按 id upsert）。

    事实字段（content/importance/tier）以 dict 为准随时更新；
    演化状态（access_count/last_access/history）以 store 为准——已存在的
    对象不被覆盖，避免 sleep 回放效果被旧快照冲掉。返回首次进入 store 的条数。

    背景：之前 coder 记忆直写 JSON，从不进 store → agent.sleep() 的回放/冷压缩
    候选永远为空（日志恒为 0）。此桥让每轮新沉淀的规则进入 store，sleep 才有
    候选可回放（新规则的 last_access = now，恰在 replay_window 内）。

    replay_rounds > 0 时：把**最近 replay_rounds 轮沉淀的规则**的 last_access
    刷成 now——让旧教训也能周期性进入回放窗口被再激活（默认窗口仅 1 秒，
    只有本轮新沉淀能回放，轮 123 的 .DataTable 教训沉淀后从未再被激活、
    轮 126 又复犯同一错）。按轮数而非秒数：不受休息时长（60-120s vs 300-900s）
    影响，语义稳定。
    """
    added = 0
    now = round_now if round_now is not None else time.time()
    # 最近 replay_rounds 轮沉淀的规则 id 集（按 last_access 新近排序取前 3×N 条，
    # 每轮约沉淀 3 条；用 set 判断 O(1)）
    recent_ids: set = set()
    if replay_rounds > 0:
        cand = [m for m in mems if m.get("kind") == "skill" and m.get("last_access")]
        cand.sort(key=lambda m: m.get("last_access", 0), reverse=True)
        recent_ids = {m["id"] for m in cand[: replay_rounds * 3]}
    for d in mems:
        mem_id = d.get("id")
        if not mem_id:
            continue
        m = store.get(mem_id)
        if m is None:
            m = Memory(
                id=mem_id,
                content=d.get("content", ""),
                tier=Tier(d.get("tier", "warm")),
                kind=d.get("kind", "fact"),
                mtype=MemType.SKILL if d.get("mtype") == "skill" else MemType(d.get("mtype", "semantic")),
                created_at=d.get("created_at", now),
                last_access=d.get("last_access", now),
                access_count=d.get("access_count", 0),
                importance=d.get("importance", 0.1),
            )
            if d.get("history"):
                m.history = list(d["history"])
            store._memories[mem_id] = m
            added += 1
        else:
            # 事实字段同步（演化状态保留）
            m.content = d.get("content", m.content)
            m.importance = d.get("importance", m.importance)
            m.tier = Tier(d.get("tier", m.tier.value))
        # 周期性再激活：最近 N 轮沉淀的规则刷 last_access，进入回放窗口
        if mem_id in recent_ids:
            m.last_access = now
    return added


def sync_store_to_ft_memory(store, mems: list) -> dict:
    """把 store 侧 sleep 效果（回放 access_count 等）写回 coder 记忆 dicts。

    返回 {"access_count": n, "importance": n} 变更计数。刻意不做的事：
    - 不写回 history（观测轨迹，对注入窗口无意义，写回会无限膨胀 JSON）；
    - 不把 cold 压缩写回（coder 规则 importance 全部 >= 0.85 > 0.8 阈值，
      本就不会进压缩候选池——防御性保护：规则必须保持可注入原文）；
    - 不写回 tier（sleep 的 hot 升降级不影响 select_injection_window）。
    """
    changed = {"access_count": 0, "importance": 0}
    for d in mems:
        m = store.get(d.get("id"))
        if m is None:
            continue
        if m.access_count != d.get("access_count", 0):
            d["access_count"] = m.access_count
            changed["access_count"] += 1
        if abs(m.importance - d.get("importance", 0.0)) > 1e-9:
            d["importance"] = round(m.importance, 4)
            changed["importance"] += 1
    return changed


def respond_with_retry(responder, prompt: str, *, timeout: float = 120.0,
                       min_len: int = 30, attempts: int = 5,
                       retry_delay: float = 8.0,
                       max_tokens: int | None = None) -> str:
    """调用 LLM 生成回复；空回复（reasoning-only 模型只返回思维链）自动重试。

    2026-08-16 修复：当前 sensenova provider 间歇性空返回，attempts 从 3 提升到 5，
    retry_delay 从 5s 拉到 8s。5 次全失败则抛异常，让上层 skip 本轮不污染产物。
    max_tokens：长输出（代码/核对表）必须传，默认 1024 会拦腰截断。
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            reply = responder.respond(prompt, memories=None, timeout=timeout,
                                      max_tokens=max_tokens)
            if reply and len(reply.strip()) >= min_len:
                return reply
            last_err = RuntimeError(f"LLM 回复过短（{len(reply.strip()) if reply else 0} 字符）")
        except Exception as e:
            last_err = e
        if attempt < attempts:
            log(f"  LLM 空回复/过短/异常（第 {attempt}/{attempts} 次），"
                f"{retry_delay:.0f}s 后重试：{str(last_err)[:60]}")
            time.sleep(retry_delay)
    log(f"⚠ LLM 连续 {attempts} 次异常，本轮跳过（不写产物）：{str(last_err)[:80]}")
    raise last_err or RuntimeError(f"LLM 连续 {attempts} 次空回复/异常")


def one_cycle(agent: MemoryAgent, i: int, *, do_critique: bool = True,
              force_cold: bool = False) -> bool:
    log(f"=== 第 {i} 轮 ===")

    # ① 抽题：优先上一轮截断告警的强制重练领域，其次冷门保底，最后随机
    forced_domain = _take_forced_domain()
    if forced_domain:
        domain, task = pick_task(force_domain=forced_domain)
        log(f"抽题: 领域「{domain}」题目：{task[:50]}...（强制重练·上轮截断告警）")
    elif force_cold:
        domain, task = pick_task(force_cold=True)
        counts = practice_counts()
        log(f"抽题: 领域「{domain}」题目：{task[:45]}...（冷门保底·已练{counts[domain]}次）")
    else:
        domain, task = pick_task()
        log(f"抽题: 领域「{domain}」题目：{task[:50]}...")

    # ② 加载记忆 + 拼 prompt
    mems = load_ft_memory()
    prompt = build_prompt(mems, domain, task)
    # ②' 注入有效性核对：铁律(/坑)必须全量进入生成 prompt，发现截断告警并自动修复排序重拼
    cov = verify_injection(prompt, domain, mems)
    if cov["missing"]:
        log(f"⚠ 注入截断告警: 「{domain}」{cov['missing']}/{cov['total']} 条铁律未进入 prompt，自动修复排序重拼")
        prompt = build_prompt(mems, domain, task)
        cov = verify_injection(prompt, domain, mems)
        log(f"  修复后: {cov['total'] - cov['missing']}/{cov['total']} 条铁律已注入")
    else:
        # 无条件记录（含 0 条坑规则的新领域），保证每轮都有注入核对痕迹可追踪
        log(f"注入覆盖: 坑规则 {cov['total']}/{cov['total']} 已注入")

    # ③ 生成代码（reasoning-only 空回复在轮内自动重试，不直接算失败；
    #    max_tokens 必须放宽——默认 1024 会把 vbnet 块拦腰截断）
    try:
        reply = respond_with_retry(agent.responder, prompt, timeout=120.0, min_len=50,
                                   max_tokens=CODEGEN_MAX_TOKENS)
    except Exception as e:
        log(f"生成代码失败: {e}")
        return False

    # 提取代码块（未闭合块取开栏后的全部截断代码，见 extract_code）
    code, t_status = extract_code(reply)

    # 若该领域已沉淀「禁止截断」铁律仍复现 → 高优先级告警 + 强制下轮重练
    # （同一领域连续 FORCED_RETRAIN_MAX_STREAK 轮仍截断则停止强制——系统性
    #   截断重练修不好，典型根因是输出上限/任务规模，继续强制只是烧配额）。
    truncated = t_status in ("截断（代码块未闭合）", "疑似截断", "空代码")
    if truncated and has_antitruncation_rule(domain, mems):
        if _note_forced_truncation(domain):
            _force_retrain(domain)
            log(f"⚠ 高优先级截断告警: 「{domain}」代码{t_status}（{len(code)}字符）"
                f"——该领域已沉淀「禁止截断」铁律仍复现截断，强制下轮重练该领域")
        else:
            log(f"⚠ 高优先级截断告警: 「{domain}」代码{t_status}（{len(code)}字符）"
                f"——已连续 {FORCED_RETRAIN_MAX_STREAK} 轮强制重练仍截断，"
                f"暂停强制重练转普通抽题（疑似任务规模超出模型单轮输出上限）")

    # 保存原始输出
    out_path = WORK_DIR / f"cycle_{i:03d}_{domain.replace('/','_')}.md"
    atomic_write_text(
        out_path,
        f"# 第{i}轮 · {domain}\n\n## 题目\n{task}\n\n## 生成代码\n\n{reply}\n",
        overwrite=False,
    )

    log(f"生成代码: {len(code)} 字符（{t_status}）→ {out_path.relative_to(Path(__file__).resolve().parent)}")

    # ④ 自检验
    if do_critique:
        try:
            crit_prompt, selected = build_critique_prompt(domain, task, code, mems,
                                                          return_selected=True)
            # 空回复自动重试（reasoning 模型偶发只返回思维链）；
            # 核对表 + 五维分数的输出量同样超默认 1024
            crit_reply = respond_with_retry(agent.responder, crit_prompt, timeout=60.0,
                                            min_len=30, max_tokens=CRITIQUE_MAX_TOKENS)
            parsed = parse_scores(crit_reply)
            scores = parsed["scores"]
            s_str = ", ".join(f"{k}={v:.1f}" for k, v in scores.items()) or "无分数"
            log(f"自检验: 五维 {{{s_str}}}")
            if parsed["overall"]:
                log(f"  综合: {parsed['overall'][:100]}")
            # ④' 代码级遵守证据：解析「铁律遵守核对表」，写回对应记忆的
            #     recent_obeyed 计数器（遵守 +1 / 未遵守 -1 / 未涉及不计数）。
            #     背景：遵守率此前只有五维「铁律遵守」代理分，本证据让每条规则
            #     都有「代码是否实际用了它」的实证，可统计真实遵守率。
            compliance = parse_compliance(crit_reply, selected)
            if compliance:
                n_obey = sum(1 for c in compliance if c["status"] == "遵守")
                n_viol = sum(1 for c in compliance if c["status"] == "未遵守")
                log(f"  遵守核对: {n_obey} 遵守 / {n_viol} 未遵守 / "
                    f"{len(compliance) - n_obey - n_viol} 未涉及（{len(compliance)} 条被核对）")
                _apply_compliance_evidence(mems, compliance)
            else:
                log(f"  遵守核对: 未解析到核对表（注入 {len(selected)} 条）——回复尾部: "
                    f"{crit_reply[-160:].replace(chr(10), ' ')}")
            # ⑤ 沉淀（有改进建议即可沉淀；五维分数缺失不影响——
            #    优秀范例入库另有单独门槛：scores 全部 >=8）
            if parsed["improvements"]:
                n = persist_improvements(mems, domain, task, code, parsed["improvements"], scores)
                log(f"  沉淀 {n} 条")
                for imp in parsed["improvements"][:3]:
                    log(f"    → {imp[:80]}")
                save_ft_memory(mems)
            else:
                log("自检验: 未解析到改进建议，跳过沉淀")
        except Exception as e:
            log(f"自检验异常: {e}")

    # ⑥ 睡眠：先把本轮记忆同步进 store（sleep 才有候选可回放——新规则
    #    last_access = now 恰在 replay_window 内），睡眠后把回放效果写回 JSON。
    #    背景：之前 store 恒空（sleep 三项恒 0），双向桥让睡眠巩固真正生效。
    #    replay_rounds>0 时最近 N 轮沉淀的规则也被刷 last_access 周期性再激活。
    sync_ft_memory_to_store(agent.store, mems, replay_rounds=_REPLAY_ROUNDS)
    sr = agent.sleep()
    replayed = sr.get("replayed_count", 0) if isinstance(sr, dict) else 0
    cold = sr.get("cold_compressed", 0) if isinstance(sr, dict) else 0
    evolved = len(sr.get("evolved", [])) if isinstance(sr, dict) else 0
    log(f"睡眠: 回放 {replayed} · 冷压缩 {cold} · 演化入库 {evolved}")
    changed = sync_store_to_ft_memory(agent.store, mems)
    if changed["access_count"] or changed["importance"]:
        save_ft_memory(mems)
        log(f"  sleep 效果写回: access_count 变化 {changed['access_count']} 条 · "
            f"importance 变化 {changed['importance']} 条")
    return True


# ---------- 日志完整性自检与重建（2026-08-15 事故：`>` 重定向清空过轮1-165 日志） ----------

_DIMS = ["语法正确性", "API 规范性", "铁律遵守", "实战可用性", "最佳实践"]


def _log_cycle_numbers() -> list[int]:
    """当前日志里已有的普通轮号（`=== 第 N 轮 ===`）。"""
    if not LOG_PATH.exists():
        return []
    txt = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    return [int(m) for m in re.findall(r"=== 第 (\d+) 轮 ===", txt)]


def _cycle_ordinary_rounds() -> list[tuple[int, Path]]:
    """从 cycle 文件头解析普通轮（`# 第N轮 · 领域`），排除验证轮（`XXX验证第N轮`）。"""
    out = []
    for p in WORK_DIR.glob("cycle_*.md"):
        head = p.read_text(encoding="utf-8", errors="replace")[:120]
        m = re.search(r"# 第(\d+)轮 · (.+)", head)
        if m:
            out.append((int(m.group(1)), p))
    return out


def detect_log_truncation() -> dict:
    """对比 cycle 文件（权威轮次记录）与日志轮头，检测日志是否被意外清空。

    被清空特征：日志轮头数远小于 cycle 普通轮数（<60% 且缺失 >3 轮）。
    返回 {"ok", "cycle_count", "log_count", "missing", "reason"}。
    """
    cycle = _cycle_ordinary_rounds()
    log_nums = _log_cycle_numbers()
    if not cycle:
        return {"ok": True, "cycle_count": 0, "log_count": len(log_nums),
                "missing": [], "reason": "无 cycle 文件（新环境），跳过检测"}
    cycle_nums = [n for n, _ in cycle]
    missing = sorted(set(cycle_nums) - set(log_nums))
    # 被清空特征：缺失轮多（>3）且占比高（≥30%）；写入失败只缺 1-2 轮不误报
    truncated = len(missing) > 3 and len(missing) >= 0.3 * len(cycle_nums)
    return {"ok": not truncated, "cycle_count": len(cycle_nums),
            "log_count": len(log_nums), "missing": missing,
            "reason": "日志疑似被清空" if truncated else "日志完整"}


def rebuild_log_from_sources() -> int:
    """从 cycle 文件 + coder_stdout + 恢复文档重建缺失普通轮的日志行。

    返回重建的轮数；0 = 无需重建。原则：
      - cycle 文件：轮头/抽题/代码长度（权威存在性；分数缺失不伪造）
      - coder_stdout.log：轮 160~165 完整行（含真实分数）整行复制
      - docs/log_recovery_20260815.md：轮 145~159 真实分数 → 自检验行
    重建行写回 foxtable_coder.log（重建前备份 .bak-pre-rebuild）。
    """
    det = detect_log_truncation()
    if det["ok"] or not det["missing"]:
        return 0
    missing = det["missing"]
    log(f"[警告] 检测到日志被清空（cycle {det['cycle_count']} 轮 vs 日志 {det['log_count']} 轮），"
        f"重建缺失 {len(missing)} 轮...")

    rebuilt: dict[int, list[str]] = {}

    def ts_for(p: Path) -> str:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # 1) cycle 文件 → 轮头/抽题/代码 行
    for n, p in _cycle_ordinary_rounds():
        if n not in missing:
            continue
        ts = ts_for(p)
        text = p.read_text(encoding="utf-8", errors="replace")
        mh = re.search(r"# 第\d+轮 · (.+)", text[:120])
        domain = mh.group(1).strip() if mh else p.stem.split("_", 1)[1]
        mt = re.search(r"## 题目\s*\n(.+)", text)
        task = mt.group(1).strip() if mt else ""
        lines = [f"[{ts}] === 第 {n} 轮 ==="]
        if task:
            lines.append(f"[{ts}] 抽题: 领域「{domain}」题目：{task[:50]}...（重建）")
        mcode = re.search(r"```vbnet\s*\n(.*?)\n```", text, re.S)
        if mcode:
            status = truncation_heuristic(mcode.group(1))
            lines.append(f"[{ts}] 生成代码: {len(mcode.group(1))} 字符（{status}）→ {p.name}（重建）")
        else:
            lines.append(f"[{ts}] 生成代码: 0 字符（空代码）→ {p.name}（重建）")
        rebuilt[n] = lines

    # 2) coder_stdout.log：整行复制（含真实分数）
    stdout_path = LOG_PATH.parent / "coder_stdout.log"
    if stdout_path.exists():
        n = None
        for ln in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"=== 第 (\d+) 轮 ===", ln)
            if m:
                n = int(m.group(1))
                if n in missing:
                    rebuilt.setdefault(n, []).append(ln)
            elif n in missing and n in rebuilt:
                rebuilt[n].append(ln)

    # 3) 恢复文档：轮 145~159 真实分数 → 自检验行（不伪造，只补已验证分数）
    rec = Path(__file__).resolve().parent / "docs" / "log_recovery_20260815.md"
    if rec.exists():
        rtext = rec.read_text(encoding="utf-8", errors="replace")
        m_sec = re.search(r"### 轮 145~165.*?\n(.*?)(?:\n### |\Z)", rtext, re.S)
        if m_sec:
            for ln in m_sec.group(1).splitlines():
                m = re.match(r"\|\s*(\d+)\s*\|\s*[^|]+?\s*\|\s*([^|]+?)\s*\|\s*[^|]*\s*\|", ln)
                if not m:
                    continue
                n = int(m.group(1))
                five = m.group(2).strip()
                parts = re.findall(r"\d+(?:\.\d+)?", five)[:5]
                if len(parts) == 5 and n in missing and n in rebuilt:
                    vals = [float(v) for v in parts]
                    s = ", ".join(f"{d}={v:.1f}" for d, v in zip(_DIMS, vals))
                    ts = rebuilt[n][0][1:20]
                    rebuilt[n].append(f"[{ts}] 自检验: 五维 {{{s}}}（重建）")

    # 4) 合并写回：启动信息 + 重建轮（缺失）+ 当前日志已有轮，备份后写
    cur_lines = (LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
                 if LOG_PATH.exists() else [])
    blocks: list[list[str]] = [[]]
    for ln in cur_lines:
        if re.search(r"=== 第 \d+ 轮 ===", ln):
            blocks.append([ln])
        else:
            blocks[-1].append(ln)
    miss_set = set(missing)
    cur_keep: list[str] = []
    for block in blocks:
        head = block[0] if block else ""
        m = re.search(r"=== 第 (\d+) 轮 ===", head)
        if m is None or int(m.group(1)) not in miss_set:
            cur_keep.extend(block)
    first_round = next((i for i, ln in enumerate(cur_keep)
                        if re.search(r"=== 第 \d+ 轮 ===", ln)), len(cur_keep))
    new_lines = [ln for n in sorted(missing) for ln in rebuilt.get(n, [])]
    final = cur_keep[:first_round] + new_lines + cur_keep[first_round:]
    if not final:
        return 0
    out_text = "\n".join(final) + "\n"
    if LOG_PATH.exists():
        shutil.copy2(LOG_PATH, LOG_PATH.with_suffix(".log.bak-pre-rebuild"))
    atomic_write_text(LOG_PATH, out_text)
    log(f"日志已重建: {len(missing)} 轮缺失行已从 cycle 文件/存档恢复"
        f"（备份 {LOG_PATH.name}.bak-pre-rebuild）")
    return len(missing)


# ---------- 休息间隙的规则治理（2026-08-15：把空转休息变成产出） ----------

# 狐表语境段已确认**不存在的 API**关键词——规则在「教使用」它们即毒规则。
# （有否定上下文如 不要/禁止/无/不存在 的是警告，不算毒）
# 注意：System.Net.Http.HttpClient 不在毒词表——「将 System.Net 改为狐表 HttpClient」
# 这类修正规则也会提到它，误判风险大于收益。
_POISON_PATTERNS = [
    "ChartAreas",               # 狐表 Chart 无 ChartAreas 集合（次轴用直接属性 AxisY2）
    "YAxisType",                # 无此属性
    "AxisTypeEnum.Secondary",   # 无 AxisType
    "PostData",                 # 狐表 HttpClient 无 PostData/PostDataAsync
    "Async Sub",                # 狐表用 '''Async 标记，不是 Async Sub
]
_NEGATION_RE = re.compile(r"(不要|禁止|不能|没有|不存在|勿|避免|慎用|不适用|无此|并非|不是|无需|不要求|不需要|无)")
_TRUNC_TAIL_RE = re.compile(r"([；，、(（:：、]|\.\.\.|……|等等|或$|如$|例如$|比如$)$")


def _is_truncated_rule(content: str) -> bool:
    """内容残缺规则：异常短 / 断在标点前 / 反引号不配对 / 等待续写结尾。

    /代码 /API /范例 模板是 persist 时 `code[:300]` 压缩的紧凑格式（以注释/括号
    结尾是正常形态，截断是设计行为）——只做极短检测，不做尾标点/反引号检测。
    """
    c = (content or "").strip()
    if len(c) < 15:
        return True
    if re.search(r"\[[^/\]]+/(代码|API|范例)\]", c):
        return False
    # 反引号类检测（奇数/结尾未闭合）误伤率高——引用单个词不对称、闭合引用结尾
    # 都是正常形态（2026-08-15 实测误判多条完整规则）。只认可靠的截断信号：
    # 断在标点前（...统一格式，/...设置请求头：）或等待续写词结尾（...或/...例如）。
    if _TRUNC_TAIL_RE.search(c):
        return True
    return False


def _is_poison_rule(content: str) -> bool:
    """教使用狐表语境已确认不存在的 API → 毒规则；带否定上下文的是警告不算。"""
    if _NEGATION_RE.search(content or ""):
        return False
    return any(p in (content or "") for p in _POISON_PATTERNS)


def _dedupe_rules(mems: list) -> list[tuple[str, str, float]]:
    """同领域高相似（≥0.88）skill 去重。返回 [(保留id, 删除id, 相似度)]。

    保留优先级：/坑 铁律 > /改进 > 其他；同优先级取 importance 高、内容长。
    跨领域不算重复（同一条知识在多个领域各有用处）。
    """
    groups: dict[str, list[tuple[int, dict]]] = {}
    for i, m in enumerate(mems):
        if m.get("kind") != "skill":
            continue
        c = m.get("content", "")
        # 只对 prose 规则（/改进 /坑）去重；/API /代码 是速查模板，
        # 短模板相似度天然偏高（SQLInsertFile vs SQLLoadFile 仅前缀相同），
        # 全部保留（2026-08-15 实测 16 条全误删）。
        if not re.search(r"\[[^/\]]+/(改进|坑)\]", c):
            continue
        mm = re.match(r"\[([^/\]]+)/", c)
        domain = mm.group(1) if mm else ""
        groups.setdefault(domain, []).append((i, m))
    removed: list[tuple[str, str, float]] = []
    drop: set[str] = set()
    for domain, items in groups.items():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i1, m1 = items[a]
                i2, m2 = items[b]
                if m1.get("id") in drop or m2.get("id") in drop:
                    continue
                # 用 _core_lesson 比较（去掉 [域/类型] 与 题目「」：前缀，
                # 否则 /坑 与 /改进 同内容也会因前缀差异判不相似）
                core1, core2 = _core_lesson(m1.get("content", "")), _core_lesson(m2.get("content", ""))
                r = difflib.SequenceMatcher(None, core1, core2).ratio()
                if r < 0.88:
                    continue
                def prio(m: dict) -> tuple:
                    c = m.get("content", "")
                    p = 3 if "/坑" in c else (2 if "/改进" in c else 1)
                    return (p, m.get("importance", 0.0), len(c))
                if prio(m1) >= prio(m2):
                    keep, drop_m = m1, m2
                else:
                    keep, drop_m = m2, m1
                removed.append((keep.get("id"), drop_m.get("id"), r))
                drop.add(drop_m.get("id"))
    return removed


def govern_memory(mems: list) -> dict:
    """跨领域规则治理：残缺扫描 + 毒规则检测 + 相似去重。就地过滤 mems 并返回报告。"""
    truncated = [m.get("id") for m in mems
                 if m.get("kind") == "skill" and _is_truncated_rule(m.get("content", ""))]
    poison = [m.get("id") for m in mems
              if m.get("kind") == "skill" and _is_poison_rule(m.get("content", ""))]
    drop = set(truncated) | set(poison)
    # 相似去重只在未被毒/残缺标记的规则里做
    kept = [m for m in mems if m.get("id") not in drop]
    deduped = _dedupe_rules(kept)
    drop |= {d for _, d, _ in deduped}
    details = []
    for m in mems:
        if m.get("id") in truncated and len(details) < 3:
            details.append(f"残缺: {m.get('content','')[:60]}")
    for m in mems:
        if m.get("id") in poison and len(details) < 6:
            details.append(f"毒规则: {m.get('content','')[:60]}")
    for keep, _, r in deduped:
        if len(details) < 9:
            details.append(f"去重 sim={r:.2f}")
    before = len(mems)
    mems[:] = [m for m in mems if m.get("id") not in drop]
    return {"truncated": len(truncated), "poison": len(poison),
            "deduped": len(deduped), "total": before - len(mems),
            "details": details}


# ---------- 休息间隙质量快照归档（每次休息留下一份可回溯档案） ----------

ARCHIVE_DOC = Path(__file__).resolve().parent / "docs" / "quality_archive.md"
ARCHIVE_LATEST = Path(__file__).resolve().parent / "docs" / "quality_latest.md"
REPLAY_TREND_DOC = Path(__file__).resolve().parent / "docs" / "replay_activity_trend.md"
REPLAY_TREND_EVERY = 20   # 每 20 轮追加一行回放活性时间序列


def _replay_activity_stats(log_text: str, mems: list) -> dict:
    """回放活性统计（纯函数，便于测试）：活性规则数 / 覆盖领域 / Δ铁律遵守。

    active_rules   = access_count >= 3 的 skill 规则数（被回放再激活的证据）
    active_domains = 这些活性规则覆盖的领域数（从 content 前缀提取）
    replay_delta   = 活性领域 vs 无活性领域的「铁律遵守」维度均分差
                     （解析日志每轮 铁律遵守 维度，按领域聚合后分组对比）
    与 track_coder_trend / evolution_report 同口径（fix3：access_count >= 3）。
    """
    from collections import defaultdict
    skills = [m for m in mems if m.get("kind") == "skill"]
    active_rules = sum(1 for m in skills if (m.get("access_count") or 0) >= 3)
    active_domains: set[str] = set()
    for m in skills:
        if (m.get("access_count") or 0) >= 3:
            md = re.match(r"\[(.+?)/(?:坑|改进|范例|代码|API)\]", m.get("content", ""))
            if md:
                active_domains.add(md.group(1))
    # Δ铁律遵守：日志每轮 领域 + 铁律遵守 维度，按领域聚合均分后分组
    dim_by_domain: dict[str, list[float]] = defaultdict(list)
    cur_dom: str | None = None
    for ln in log_text.splitlines():
        m = re.search(r"抽题: 领域「([^」]+)」", ln)
        if m:
            cur_dom = m.group(1)
            continue
        m = re.search(r"自检验: 五维 \{.*?铁律遵守=([\d.]+)", ln)
        if m and cur_dom:
            dim_by_domain[cur_dom].append(float(m.group(1)))
    act_dims: list[float] = []
    inact_dims: list[float] = []
    for d, dims in dim_by_domain.items():
        avg = sum(dims) / len(dims)
        (act_dims if d in active_domains else inact_dims).append(avg)
    replay_delta = None
    if act_dims and inact_dims:
        replay_delta = round(sum(act_dims) / len(act_dims)
                             - sum(inact_dims) / len(inact_dims), 2)
    # 代码级遵守证据（recent_obeyed）：遵守/未遵守 计数合计
    total_obey = sum((m.get("recent_obeyed") or {}).get("obeyed", 0) for m in skills)
    total_viol = sum((m.get("recent_obeyed") or {}).get("violated", 0) for m in skills)
    n_ev = sum(1 for m in skills if m.get("recent_obeyed"))
    return {"active_rules": active_rules, "active_domains": len(active_domains),
            "replay_delta": replay_delta, "act_n": len(act_dims),
            "inact_n": len(inact_dims),
            "obey_total": total_obey, "viol_total": total_viol, "ev_n": n_ev}


def _snapshot_data(round_no: int, govern_rep: dict | None = None) -> dict:
    """收集一份质量快照的数据（纯函数，便于测试）。

    板块：均分曲线（最近 10 轮 + 累计均分）· 领域覆盖（练习数/已覆盖领域）
    · 注入核对（注入覆盖轮数/告警计数）· 治理结果 · 记忆规模 · 回放活性。
    从主日志解析，不依赖 track_coder_trend 的全源扫描（休息间隙要快）。
    """
    log_text = LOG_PATH.read_text(encoding="utf-8", errors="replace") if LOG_PATH.exists() else ""
    # --- 均分曲线：按轮头切段，取每段首条五维均分 ---
    recent: list[tuple[int, float]] = []
    cur: int | None = None
    cur_avg: float | None = None
    for ln in log_text.splitlines():
        m = re.search(r"=== 第 (\d+) 轮 ===", ln)
        if m:
            if cur is not None and cur_avg is not None:
                recent.append((cur, cur_avg))
            cur = int(m.group(1)); cur_avg = None
            continue
        if cur is None:
            continue
        if cur_avg is None and "自检验: 五维" in ln:
            nums = [float(x) for x in re.findall(r"=(\d+\.\d)", ln)]
            if len(nums) >= 3:
                cur_avg = sum(nums) / len(nums)
    if cur is not None and cur_avg is not None:
        recent.append((cur, cur_avg))
    recent.sort()
    last10 = recent[-10:]
    all_avgs = [a for _, a in recent]
    # --- 均分曲线：移动平均（MA5，最近 20 轮）+ 拐点检测 ---
    last20 = recent[-20:]
    ma5: list[tuple[int, float]] = []
    vals = [a for _, a in last20]
    for i in range(len(vals)):
        lo = max(0, i - 4)
        ma5.append((last20[i][0], round(sum(vals[lo:i + 1]) / (i - lo + 1), 2)))
    # 拐点：zigzag 折线法——从上一个极值反方向移动 ≥ 阈值才确认拐点。
    # 抗平台（上升段顶部打平再回落）与单轮抖动；阈值为 0.8 或窗口幅度 20% 取大。
    pivots: list[tuple[int, float, str]] = []
    if len(ma5) >= 3:
        # 阈值从平滑序列自身计算（原始分数含单轮极值，会让阈值过大而永不触发）
        sm_vals = [a for _, a in ma5]
        rng = (max(sm_vals) - min(sm_vals)) if sm_vals else 0.0
        thr = max(0.8, 0.2 * rng)
        last = (0, ma5[0][1], 1)  # (idx, value, dir) dir=1 向上跟踪找顶, -1 向下跟踪找底
        trend = 1
        for i in range(1, len(ma5)):
            _, v = ma5[i]
            if trend >= 0:  # 向上跟踪：创新高则更新，回撤 ≥ thr 确认顶
                if v >= last[1]:
                    last = (i, v, 1)
                elif last[1] - v >= thr:
                    pivots.append((ma5[last[0]][0], last[1], "顶"))
                    last = (i, v, -1)
                    trend = -1
            else:  # 向下跟踪：创新低则更新，反弹 ≥ thr 确认底
                if v <= last[1]:
                    last = (i, v, -1)
                elif v - last[1] >= thr:
                    pivots.append((ma5[last[0]][0], last[1], "底"))
                    last = (i, v, 1)
                    trend = 1
    # --- 最近治理/截断告警轮次（拐点标注）---
    last_govern_round: int | None = None
    last_trunc_round: int | None = None
    cur_r: int | None = None
    for ln in log_text.splitlines():
        m = re.search(r"=== 第 (\d+) 轮 ===", ln)
        if m:
            cur_r = int(m.group(1))
            continue
        if re.search(r"规则治理: 去重", ln) and cur_r is not None:
            last_govern_round = cur_r
        if re.search(r"高优先级截断告警", ln) and cur_r is not None:
            last_trunc_round = cur_r
    # --- 领域覆盖 ---
    counts = practice_counts()
    practiced = {d: c for d, c in counts.items() if c > 0}
    # --- 注入核对 ---
    inject_ok = len(re.findall(r"注入覆盖:", log_text))
    inject_alerts = len(re.findall(r"注入截断告警", log_text))
    trunc_alerts = len(re.findall(r"高优先级截断告警", log_text))
    forced_runs = len(re.findall(r"强制重练·上轮截断告警", log_text))
    # --- 记忆规模 ---
    mems = load_ft_memory()
    skills = [m for m in mems if m.get("kind") == "skill"]
    pitfalls = sum(1 for m in skills if "/坑" in m.get("content", ""))
    access3 = sum(1 for m in skills if (m.get("access_count") or 0) >= 3)
    replay = _replay_activity_stats(log_text, mems)
    return {
        "round": round_no,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recent": last10,
        "ma5": ma5,
        "pivots": pivots,
        "last_govern_round": last_govern_round,
        "last_trunc_round": last_trunc_round,
        "all_avg": (sum(all_avgs) / len(all_avgs)) if all_avgs else None,
        "all_n": len(all_avgs),
        "practiced_n": len(practiced),
        "total_domains": len(TASK_POOL),
        "least": sorted(practiced.items(), key=lambda kv: kv[1])[:5],
        "inject_ok": inject_ok,
        "inject_alerts": inject_alerts,
        "trunc_alerts": trunc_alerts,
        "forced_runs": forced_runs,
        "govern": govern_rep,
        "mems": len(mems),
        "skills": len(skills),
        "pitfalls": pitfalls,
        "access3": access3,
        "active_rules": replay["active_rules"],
        "active_domains": replay["active_domains"],
        "replay_delta": replay["replay_delta"],
        "act_n": replay["act_n"],
        "inact_n": replay["inact_n"],
        "obey_total": replay["obey_total"],
        "viol_total": replay["viol_total"],
        "ev_n": replay["ev_n"],
    }


def _snapshot_md(d: dict) -> str:
    """把快照数据渲染成 markdown 段落。"""
    lines = [f"## 快照 @ 轮 {d['round']}（{d['time']}）", ""]
    # 均分曲线
    if d["recent"]:
        curve = " · ".join(f"轮{n}:{a:.1f}" for n, a in d["recent"])
        lines.append(f"**均分曲线**（最近 {len(d['recent'])} 轮）: {curve}")
    # 移动平均趋势线（MA5，最近 20 轮）+ 拐点标注
    if d.get("ma5"):
        ma_curve = " → ".join(f"轮{n}:{a:.1f}" for n, a in d["ma5"])
        lines.append(f"**MA5 趋势线**: {ma_curve}")
    if d.get("pivots"):
        pv = "、".join(f"轮{n} {tag}({a:.1f})" for n, a, tag in d["pivots"])
        lines.append(f"  MA5 拐点: {pv}")
    lines.append(f"**累计均分**: {d['all_avg']:.2f}（{d['all_n']} 轮）" if d["all_avg"] is not None
                  else "**累计均分**: 暂无")
    # 领域覆盖
    lines.append(f"**领域覆盖**: {d['practiced_n']}/{d['total_domains']} 个领域已练习")
    if d["least"]:
        least = "、".join(f"{dom}({c})" for dom, c in d["least"])
        lines.append(f"  最少练习: {least}")
    # 注入核对
    lines.append(f"**注入核对**: 覆盖 {d['inject_ok']} 轮 · 注入截断告警 {d['inject_alerts']} · "
                 f"代码截断告警 {d['trunc_alerts']} · 强制重练 {d['forced_runs']} 轮")
    # 最近治理/截断轮次（质量拐点标注）
    mark = []
    if d.get("last_govern_round"):
        mark.append(f"最近规则治理 轮{d['last_govern_round']}")
    if d.get("last_trunc_round"):
        mark.append(f"最近截断告警 轮{d['last_trunc_round']}")
    if mark:
        lines.append(f"**拐点标注**: {' · '.join(mark)}")
    # 治理
    if d["govern"] and d["govern"].get("total", 0) > 0:
        g = d["govern"]
        lines.append(f"**规则治理**: 去重 {g.get('deduped', 0)} · 毒规则 {g.get('poison', 0)} · "
                     f"残缺 {g.get('truncated', 0)}（共清理 {g.get('total', 0)} 条）")
    # 记忆规模
    lines.append(f"**记忆规模**: {d['mems']} 条（skill {d['skills']} · 其中 /坑 {d['pitfalls']} · "
                 f"被回放≥3次 {d['access3']}）")
    # 回放活性
    delta_txt = (f"Δ铁律遵守 {d['replay_delta']:+.2f}"
                 if d["replay_delta"] is not None else "Δ铁律遵守 样本不足")
    obey_txt = (f"代码级遵守 {d['obey_total']} 遵守 / {d['viol_total']} 未遵守"
                f"（{d['ev_n']} 条规则有证据）" if d["obey_total"] or d["viol_total"]
                else "代码级遵守 暂无证据")
    lines.append(f"**回放活性**: 规则 {d['active_rules']} · 覆盖 {d['active_domains']} 领域 · "
                 f"{delta_txt}（活性{d['act_n']} vs 无活性{d['inact_n']}）· {obey_txt}")
    lines.append("")
    return "\n".join(lines)


def append_replay_trend(d: dict) -> Path | None:
    """每 REPLAY_TREND_EVERY 轮追加一行回放活性时间序列到 replay_activity_trend.md。

    幂等：同一 20 轮桶只记一次（按轮号检查是否已记录）。返回文件路径，
    非记录轮返回 None。行格式：| 轮 | 时间 | 活性规则 | 覆盖领域 | Δ铁律遵守 | 活性n | 无活性n |。
    """
    r = d.get("round", 0)
    if r <= 0 or r % REPLAY_TREND_EVERY != 0:
        return None
    REPLAY_TREND_DOC.parent.mkdir(parents=True, exist_ok=True)
    header = ("# 回放活性时间序列（每 20 轮一行，长周期趋势）\n\n"
              "| 轮 | 时间 | 活性规则 | 覆盖领域 | Δ铁律遵守 | 活性n | 无活性n |\n"
              "|---|---|---|---|---|---|---|\n")
    old = REPLAY_TREND_DOC.read_text(encoding="utf-8") if REPLAY_TREND_DOC.exists() else ""
    if not old:
        old = header
    elif f"| {r} |" in old:
        return REPLAY_TREND_DOC          # 幂等：该轮已记录
    delta_txt = f"{d['replay_delta']:+.2f}" if d.get("replay_delta") is not None else "样本不足"
    line = (f"| {r} | {d.get('time', '')} | {d.get('active_rules', 0)} | "
            f"{d.get('active_domains', 0)} | {delta_txt} | "
            f"{d.get('act_n', 0)} | {d.get('inact_n', 0)} |\n")
    REPLAY_TREND_DOC.write_text(old + line, encoding="utf-8")
    return REPLAY_TREND_DOC


def snapshot_quality_archive(round_no: int, govern_rep: dict | None = None) -> Path:
    """休息间隙归档一份质量快照：追加到 quality_archive.md（历史累积），
    同时覆盖写 quality_latest.md（最新状态），每 20 轮追加回放活性时间序列。
    返回 latest 文件路径。"""
    d = _snapshot_data(round_no, govern_rep)
    md = _snapshot_md(d)
    ARCHIVE_DOC.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_DOC.exists():
        ARCHIVE_DOC.write_text(ARCHIVE_DOC.read_text(encoding="utf-8") + md, encoding="utf-8")
    else:
        ARCHIVE_DOC.write_text(f"# 质量快照归档\n\n{md}", encoding="utf-8")
    atomic_write_text(ARCHIVE_LATEST, f"# 最新质量快照（轮 {round_no}）\n\n{md}", overwrite=True)
    try:
        append_replay_trend(d)
    except Exception:
        pass
    return ARCHIVE_LATEST


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="FoxTable 自主进化 agent")
    parser.add_argument("--cycles", type=int, default=10, help="循环轮数（默认 10；0=无限）")
    parser.add_argument("--min-interval", type=float, default=300.0, help="轮间最小休息秒数")
    parser.add_argument("--max-interval", type=float, default=900.0, help="轮间最大休息秒数")
    parser.add_argument("--no-critique", action="store_true", help="关闭自检验")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="连续生成失败达到该次数后熔断退出（默认 3）")
    parser.add_argument("--cold-every", type=int, default=5,
                        help="每 N 轮强制抽一次练习最少的冷门领域（默认 5；0=关闭）")
    parser.add_argument("--coverage-every", type=int, default=20,
                        help="每 N 轮重写 docs/foxtable-domain-coverage.md（默认 20；0=关闭）")
    parser.add_argument("--govern-every", type=int, default=5,
                        help="每 N 轮在休息间隙做一次规则治理（残缺/毒规则/相似去重；默认 5；0=关闭）")
    parser.add_argument("--archive-every", type=int, default=1,
                        help="每 N 轮在休息间隙归档一份质量快照到 docs/（均分曲线/领域覆盖/注入核对；默认 1；0=关闭）")
    parser.add_argument("--replay-rounds", type=int, default=0,
                        help="回放窗口轮数：>0 时最近 N 轮沉淀的规则也在 sleep 前被再激活"
                        "（默认 0=仅本轮新沉淀；研究用 5/10 对比旧教训周期性回放效果）")
    parser.add_argument("--write-coverage", action="store_true",
                        help="立即生成领域覆盖文档后退出（不进入主循环）")
    args = parser.parse_args()

    if args.write_coverage:
        generate_coverage_doc(args.coverage_every)
        log(f"领域覆盖文档已生成: {COVERAGE_DOC}")
        return 0

    if not FT_MEM_PATH.exists():
        log(f"错误: {FT_MEM_PATH} 不存在，请先运行 foxtable_build.py")
        return 1

    instance_lock = FileLock(str(FT_MEM_PATH) + ".autonomous.lock", timeout=0.0)
    try:
        instance_lock.acquire()
    except LockTimeoutError:
        log("已有 FoxTable 自主进程在运行，本次启动退出。")
        return 2

    # 日志完整性自检：若日志被意外清空（对比 cycle 文件），自动从存档重建缺失轮。
    # （2026-08-15 事故：`>` 重定向清空过轮1-165 日志，此检查让事故不再静默发生）
    try:
        rebuild_log_from_sources()
    except Exception as e:
        log(f"日志自检/重建异常（不影响运行）: {e}")

    global _REPLAY_ROUNDS
    _REPLAY_ROUNDS = max(0, args.replay_rounds)
    if _REPLAY_ROUNDS:
        log(f"回放窗口: 最近 {_REPLAY_ROUNDS} 轮沉淀的规则将周期性再激活（实验策略 B）")

    # 用空 store 实例化 agent（不碰小说的记忆文件），但需要加载 LLM responder
    from memagent.memory import MemoryStore
    from memagent.responder import LLMResponder
    store = MemoryStore()  # 内存 store，不写盘
    # FoxTable 专用 persona：明确自己是 FoxTable 开发专家，不走小说路线
    responder = LLMResponder(persona="FoxTable 低代码开发专家")
    agent = MemoryAgent(store=store, responder=responder, cfg=AgentConfig(evolve_on_sleep=False))

    # 启动时把已有记忆全量同步进 store：agent.sleep() 的回放/冷压缩候选来自
    # store.all()，不同步则 store 恒空、睡眠巩固恒 0（之前日志三项全 0 的根因）。
    try:
        n_synced = sync_ft_memory_to_store(store, load_ft_memory())
        log(f"记忆已同步进 MemoryAgent store: {n_synced} 条")
    except Exception as e:
        log(f"启动记忆同步异常（不影响运行）: {e}")

    log(f"FoxTable 自主进化 agent 启动: {'无限循环' if args.cycles <= 0 else str(args.cycles)+' 轮'}，"
        f"轮间休息 {args.min_interval:.0f}~{args.max_interval:.0f}s，"
        f"自检验={'开' if not args.no_critique else '关'}，"
        f"冷门保底={'每'+str(args.cold_every)+'轮' if args.cold_every > 0 else '关'}")
    log(f"FoxTable 记忆: {FT_MEM_PATH}")
    log(f"小说记忆（不触碰）: E:\\神经网络\\agent_memory.json")
    log(f"覆盖文档: 每 {args.coverage_every} 轮重写 {'开' if args.coverage_every > 0 else '关'}")

    i = 0
    cycle_number = next_cycle_number()
    consecutive_failures = 0
    try:
        while args.cycles <= 0 or i < args.cycles:
            i += 1
            ok = False
            try:
                force_cold = args.cold_every > 0 and i % args.cold_every == 0
                ok = one_cycle(agent, cycle_number, do_critique=not args.no_critique,
                               force_cold=force_cold)
                if args.coverage_every > 0 and i % args.coverage_every == 0:
                    generate_coverage_doc(args.coverage_every)
                    log(f"领域覆盖文档已更新（第 {i} 轮）: {COVERAGE_DOC}")
                # 休息间隙规则治理：残缺/毒规则/相似去重（把空转休息变成产出）
                govern_rep: dict | None = None
                if args.govern_every > 0 and i % args.govern_every == 0:
                    try:
                        mems_now = load_ft_memory()
                        rep = govern_memory(mems_now)
                        if rep["total"] > 0:
                            save_ft_memory(mems_now)
                        log(f"规则治理: 去重 {rep['deduped']} · 毒规则 {rep['poison']} · "
                            f"残缺 {rep['truncated']}（共清理 {rep['total']} 条）")
                        for d in rep["details"]:
                            log(f"    → {d}")
                        govern_rep = rep
                    except Exception as e:
                        log(f"规则治理异常（不影响运行）: {e}")
                # 休息间隙质量快照归档：每次休息留下一份可回溯的质量档案
                if args.archive_every > 0 and i % args.archive_every == 0:
                    try:
                        path = snapshot_quality_archive(cycle_number, govern_rep)
                        log(f"质量快照已归档: {path.relative_to(Path(__file__).resolve().parent)}")
                    except Exception as e:
                        log(f"质量快照归档异常（不影响运行）: {e}")
            except Exception as e:
                import traceback
                log(f"第 {i} 轮异常: {e}")
                log("".join(traceback.format_exc(limit=8).splitlines(keepends=True)))
            cycle_number += 1
            if ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                log(f"连续生成失败 {consecutive_failures}/{args.max_consecutive_failures}")
                if consecutive_failures >= max(1, args.max_consecutive_failures):
                    log("触发失败熔断，停止后台循环以保护 API 额度和产物。")
                    break
            if args.cycles <= 0 or i < args.cycles:
                rest = random.uniform(args.min_interval, args.max_interval)
                log(f"休息 {rest:.0f}s ...")
                time.sleep(rest)
    except KeyboardInterrupt:
        log("收到中断，退出。")
    finally:
        instance_lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
