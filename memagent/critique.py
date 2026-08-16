"""章节自评与对标抓取模块（CritiqueModule）——写完一章立刻触发的反思循环。

三件事：
1. benchmark_samples()   —— 从公开小说站抓对标章节片段（双轨：wcshuba 直链 + 内置风格库）
2. self_critique()       —— 五维自评 + 对标对比 + 给具体改进点
3. 改进点用 remember_skill 沉淀进技能记忆（MemType.SKILL，遗忘最慢），下次写作自动吸收

对标策略（双轨）：
  - 主轨：直连 www.wcshuba.com（无错书吧），该站由 yckceo 书源目录收录、公开可抓，
    首页热门玄幻/高武可直接拿到章节正文；
  - 副轨：内置 5 作家标杆风格库（我吃西红柿/耳根/江南/烽火戏诸侯/猫腻），
    用散文式技法描述替代抓不到的章节正文——比假装抓到更诚实；
  - 兜底：两条都失败时降级为通用规则模板，永远不抛错，绝不拖垮自主循环。

设计映射：agent 不只是"写完就忘"，而是写完→自评→对标→沉淀→下次写作自动带上改进，
形成一个不断变强的成长闭环。
"""
from __future__ import annotations

import html as _html
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .compat import call_responder

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_TIMEOUT = 15.0
_WCSHUBA_BASE = "https://www.wcshuba.com"
# 自主评价依据的持久追踪：每次自评实际用了什么对标样本都记录在这里
_BENCH_TRACE = Path(__file__).resolve().parent.parent / "docs" / "benchmark_trace.md"

# ---- 内置 5 作家标杆风格库（副轨） ----
# 这是 prose 式的写作技法描述，本身就能当对标参照喂给自评，
# 比假装抓到了章节正文更诚实。
_BUILTIN_STYLE_LIBRARY = [
    {
        "author": "我吃西红柿", "work": "吞噬星空",
        "style": (
            "主视角第三人称限知叙事，动作场面用短句快切，每 3-5 句必有一次镜头切换；"
            "数值化修炼进程（境界、修为、异火序号、能量刻度），让读者对主角成长有量化感知；"
            "情绪高度克制，主角在重大转折后先做 1-2 个具体动作（拍灰、握拳、抬头看天）再表达内心；"
            "战斗前后常留 2-3 句环境收束（风吹、云散、鸟鸣），形成呼吸感；"
            "大段心理活动拆成'感知→判断→抉择'三步，不整段独白。"
        ),
    },
    {
        "author": "耳根", "work": "仙逆",
        "style": (
            "苍凉氛围词密度极高（孤、逆、道、劫、缘、尘），善用独白式内心独白与宿命反问句；"
            "节奏偏慢，长段落铺陈 + 一句短收形成情绪落差；"
            "主角每次重大抉择前后必有一次与环境或旧物的物理互动（抚剑、望月、拈花），"
            "用'物'承担情绪，不让主角直接喊出来；"
            "对话少而重，配角开口常有'一语点破'的效果，一句话推进整段关系。"
        ),
    },
    {
        "author": "江南", "work": "龙族",
        "style": (
            "都市与奇幻在同一场景内交织，对话承担大量信息推进，一句话说两件事（表面闲聊+暗藏情报）；"
            "幽默与悲剧在相邻段落内切换，前一章搞笑、下一章猝然沉重；"
            "伏笔靠'随口一提'埋下——看似无关的细节，在 10 章后突然回收；"
            "主角常有'中二式'内心 OS，但 OS 越浮夸，现实处境越沉重，形成反差张力。"
        ),
    },
    {
        "author": "烽火戏诸侯", "work": "雪中悍刀行",
        "style": (
            "群像戏为主，每个配角登场都有一句'定调台词'锁定其人设，此后全程不越界；"
            "文白夹杂——雅言用于正式场合、江湖白话用于私下场景，切换自然；"
            "重场景镜头感，一段打斗或一场对话像电影分镜：全景→中景→特写→空镜；"
            "环境描写不只是背景，而是情绪的镜像（雪大=情重、风急=势危）。"
        ),
    },
    {
        "author": "猫腻", "work": "庆余年",
        "style": (
            "权谋叙事，对话中藏机锋，一句看似平常的客套话暗含三个信息层；"
            "叙事者口吻带轻微戏谑，用反讽和旁白调侃主角处境，形成'全知视角+主角限知'的张力；"
            "悬念靠信息差而非靠暴力转折——读者知道的多于角色，焦虑感来自'什么时候会穿帮'；"
            "章节结尾常留一个'意料之外的日常细节'作钩子，比大场面钩子更耐回味。"
        ),
    },
]

_RUSTLE_FALLBACK = [
    "我吃西红柿·吞噬星空【规则模板】: 主视角第三人称限知，动作短句快切，"
    "数值化修炼进程，情绪克制，战斗前后留 2-3 句环境收束。",
    "耳根·仙逆【规则模板】: 苍凉氛围词密度高，独白式内心独白与宿命反问句，"
    "长段落铺陈 + 一句短收。",
    "江南·龙族【规则模板】: 都市与奇幻交织，对话承担信息推进，"
    "幽默与悲剧切换，伏笔靠'随口一提'埋下。",
    "烽火戏诸侯·雪中悍刀行【规则模板】: 群像戏，配角一语定调，"
    "文白夹杂，重场景镜头感。",
    "猫腻·庆余年【规则模板】: 权谋叙事，对话藏机锋，"
    "叙事者戏谑，悬念靠信息差。",
]


# ---------- 无代理 fetch 客户端 ----------

_SSL_CTX = ssl.create_default_context()
_HTTPS_HANDLER = urllib.request.HTTPSHandler(context=_SSL_CTX)
_OPENER = urllib.request.build_opener(_HTTPS_HANDLER)
_OPENER.addheaders = [("User-Agent", _BROWSER_UA)]


def _fetch(url: str, timeout: float = _TIMEOUT) -> str:
    """无代理直接请求，绕开本机金融代理。"""
    with _OPENER.open(url, timeout=timeout) as r:
        data = r.read()
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _text_of(html_block: str) -> str:
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_block))).strip()


def _extract_chapter_text(page_html: str, min_chars: int = 500) -> str:
    """从小说章节页提取正文。兼容 wcshuba 及其他站常见容器。"""
    for pat in [
        r'<div[^>]*id=["\'](?:content|text|main|txt|chapter|textwrap)["\'][^>]*>([\s\S]{500,30000})',
        r'<div[^>]*class=["\'](?:showtxt|read-content|readcontent|chapter-content|readBox|chapter_content|book-read|read_box)["\'][^>]*>([\s\S]{500,30000})',
    ]:
        m = re.search(pat, page_html)
        if m:
            t = _text_of(m.group(1))
            t = re.sub(r"(版权声明|加入书签|章节目录|本章完|投推荐票|推荐阅读|广告)", "", t)
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) >= min_chars:
                return t
    paras = re.findall(r"<p[^>]*>(.*?)</p>", page_html, re.S)
    joined = "".join(_text_of(p) for p in paras)
    joined = re.sub(r"(版权声明|加入书签|章节目录|本章完)", "", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined if len(joined) >= min_chars else ""


# ---------- 主轨：wcshuba 直链 ----------

def _get_wcshuba_home_books() -> list[tuple[str, str, str]]:
    """抓 wcshuba 首页热门小说，返回 [(bid, title, href), ...]。"""
    try:
        html = _fetch(f"{_WCSHUBA_BASE}/")
        items = re.findall(
            r'href=["\'](/book/(\d+)\.html)["\'][^>]*>([^<]{2,30})<', html
        )
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str, str]] = []
        for href, bid, title in items:
            t = title.strip()
            key = (bid, t)
            if key in seen:
                continue
            seen.add(key)
            out.append((bid, t, href))
        return out
    except Exception:
        return []


def _fetch_wcshuba_chapter(book_id: str, chapter_path: str) -> str:
    """从 wcshuba 抓单章正文。"""
    url = _WCSHUBA_BASE + chapter_path if chapter_path.startswith("/") else _WCSHUBA_BASE + "/" + chapter_path
    try:
        html = _fetch(url)
        return _extract_chapter_text(html, min_chars=800)
    except Exception:
        return ""


def _get_wcshuba_chapter(book_id: str) -> str:
    """拿 wcshuba 一本热门小说的第一章正文。"""
    try:
        html = _fetch(f"{_WCSHUBA_BASE}/book/{book_id}.html")
        m = re.search(r'href=["\'](/read/' + re.escape(book_id) + r'/(\d+)\.html)["\']', html)
        if not m:
            return ""
        return _fetch_wcshuba_chapter(book_id, m.group(1))
    except Exception:
        return ""


# ---------- 副轨：内置风格库 ----------

def _pick_builtin(n: int = 3) -> list[dict]:
    """按作品题材挑选对标作家（默认全 5 家，取前 n）。"""
    import random

    pool = _BUILTIN_STYLE_LIBRARY.copy()
    random.shuffle(pool)
    return pool[:n]


# ---------- BenchmarkSample ----------

@dataclass
class BenchmarkSample:
    author: str
    work: str
    chapter: str | None = None
    text: str = ""
    from_net: bool = False
    source: str = ""  # "wcshuba" | "builtin" | "fallback"
    note: str = ""

    @property
    def preview(self) -> str:
        if self.text:
            return self.text[:400]
        return self.note[:400]


def _trace_benchmark(samples: list[BenchmarkSample]) -> None:
    """把本次自评实际使用的对标样本追加到 docs/benchmark_trace.md。

    这是"自主找评价依据"的持久证据：记录网络真实章节(wcshuba)/内置风格库/兜底模板
    各用了多少、用了哪些作品——让自评到底拿什么做参照可回看、可统计。
    任何异常都不允许拖垮自主循环。
    """
    try:
        from datetime import datetime

        net = [s for s in samples if s.source == "wcshuba"]
        works = "、".join(f"{s.work}({s.source})" for s in samples) or "无"
        line = (f"- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"| 真实网络 {len(net)}/{len(samples)} | {works}")
        _BENCH_TRACE.parent.mkdir(parents=True, exist_ok=True)
        with open(_BENCH_TRACE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def benchmark_samples(n: int = 3) -> list[BenchmarkSample]:
    """抓 n 个对标样本。主轨 wcshuba，副轨内置库，最后规则模板兜底。"""
    got: list[BenchmarkSample] = []
    used_authors: set[str] = set()

    # --- 主轨：wcshuba ---
    books = _get_wcshuba_home_books()
    for bid, title, _ in books:
        if len(got) >= n:
            break
        body = _get_wcshuba_chapter(bid)
        if body:
            # 从书名里尽量推断作者
            author = "佚名"
            m = re.match(r"(.+?)[著撰].*", title)
            if m:
                author = m.group(1)
            got.append(BenchmarkSample(
                author=author, work=title, chapter=f"/book/{bid}",
                text=body, from_net=True, source="wcshuba",
            ))
            used_authors.add(author)

    # --- 副轨：内置风格库 ---
    for entry in _pick_builtin(n):
        if len(got) >= n:
            break
        if entry["author"] in used_authors:
            continue
        got.append(BenchmarkSample(
            author=entry["author"], work=entry["work"],
            text=entry["style"], from_net=False, source="builtin",
            note=f"[内置风格库] {entry['author']}·{entry['work']}：{entry['style']}",
        ))

    # --- 兜底：规则模板 ---
    if not got:
        for line in _RUSTLE_FALLBACK[:n]:
            author, rest = line.split("·", 1)
            work, rest = rest.split("【", 1)
            got.append(BenchmarkSample(
                author=author.strip(), work=work.strip(),
                note=rest, source="fallback",
            ))
    got = got[:n]
    _trace_benchmark(got)  # 持久记录本次实际使用的评价依据
    return got


# ---------- 自评 ----------

@dataclass
class Critique:
    chapter: int
    title: str
    scores: dict = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    overall: str = ""
    benchmark_note: str = ""


_DIMENSIONS = [
    ("文风一致", "风格与既有设定、人物语气是否统一，有无跳戏或 AI 味过重"),
    ("节奏", "章节内起承转合、段落长短交替、悬念落点是否舒服"),
    ("伏笔回收", "是否埋下新伏笔 / 回收旧伏笔，呼应设定档案"),
    ("露骨场景分寸", "若含亲密/性描写：是否贴合人物弧线与情境、"
                     "是否守住人设铁律（成年+自愿，无未成年、无非自愿暗示）"),
    ("人物弧光", "主角或配角是否在本章有可感知的变化、抉择或代价"),
]


def _build_critique_prompt(chapter_text: str, chapter_no: int,
                           title: str, samples: list[BenchmarkSample],
                           persona_sheet: str | None) -> str:
    dims_block = "\n".join(
        f"  {i}. **{name}**：{desc}" for i, (name, desc) in enumerate(_DIMENSIONS, 1)
    )
    sheet = persona_sheet or "（无设定档案）"

    sample_parts = []
    for s in samples:
        if s.source == "wcshuba":
            flag = "[真实章节片段·无错书吧]"
        elif s.source == "builtin":
            flag = "[内置风格技法库]"
        else:
            flag = "[规则模板兜底]"
        sample_parts.append(f"### {s.author}·《{s.work}》{flag}\n{s.preview}\n")
    sample_block = "\n".join(sample_parts)

    return f"""【章节自评任务】你是严苛的文学编辑，正在为作家'夜航墨客'新作《{title}》第{chapter_no}章做复盘。

—— 待评章节全文 ——
{chapter_text[:6000]}

—— 当前设定档案 ——
{sheet}

—— 对标参照（学习对象，不是抄袭；含真实章节与技法库）——
{sample_block}

## 请严格按以下五维逐条评分（1-10），每维给一句依据：
{dims_block}

## 对标差距分析：
对照上面的对标参照，你的章节在哪些地方弱于他们？给出 3-5 条具体差距（不要泛泛而谈，要落到"句式/镜头/情绪节奏/伏笔手法"这一层）。

## 具体改进建议：
给出 3-5 条"下一章写作时应执行"的可操作规则，每条以「改进：」开头，一句话，够具体，下次写作时能直接照做。

## 亮点：
给出 1-3 条"这一章做得好的地方"，以「亮点：」开头，用于保留。

## 综合评语：
一句话总结这一章的整体质量与下一章最重要的 1 个改进方向。

只输出上述结构的内容，不要多余寒暄。"""


def _parse_critique(reply: str, chapter_no: int, title: str) -> Critique:
    crit = Critique(chapter=chapter_no, title=title)
    # 预处理：去 markdown 粗体，归一全角等号，容忍"**1. 文风一致：9.0**"等漂移格式
    plain = reply.replace("**", "").replace("＝", "=")
    for name, _ in _DIMENSIONS:
        pat = rf"{re.escape(name)}\s*[:：=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?\s*分?"
        matches = re.findall(pat, plain)
        if matches:
            last = matches[-1]
            try:
                crit.scores[name] = float(last)
            except ValueError:
                pass
    for line in reply.splitlines():
        low = line.strip()
        low = re.sub(r"^[–\-•\*]+\s*\*{0,2}", "", low).strip()
        if low.startswith("改进：") or low.startswith("改进:"):
            body = low.split("：", 1)[-1].split(":", 1)[-1]
            body = re.sub(r"\*{1,2}\s*$", "", body).strip()
            body = re.sub(r"^\*{1,2}\s*", "", body).strip()
            if body:
                crit.improvements.append(body)
        if low.startswith("亮点：") or low.startswith("亮点:"):
            body = low.split("：", 1)[-1].split(":", 1)[-1]
            body = re.sub(r"\*{1,2}\s*$", "", body).strip()
            body = re.sub(r"^\*{1,2}\s*", "", body).strip()
            if body:
                crit.strengths.append(body)
    gap_section = re.search(r"对标差距[^#]*?(?:\n##|\Z)", reply, re.S)
    if gap_section:
        for line in gap_section.group(0).splitlines():
            line = line.strip()
            if re.match(r"^[0-9一二三四五六七八九十]+[.、]?\s*\*{0,2}", line):
                gap = re.sub(r"^\s*\*{1,2}\s*", "", line)
                gap = re.sub(r"\*{1,2}\s*$", "", gap).strip()
                crit.gaps.append(gap)
    ov = re.search(r"综合评语[^#]*?(?:\n##|\Z)", reply, re.S)
    if ov:
        first = next(
            (ln.strip() for ln in ov.group(0).splitlines()
             if ln.strip() and "综合" not in ln),
            "",
        )
        crit.overall = first
    return crit


def self_critique(
    chapter_text: str,
    chapter_no: int,
    title: str,
    responder,
    persona_sheet: str | None = None,
    n_samples: int = 3,
    timeout: float = 90.0,
) -> Critique | None:
    if responder is None or not getattr(responder, "available", False):
        return None
    samples = benchmark_samples(n=n_samples)
    net_count = sum(1 for s in samples if s.source == "wcshuba")
    builtin_count = sum(1 for s in samples if s.source == "builtin")
    try:
        prompt = _build_critique_prompt(chapter_text, chapter_no, title, samples, persona_sheet)
        reply = call_responder(responder, prompt, memories=None, timeout=timeout)
    except Exception as e:
        return Critique(chapter=chapter_no, title=title,
                        overall=f"自评请求失败：{e}",
                        benchmark_note=f"对标：{net_count} 篇真实章节 + {builtin_count} 家风格库")
    crit = _parse_critique(reply, chapter_no, title)
    crit.benchmark_note = (
        f"对标：{net_count} 篇真实章节(wcshuba) + "
        f"{builtin_count} 家内置风格库"
    )
    return crit


def persist_improvements(agent, critique: Critique, importance: float = 0.7) -> int:
    added = 0
    for imp in critique.improvements:
        if not imp:
            continue
        content = f"写作改进：{imp}"
        if any(
            _sim(content, m.content) > 0.82
            for m in agent.store.all() if m.kind == "skill"
        ):
            continue
        agent.remember_skill(content, importance=importance)
        added += 1
    return added


def _sim(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def writing_improvements(agent) -> str:
    sk = [m for m in agent.store.all() if m.kind == "skill"]
    sk.sort(key=lambda m: m.importance, reverse=True)
    seen: set[str] = set()
    lines: list[str] = []
    for m in sk:
        body = m.content
        key = body[:20]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {body}")
        if len(lines) >= 12:
            break
    return "\n".join(lines) if lines else "（暂无——等你写完几章、自评沉淀后会自动出现）"
