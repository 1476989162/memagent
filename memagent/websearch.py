"""自主联网搜索：纯 stdlib urllib，零第三方依赖。

search_web(query, n=5) -> list[dict]  # [{title, url, snippet}, ...]

实现：
- 首选 Bing（国内可达），备用 DuckDuckGo HTML 端点；
- 带浏览器 User-Agent（与 llm._default_post 一致），绕过 Cloudflare 机器人拦截；
- HTML 用正则防御式解析（原型级），任何一步失败都返回空列表而不是抛错，
  保证自主演化/检索链路不被搜索失败拖垮。

设计映射：人脑的“好奇驱动探索”——遇到知识缺口时主动上网找资料，再把
收获沉淀成记忆（配合 agent.evolve() 的 with_web=True）。
"""

from __future__ import annotations

import html as _html
import re
import urllib.parse
import urllib.request

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_TIMEOUT = 15.0


def _fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _BROWSER_UA, "Accept": "text/html,application/xhtml+xml"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _clean_title(raw: str) -> str:
    # 去掉 Bing 结果标题里的高亮标记与尾缀
    return _html.unescape(_strip_tags(raw))


def _parse_bing(html: str, n: int) -> list[dict]:
    out: list[dict] = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        url, title = m.group(1), _clean_title(m.group(2))
        if not url.startswith("http"):
            continue
        snippet = ""
        sm = re.search(
            r'<div class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>', block, re.S
        )
        if sm:
            snippet = _html.unescape(_strip_tags(sm.group(1)))
        out.append({"title": title, "url": url, "snippet": snippet[:300]})
        if len(out) >= n:
            break
    return out


def _parse_duckduckgo(html: str, n: int) -> list[dict]:
    out: list[dict] = []
    for block in re.findall(r'<div[^>]*class="[^"]*result[^"]*".*?</div>\s*</div>', html, re.S):
        a = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        url, title = a.group(1), _clean_title(a.group(2))
        # DuckDuckGo 用 /l/?uddg= 重定向包装真实 URL
        um = re.search(r"uddg=([^&]+)", url)
        if um:
            url = urllib.parse.unquote(um.group(1))
        if not url.startswith("http"):
            continue
        snippet = ""
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        if sm:
            snippet = _html.unescape(_strip_tags(sm.group(1)))
        out.append({"title": title, "url": url, "snippet": snippet[:300]})
        if len(out) >= n:
            break
    return out


def search_web(query: str, n: int = 5) -> list[dict]:
    """联网搜索，返回 [{title, url, snippet}, ...]；失败/无结果返回空列表。

    依次尝试 Bing（国内可达）与 DuckDuckGo（备用），都失败则返回 []。
    """
    q = urllib.parse.quote(query)
    attempts = [
        _parse_bing,
        _parse_duckduckgo,
    ]
    urls = [
        f"https://www.bing.com/search?q={q}&setlang=zh-hans&count={n}",
        f"https://html.duckduckgo.com/html/?q={q}",
    ]
    for parser, url in zip(attempts, urls):
        try:
            results = parser(_fetch(url), n)
            if results:
                return results
        except Exception:
            continue  # 换下一个引擎
    return []
