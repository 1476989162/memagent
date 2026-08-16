"""联网搜索测试：Bing / DuckDuckGo HTML 解析、失败回退为空列表。"""

from memagent.websearch import _parse_bing, _parse_duckduckgo, search_web

_BING_SAMPLE = """
<html><body>
<li class="b_algo">
  <h2><a href="https://example.com/xuanhuan">玄幻小说_百度百科</a></h2>
  <div class="b_caption"><p>玄幻小说的准确定义是：以玄学为基础……</p></div>
</li>
<li class="b_algo">
  <h2><a href="https://qidian.com/list">玄幻小说大全</a></h2>
  <div class="b_caption"><p>好看的玄幻小说推荐</p></div>
</li>
</body></html>
"""

_DDG_SAMPLE = """
<div class="result results_links_deep">
  <div class="links_main">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fzhu">仙侠体系设定</a>
    <a class="result__snippet">关于修炼境界的详细梳理……</a>
  </div>
</div>
"""


def test_parse_bing():
    res = _parse_bing(_BING_SAMPLE, 5)
    assert len(res) == 2
    assert res[0]["title"] == "玄幻小说_百度百科"
    assert res[0]["url"] == "https://example.com/xuanhuan"
    assert "玄学为基础" in res[0]["snippet"]


def test_parse_bing_respects_n():
    res = _parse_bing(_BING_SAMPLE, 1)
    assert len(res) == 1


def test_parse_duckduckgo_decodes_redirect():
    res = _parse_duckduckgo(_DDG_SAMPLE, 5)
    assert len(res) == 1
    assert res[0]["url"] == "https://example.com/zhu"     # uddg= 解码还原真实 URL
    assert res[0]["title"] == "仙侠体系设定"
    assert "修炼境界" in res[0]["snippet"]


def test_search_web_network_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise OSError("网络不可用")
    monkeypatch.setattr("memagent.websearch._fetch", boom)
    assert search_web("任何查询") == []


def test_search_web_unparsable_returns_empty(monkeypatch):
    monkeypatch.setattr("memagent.websearch._fetch", lambda url: "<html>no results</html>")
    assert search_web("随便") == []
