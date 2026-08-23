# -*- coding: utf-8 -*-
"""读者友好度 post-processor 测试：机制 + 出厂婴儿原则（包内零词表）。"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memagent.reader_postproc import (  # noqa: E402
    inject_explanations,
    load_terms_for_work,
)

TERMS = {"错季相": "——同时掌控淬生与蚀灭两种灵气相位的至高境界",
         "骨契": "——以骨为契、以血为印的古老契约"}


def test_injects_explanation_for_unexplained_term():
    text = "他掌心的塔纹亮起，那是骨契的力量。"
    out, injected = inject_explanations(text, TERMS)
    assert "骨契" in injected and "以血为印" in out


def test_skips_already_explained_term():
    text = "这是骨契——以血为印的古老契约，不容违背。"
    out, injected = inject_explanations(text, TERMS)
    assert "骨契" not in injected
    assert out.count("以血为印") == 1          # 不重复注入


def test_empty_terms_is_noop():
    text = "他掌心的骨契发烫。"
    out, injected = inject_explanations(text, {})
    assert out == text and injected == []


def test_loader_missing_config_returns_empty(tmp_path):
    """没有 term_explanations.json = 功能静默关闭（出厂婴儿原则）。"""
    assert load_terms_for_work(tmp_path) == {}
    # 父目录兜底查找
    cfg = tmp_path / "term_explanations.json"
    cfg.write_text(json.dumps(TERMS, ensure_ascii=False), encoding="utf-8")
    sub = tmp_path / "chapters"
    sub.mkdir()
    assert load_terms_for_work(sub) == TERMS


def test_loader_corrupt_json_returns_empty(tmp_path):
    cfg = tmp_path / "term_explanations.json"
    cfg.write_text("{broken", encoding="utf-8")
    assert load_terms_for_work(tmp_path) == {}


def test_longer_term_wins_over_prefix():
    """长术语优先：'错季相'不被短术语'错季'的解释抢先占用。"""
    terms = dict(TERMS, 错季="——时间错位现象")
    text = "他终于踏入了错季相的门槛。"
    out, injected = inject_explanations(text, terms)
    assert injected == ["错季相"]
    assert "至高境界" in out
