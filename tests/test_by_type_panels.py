"""按类型分面板曲线图测试：三张子图、参考曲线、空面板占位、导出集成。"""

from memagent import MemoryAgent
from memagent.visualize import render_svg_by_type


def _agent() -> MemoryAgent:
    agent = MemoryAgent()
    agent.remember("我在学习做饭")      # skill
    agent.remember("我昨天去吃了火锅")   # episodic
    agent.remember("北京是中国的首都")   # semantic
    return agent


def test_three_panels_with_reference_curves(tmp_path):
    out = render_svg_by_type(_agent(), str(tmp_path / "t.svg"), horizon_seconds=120.0)
    svg = open(out, encoding="utf-8").read()
    for t in ("skill", "semantic", "episodic"):
        assert f">{t}（" in svg          # 面板标题
    assert svg.count("参考：") == 3       # 每张子图都有参考曲线
    assert "强度下限 0.2" in svg
    assert svg.count("</svg>") == 1


def test_empty_type_panel_placeholder(tmp_path):
    agent = MemoryAgent()
    agent.remember("我昨天去吃了火锅")    # 只有 episodic
    out = render_svg_by_type(agent, str(tmp_path / "t.svg"), horizon_seconds=60.0)
    svg = open(out, encoding="utf-8").read()
    assert "无 skill 类记忆" in svg
    assert "无 semantic 类记忆" in svg


def test_panel_labels_include_tau_and_fit(tmp_path):
    agent = _agent()
    out = render_svg_by_type(agent, str(tmp_path / "t.svg"), horizon_seconds=120.0)
    svg = open(out, encoding="utf-8").read()
    assert "τ=" in svg                      # 每面板标配置 τ


def test_plot_curves_includes_by_type(tmp_path):
    agent = _agent()
    files = agent.plot_curves(str(tmp_path / "curves"))
    assert len(files) == 5
    assert any(f.endswith("_by_type.svg") for f in files)
    assert any(f.endswith(".csv") for f in files)          # 曲线长表
    assert any(f.endswith("_awakenings.csv") for f in files)  # 唤醒事件明细表
    assert any(f.endswith(".json") for f in files)
