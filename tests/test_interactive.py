"""交互式曲线测试：单文件 HTML 结构、分组、数据嵌入、交互 JS。"""

import json
import re

from memagent import MemoryAgent
from memagent.interactive import render_interactive_html


def _html(tmp_path) -> str:
    agent = MemoryAgent()
    agent.remember("我叫小林，喜欢爬山")
    agent.remember("我昨天去吃了火锅")
    agent.remember("我在学习做饭")
    agent.retrieve("我昨天去吃了火锅", k=1)  # 制造一次检索事件（环标）
    path = render_interactive_html(agent, str(tmp_path / "inter.html"), horizon_seconds=120.0)
    return open(path, encoding="utf-8").read()


def test_html_structure(tmp_path):
    html = _html(tmp_path)
    assert "<svg" in html and "id=\"plot\"" in html
    assert "mem-curve" in html
    assert "tier-group" in html
    assert "滚轮缩放" in html  # 交互提示
    assert "application/json" in html
    # 多视图仪表盘面板
    assert 'id="bubble"' in html       # 记忆地图气泡图
    assert 'id="toplist"' in html      # 最强记忆列表
    assert 'id="dist"' in html         # 层级×类型分布
    assert 'id="stats"' in html        # 统计条
    assert 'class="access-tick"' in html  # 检索事件环标
    assert 'data-base-w="' in html     # 线宽随重要性


def test_curves_grouped_by_tier(tmp_path):
    html = _html(tmp_path)
    for tier in ("hot", "warm", "cold"):
        # 有记忆的层级应有分组标签；Warm 至少包含 mem-curve
        assert f'data-tier="{tier}"' in html


def test_embedded_data_parses(tmp_path):
    html = _html(tmp_path)
    m = re.search(r'<script id="memdata" type="application/json">(.*?)</script>', html, re.S)
    assert m, "缺少数据脚本"
    data = json.loads(m.group(1))
    assert "now" in data and len(data["memories"]) == 3
    fields = {"id", "tier", "mtype", "importance", "access_count", "revision_count", "content",
              "strength", "access_events"}
    assert fields <= set(data["memories"][0].keys())
    assert isinstance(data["memories"][0]["access_events"], list)


def test_interactive_js_present(tmp_path):
    html = _html(tmp_path)
    # 缩放、平移、点击高亮、层级切换、重置、联动渲染
    for snippet in ("addEventListener('wheel'", "pointerdown", "function select(", "data-tier",
                    "function deselect", "function renderBubble", "function renderDist",
                    "function renderToplist", "function renderStats"):
        assert snippet in html


def test_no_external_resources(tmp_path):
    html = _html(tmp_path)
    assert "src=" not in html and "href=" not in html  # 无外部脚本/样式/图片，单文件离线可用


def test_type_compare_view(tmp_path):
    """类型对比视图：JS 渲染、时间窗控件、参考曲线数据、可点击元素。"""
    html = _html(tmp_path)
    assert 'id="typechart"' in html            # 类型对比 SVG 容器
    assert "类型对比" in html
    # JS 渲染逻辑（三列子图 + 可点击曲线/观测点 + 参考曲线）
    assert "function renderTypeChart" in html
    assert 'class="sub-curve"' in html         # 预测曲线渲染模板
    assert 'class="sub-dot"' in html           # 观测点渲染模板
    assert 'class="sub-ref"' in html           # 典型遗忘参考曲线
    # 时间窗控件
    for wid in ("twPast", "twFuture", "twApply", "twReset"):
        assert f'id="{wid}"' in html
    assert "tw-preset" in html                 # 预设按钮
    # JS 联动：子图元素接入全局高亮、层级切换
    assert "'.sub-curve'" in html
    assert "getElementById('typechart')" in html


def test_type_compare_data_complete(tmp_path):
    """类型对比数据：type_window 元数据 + 记忆状态（JS 端按窗口自适应生成曲线点）。"""
    html = _html(tmp_path)
    m = re.search(r'<script id="memdata" type="application/json">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    assert {"t0", "t1", "now"} <= set(data["type_window"])
    for mem in data["memories"]:
        # JS 端按窗口用强度公式生成预测点所需的状态字段
        for field in ("id", "mtype", "tier", "importance", "access_count",
                      "last_access", "strength", "recorded"):
            assert field in mem, field
    # JS 端自适应点生成：公式常量 + 窗口重绘入口
    assert "DECAY.kappa" in html and "DECAY.wRec" in html
    assert "function renderTypeChart(w0, w1)" in html


def test_main_plot_overlays_type_reference_curves(tmp_path):
    """交互版主图同样叠加 3 条类型参考曲线（main-ref，类型色虚线）。"""
    html = _html(tmp_path)
    assert html.count('class="main-ref"') == 3
    for color in ("#2f9e44", "#7048e8", "#f08c00"):
        assert f'stroke="{color}"' in html
    # 参考曲线不参与点击/层级切换（无 data-mem/data-tier）
    m = re.search(r'class="main-ref"[^>]*>', html)
    assert m and "data-mem" not in m.group(0)


def _html_with_awakening(tmp_path) -> str:
    """制造真实唤醒事件的仪表盘：Cold → recall → 唤醒偏差观测。"""
    from memagent.agent import AgentConfig
    from memagent.memory import MemType

    clock = [1000.0]
    cfg = AgentConfig(tau_by_type={MemType.EPISODIC: 3 * 86400.0},
                      true_tau_by_type={MemType.EPISODIC: 2 * 86400.0},
                      cold_after_seconds=50.0)
    agent = MemoryAgent(cfg=cfg, now_fn=lambda: clock[0])
    m = agent.store.add("我昨天去吃了火锅", importance=0.5,
                        mtype=MemType.EPISODIC, now=clock[0])
    clock[0] += 200
    m.demote_to_cold("火锅聚餐（已归档）")
    clock[0] += 1000
    agent.recall(m.id[:6])
    path = render_interactive_html(agent, str(tmp_path / "aw.html"), horizon_seconds=120.0)
    return open(path, encoding="utf-8").read()


def test_awakening_markers_and_dual_values(tmp_path):
    """唤醒点在曲线图标注 dev vs expected 双值：菱形标记 + 红条（实测）/青条（类型
    预期）+ 双值标签；悬停提示含比值；数据注入 awakening_events。"""
    html = _html_with_awakening(tmp_path)
    assert 'class="awake-mark"' in html          # 菱形唤醒点
    assert 'class="awake-dev"' in html           # 红条 = 实测 dev
    assert 'class="awake-exp"' in html           # 青条 = 类型预期 expected
    assert 'stroke="#e34a2f"' in html            # 红
    assert 'stroke="#2a9d8f"' in html            # 青
    assert re.search(r'class="awake-label"[^>]*>dev 0\.0\d+/预期 0\.0\d+', html)
    assert "vs 预期" in html                       # 悬停提示双值
    assert "唤醒事件" in html                      # 统计条
    # 数据注入
    m = re.search(r'<script id="memdata" type="application/json">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    aw = [e for mem in data["memories"] for e in mem.get("awakening_events", [])]
    assert len(aw) == 1
    assert {"ts", "dev", "expected", "ratio", "mtype"} <= set(aw[0])
    assert aw[0]["mtype"] == "episodic"
    # 交互接线：唤醒标注参与点击高亮（data-mem + 选择器）
    assert 'class="awake-mark" data-mem=' in html
    assert "awake-mark" in re.search(r"querySelectorAll\('\.mem-trajectory[^;]+;" , html).group(0)
    # 交互展开接线：每个唤醒元素带事件序号（点击展开 dev vs expected 双条 + 信号方向）
    assert 'data-evi="0"' in html
    assert 'id="awakeCallout"' in html          # 悬浮 callout 容器
    assert "function showAwakening" in html     # 点击唤醒点 → 展开
    assert "function drawAwakeningExpansion" in html
    assert "function clearAwakening" in html


def test_awakening_expansion_and_signal_direction(tmp_path):
    """唤醒点交互展开：点击 ◇/双条 → 展开 dev vs expected 双条 + 信号方向；
    与类型面板联动（选中记忆曲线高亮 + 面板内渲染可点击的唤醒菱形）。"""
    html = _html_with_awakening(tmp_path)
    # 信号方向三态：红↓=τ应下调（比值>1.05）/ 青↑=应上调（<0.95）/ 灰✓=已校准
    assert "ratio > 1.05" in html and "τ 应下调" in html
    assert "ratio < 0.95" in html and "τ 应上调" in html
    assert "已校准" in html
    # callout 双条比例条 + 解释
    assert 'class="act-bar"' in html and 'class="act-dir"' in html
    assert "埋得比信念深" in html or "忘得比信念慢" in html
    # 展开双条/箭头元素 class
    assert "awake-exp-bar" in html and "awake-exp-arrow" in html
    # 类型面板联动：选中记忆的唤醒点菱形（type-awake，点击 → showAwakening）+ 全局高亮选择器
    assert 'class="type-awake"' in html
    assert "type-awake" in html and "showAwakening(aw.dataset.mem" in html
    # 展开由 select 重建（类型面板重绘后不丢失），防重入守卫避免递归
    assert "重绘后重建展开元素" in html
    assert "let selecting" in html
    # 点击其他唤醒点/Esc/空白收起
    assert "点击其他唤醒点收起" in html


def test_bubble_slope_encoding_and_countdown(tmp_path):
    """记忆地图：气泡大小切换斜率（触底倒计时）/强度，点击显示倒计时。"""
    html = _html(tmp_path)
    # 大小模式切换按钮（默认斜率=触底倒计时）
    assert 'data-bsize="slope"' in html and 'data-bsize="strength"' in html
    assert "bubbleSizeMode" in html
    # 斜率编码：不触底→最大、已触底→最小、线性归一化
    assert "ttf == null) return 1" in html
    assert "ttf <= 0) return 0" in html
    # 点击气泡显示触底倒计时：详情面板倒计时行 + 每秒滴答
    assert 'id="floorcd"' in html
    assert "function updateFloorCountdown" in html
    assert "floorTimer = setInterval" in html
    # 气泡悬停提示带倒计时
    assert "触底倒计时 " in html
