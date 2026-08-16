"""按类型分流内容钩子测试：技能类回忆用一致性校验而非情境改写。"""

import json
import os
import tempfile

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.checkers import consistency_checker
from memagent.memory import MemType


def _agent(**kw) -> MemoryAgent:
    cfg_kw = {k: v for k, v in kw.items() if k in AgentConfig.__dataclass_fields__}
    hook_kw = {k: v for k, v in kw.items() if k in ("content_updater", "content_updaters")}
    return MemoryAgent(
        cfg=AgentConfig(tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0}, **cfg_kw),
        **hook_kw,
    )


def test_type_dispatch_skill_uses_own_hook_episodic_falls_back():
    # 通用钩子：情境改写；技能专属：一致性校验
    agent = _agent(
        content_updater=lambda m, q, lab: m.content + f"（回忆情境:{q}）",
        content_updaters={MemType.SKILL: consistency_checker()},
    )
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    ep = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
    agent.retrieve("西红柿炒蛋的做法", k=1)
    agent.retrieve("昨天去吃了火锅", k=1)
    assert "回忆情境" not in sk.content      # 技能：校验，不改写
    assert "回忆情境" in ep.content         # 情景：回退通用钩子，情境改写
    assert sk.checks                        # 技能记了校验事件
    assert ep.checks == []                  # 情景走改写，不记校验


def test_consistent_recall_leaves_skill_unchanged():
    agent = _agent(content_updaters={MemType.SKILL: consistency_checker()})
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    before = sk.content
    agent.retrieve("西红柿炒蛋的做法", k=1)
    assert sk.content == before
    assert sk.revision_count == 0                     # 校验不算修订（内容未变）
    assert sk.checks and sk.checks[-1][2] in ("consistent", "unknown")


def test_conflict_recall_logged_but_not_rewritten():
    agent = _agent(content_updaters={MemType.SKILL: consistency_checker()})
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    before = sk.content
    agent.retrieve("我不会做西红柿炒蛋", k=1)   # 含否定信号 → 冲突
    assert sk.content == before                  # 留痕不改写
    assert sk.revision_count == 0
    assert sk.checks[-1][2] == "conflict"


def test_conflict_with_rewriter_corrects_content():
    def rewriter(mem, query):
        return mem.content.replace("会", "不会")  # 修正：技能内容被情境纠正

    agent = _agent(
        content_updaters={
            MemType.SKILL: consistency_checker(rewrite_on_conflict=rewriter)
        }
    )
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    agent.retrieve("我不会做西红柿炒蛋", k=1)
    assert sk.content == "我不会做西红柿炒蛋"
    assert sk.revision_count == 1                # 真正改写才算修订
    assert sk.checks[-1][2] == "corrected"


def test_dispatch_by_string_key():
    agent = _agent(content_updaters={"skill": consistency_checker()})
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    ep = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
    agent.retrieve("西红柿炒蛋的做法", k=1)
    agent.retrieve("昨天去吃了火锅", k=1)
    assert sk.checks          # 字符串键 "skill" 生效
    assert ep.checks == []    # episodic 无专属钩子且无通用钩子 → 不跑


def test_no_hook_means_no_content_editing():
    agent = _agent()
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    before = sk.content
    agent.retrieve("西红柿炒蛋的做法", k=1)
    assert sk.content == before   # 无钩子：不做内容编辑
    assert sk.checks == []        # 也无校验事件
    assert sk.revision_count == 1  # 但向量漂移（drift 因子 0.15）照常计为修订


def test_checks_exported_in_json():
    agent = _agent(content_updaters={MemType.SKILL: consistency_checker()})
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    agent.retrieve("我不会做西红柿炒蛋", k=1)
    assert sk.checks
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "m")
        agent.plot_curves(base)
        with open(base + ".json", encoding="utf-8") as f:
            data = json.load(f)
        entry = next(x for x in data["memories"] if x["id"] == sk.id)
        assert entry["checks"] == sk.checks
        assert entry["checks"][0][2] == "conflict"


def test_checks_persist_across_roundtrip():
    agent = _agent(content_updaters={MemType.SKILL: consistency_checker()})
    sk = agent.remember("我会做西红柿炒蛋", importance=0.1, mtype=MemType.SKILL)
    agent.retrieve("我不会做西红柿炒蛋", k=1)
    d = sk.to_dict()
    restored = type(sk).from_dict(d)
    assert restored.checks == sk.checks
