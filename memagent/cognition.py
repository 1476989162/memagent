"""认知高层模块：技能发展 / 长期目标 / 自我模型。

Skill 发展：练习 → 熟练度曲线（指数增长）→ mastery 阶梯。
Goal 目标：显式设定长期目标 → 兴趣向量被目标牵引 → 进展追踪。
SelfModel 自我模型：元认知——知道"我知道什么/不知道什么"，给出
"认知边界报告"与各领域自信度。

零依赖纯 Python。
"""
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class Skill:
    """技能发展记录。"""
    name: str
    domain: str           # 所属兴趣领域
    practice_count: int = 0
    success_count: int = 0
    last_practice: float = 0.0
    last_success: float = 0.0
    practice_history: list = field(default_factory=list)
    _curves: list = field(default_factory=list)  # [(t, level)]

    @property
    def mastery(self) -> float:
        """熟练度 0~1：指数增长 + 失败惩罚。"""
        if self.practice_count == 0:
            return 0.0
        # 指数增长：每 5 次练习接近 mastery=1，但永远不到 1
        k = 0.4
        level = 1 - math.exp(-k * self.practice_count)
        # 失败惩罚：成功率低时打折扣
        success_rate = (self.success_count / self.practice_count) if self.practice_count else 0
        level *= (0.4 + 0.6 * success_rate)  # 最低 0.4 倍
        return min(1.0, level)

    def record_practice(self, success: bool, t: float):
        self.practice_count += 1
        self.practice_history.append({"success": success, "t": t})
        self.last_practice = t
        if success:
            self.success_count += 1
            self.last_success = t
        self._curves.append((t, self.mastery))

    def practice_history_str(self, top: int = 5) -> str:
        rows = [f"{'✓' if h['success'] else '✗'}" for h in self.practice_history[-top:]]
        return "".join(rows)


@dataclass
class Goal:
    """长期目标。"""
    name: str
    description: str
    domain: str
    target_skill: str | None = None  # 关联技能
    target_mastery: float = 0.9
    created_at: float = 0.0
    milestones: list = field(default_factory=list)
    _progress_history: list = field(default_factory=list)

    def progress(self, current_mastery: float) -> dict:
        ratio = current_mastery / self.target_mastery
        entry = {"mastery": round(current_mastery, 4), "ratio": round(ratio, 4)}
        self._progress_history.append(entry)
        return entry

    @property
    def achieved(self) -> bool:
        last = self._progress_history[-1] if self._progress_history else {"ratio": 0.0}
        return last["ratio"] >= 1.0


class Cognition:
    """自我模型（元认知）：整合兴趣/技能/目标/知识密度，提供
    '我知道什么/不知道什么'的元知识。"""

    def __init__(self, agent=None):
        self.skills: dict[str, Skill] = {}
        self.goals: dict[str, Goal] = {}
        self._skill_practice_handler: Callable[[str, bool, float], None] | None = None

    def register_skill(self, name: str, domain: str) -> Skill:
        if name not in self.skills:
            self.skills[name] = Skill(name=name, domain=domain)
        return self.skills[name]

    def practice(self, skill_name: str, success: bool, t: float):
        if skill_name in self.skills:
            self.skills[skill_name].record_practice(success, t)
        if self._skill_practice_handler:
            self._skill_practice_handler(skill_name, success, t)

    def set_goal(self, name: str, description: str, domain: str,
                 target_skill: str | None = None, target_mastery: float = 0.9,
                 created_at: float = 0.0):
        g = Goal(name=name, description=description, domain=domain,
                 target_skill=target_skill, target_mastery=target_mastery,
                 created_at=created_at)
        self.goals[name] = g

    def update_goal_progress(self, name: str):
        if name not in self.goals:
            return None
        g = self.goals[name]
        if g.target_skill and g.target_skill in self.skills:
            return g.progress(self.skills[g.target_skill].mastery)
        return g.progress(0.0)

    def goal_summary(self) -> list[dict]:
        out = []
        for g in self.goals.values():
            mastery = self.skills[g.target_skill].mastery if g.target_skill and g.target_skill in self.skills else 0.0
            out.append({
                "name": g.name, "domain": g.domain,
                "target_skill": g.target_skill,
                "mastery": round(mastery, 4),
                "ratio": round(mastery / g.target_mastery, 4) if g.target_mastery else 0.0,
                "achieved": g.achieved,
            })
        return sorted(out, key=lambda x: -x["ratio"])

    def skill_summary(self) -> list[dict]:
        return sorted(
            [{"name": s.name, "domain": s.domain,
              "mastery": round(s.mastery, 4),
              "practices": s.practice_count,
              "success_rate": round(s.success_count / s.practice_count, 3) if s.practice_count else 0.0}
             for s in self.skills.values()],
            key=lambda x: -x["mastery"],
        )

    def knowledge_boundary(self, interest_getter: Callable[[str], float] | None = None,
                            topics: list[str] | None = None) -> list[dict]:
        """认知边界报告：整合技能/目标/兴趣/知识图谱密度，
        给出各领域的'我知道什么/不知道什么'。"""
        topic_list = topics or [g.domain for g in self.goals.values()]
        seen = set()
        for d in [s.domain for s in self.skills.values()]:
            if d not in seen:
                topic_list.append(d)
                seen.add(d)

        report = []
        for topic in topic_list:
            known = []
            unknown = []
            skill_level = max((s.mastery for s in self.skills.values() if s.domain == topic), default=0.0)
            interest = interest_getter(topic) if interest_getter else 0.0

            # 根据技能/目标/兴趣推断认知状态
            if skill_level > 0.5:
                known.append(f"技能'{topic}'熟练度{skill_level:.2f}")
            elif skill_level > 0.2:
                known.append(f"技能'{topic}'入门{skill_level:.2f}")
                unknown.append(f"技能'{topic}'熟练度{skill_level:.2f}(目标0.9)")
            elif interest > 0.3:
                known.append(f"对'{topic}'有兴趣(兴趣值{interest:.2f})")
                unknown.append(f"尚未在'{topic}'建立技能")

            report.append({
                "topic": topic, "interest": round(interest, 4),
                "skill_level": round(skill_level, 4),
                "known": known, "unknown": unknown,
            })
        return report

    def self_summary(self) -> dict:
        return {
            "skills": self.skill_summary(),
            "goals": self.goal_summary(),
            "skill_count": len(self.skills),
            "goal_count": len(self.goals),
            "achieved_count": sum(1 for g in self.goals.values() if g.achieved),
        }

    def to_dict(self) -> dict:
        return {
            "skills": {name: asdict(skill) for name, skill in self.skills.items()},
            "goals": {name: asdict(goal) for name, goal in self.goals.items()},
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Cognition":
        obj = cls()
        data = data or {}
        obj.skills = {
            str(name): Skill(**row)
            for name, row in (data.get("skills") or {}).items()
        }
        obj.goals = {
            str(name): Goal(**row)
            for name, row in (data.get("goals") or {}).items()
        }
        return obj
