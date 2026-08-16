"""好奇驱动探索（CuriosityDrivenExplore）：提问→主动搜索→写入的完整闭环。

流程：
  1) 从 GrowthEngine 自主提问队列取问题
  2) 主动搜索自身记忆和知识图谱尝试回答
  3) 如果找到答案 → 直接写入新记忆（闭环）
  4) 如果找不到 → 标记为"待问用户"
  5) 探索结果更新兴趣向量（找到答案的兴趣↑↑，未找到的兴趣↑）

零依赖。
"""
from typing import Callable


class CuriosityDrivenExplore:
    def __init__(self, agent=None):
        self.agent = agent
        self.discovered: list = []          # 自己探索得到的答案
        self.unanswered: list = []          # 待问用户的问题
        self.explore_history: list = []
        self._on_discover: Callable | None = None

    def explore_loop(self, max_steps: int = 3) -> dict:
        """执行一轮好奇驱动探索。返回本轮探索报告。"""
        if not self.agent:
            return {"errors": ["未绑定 agent"]}
        growth = self.agent.growth
        report = {"explore_count": 0, "discovered": 0, "unanswered": 0,
                  "discovered_list": [], "unanswered_list": []}

        questions = growth.autonomous_query(max_steps)
        if not questions:
            report["explore_count"] = 0
            return report

        report["explore_count"] = len(questions)
        now = self.agent._now()

        for q in questions:
            # 步骤1：主动搜索记忆和知识图谱
            hits = self.agent.retrieve(q, k=3)
            graph = self.agent.graph
            topics = self.agent.interest.detect_topics(q)

            # 步骤2：判断是否找到答案
            found = False
            answer_text = None

            # 2a) 记忆命中
            hit_memories = [h.memory for h in hits if h.total > 0.1]
            if hit_memories:
                answer_text = "；".join(m.content[:40] for m in hit_memories[:2])
                found = True

            # 2b) 知识图谱有结构信息
            if not found and topics:
                topic_nodes = [n for n in graph.nodes.values() if any(t in n["labels"] for t in topics)]
                if topic_nodes:
                    props = []
                    for n in topic_nodes:
                        for k, v in n.get("properties", {}).items():
                            props.append(f"{k}={v}")
                    if props:
                        answer_text = "、".join(props[:5])
                        found = True

            # 2c) 因果模式能给出预测
            if not found and topics:
                patterns = [p for p in growth.patterns if p.topic == topics[0]]
                if patterns:
                    p = max(patterns, key=lambda x: x.confidence)
                    answer_text = f"如果{p.antecedent}则{p.consequent}(置信{p.confidence:.2f})"
                    found = True

            if found:
                # 写入新记忆（闭环）
                content = f"[自我发现] {q} → {answer_text}"
                self.agent.remember(content, kind="fact", importance=0.7)
                self.discovered.append(content)
                report["discovered"] += 1
                report["discovered_list"].append(content[:60])
                # 兴趣值↑↑
                for t in topics:
                    self.agent.interest.set(t, self.agent.interest.get(t) + 0.03)
                if self._on_discover:
                    self._on_discover(content)
            else:
                # 待问用户
                self.unanswered.append(q)
                report["unanswered"] += 1
                report["unanswered_list"].append(q)
                # 兴趣值↑（好奇但未满足）
                for t in topics:
                    self.agent.interest.set(t, self.agent.interest.get(t) + 0.01)

            self.explore_history.append({
                "query": q, "found": found, "answer": answer_text, "t": now,
            })

        return report

    def ask_user_questions(self, max_questions: int = 3) -> list[str]:
        """返回待问用户的问题列表（未自行探索得到的）。"""
        return list(self.unanswered[-max_questions:])

    def answer_user_question(self, question: str, answer: str) -> dict:
        """用户回答了问题 → 写入记忆 → 更新兴趣。"""
        if not self.agent:
            return {"error": "未绑定 agent"}
        topics = self.agent.interest.detect_topics(question + " " + answer)
        self.agent.reremember(f"[用户解答] {question} → {answer}", kind="fact", importance=0.9)
        for t in topics:
            self.agent.interest.set(t, self.agent.interest.get(t) + 0.05)
        self.unanswered = [q for q in self.unanswered if q != question]
        self.explore_history.append({"type": "user_answer", "question": question,
                                     "answer": answer, "t": self.agent._now()})
        return {"question": question, "topics": topics, "written": True}

    def to_dict(self) -> dict:
        return {
            "discovered": list(self.discovered),
            "unanswered": list(self.unanswered),
            "explore_history": list(self.explore_history),
        }

    @classmethod
    def from_dict(cls, data: dict | None, *, agent=None) -> "CuriosityDrivenExplore":
        obj = cls(agent=agent)
        data = data or {}
        obj.discovered = list(data.get("discovered") or [])
        obj.unanswered = list(data.get("unanswered") or [])
        obj.explore_history = list(data.get("explore_history") or [])
        return obj
