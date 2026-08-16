"""多智能体社交学习（SocialLearner）：从其他 agent 获取知识/技能/模式。

支持三种社交学习模式：
  1) 知识交换：从一个 agent 拉取知识图谱节点+边
  2) 技能示范：从另一个 agent 观察其技能表现
  3) 经验共享：从另一个 agent 拉取高价值记忆

零依赖。社交学习通过显式调用，不需要网络。
"""
from typing import Any
from .memory import Tier


class SocialLearner:
    def __init__(self, agent=None):
        self.agent = agent
        self.peers: dict[str, Any] = {}  # peer_id -> agent
        self.social_history: list = []

    def register_peer(self, peer_id: str, peer_agent) -> None:
        self.peers[peer_id] = peer_agent

    def register_peers(self, peers: dict[str, Any]) -> None:
        self.peers.update(peers)

    def share_knowledge(self, peer_id: str) -> dict:
        """从 peer 获取知识图谱节点+边。"""
        peer = self.peers.get(peer_id)
        if not peer:
            return {"error": f"未知 peer: {peer_id}"}
        report = {"peer": peer_id, "nodes_learned": 0, "edges_learned": 0}
        src_graph = peer.graph
        dst_graph = self.agent.graph
        before_nodes = len(dst_graph.nodes)
        before_edges = len(dst_graph.edges)
        for nid, node in src_graph.nodes.items():
            if nid not in dst_graph.nodes:
                dst_graph.add_node(nid, **dict(node.get("properties") or {}))
                dst_graph.nodes[nid]["strength"] = float(node.get("strength", 0.0))
                for label in node.get("labels", []):
                    dst_graph.label_node(nid, label)
        for edge in src_graph.edges:
            # 检查目标图谱中是否已有同名同向同关系边
            exists = any(e["source"] == edge["source"] and e["target"] == edge["target"]
                        and e["relation"] == edge["relation"] for e in dst_graph.edges)
            if not exists:
                dst_graph.add_edge(edge["source"], edge["target"], edge["relation"],
                                  edge["strength"])
        report["nodes_learned"] = len(dst_graph.nodes) - before_nodes
        report["edges_learned"] = len(dst_graph.edges) - before_edges
        self.social_history.append({"type": "share_knowledge", **report})
        return report

    def share_memory(self, peer_id: str, topics: list[str] | None = None,
                     max_count: int = 10) -> dict:
        """从 peer 获取高价值记忆（按指定主题或全局 Top-N）。"""
        peer = self.peers.get(peer_id)
        if not peer:
            return {"error": f"未知 peer: {peer_id}"}
        now = self.agent._now()
        report = {"peer": peer_id, "memories_learned": 0, "topics": topics}
        # 从 peer 的 Hot/Warm 层选 Top-N 高重要性记忆
        candidates = [m for m in peer.store.all() if m.tier != Tier.COLD and m.kind != "turn"]
        candidates.sort(key=lambda m: -m.importance)
        for m in candidates[: max_count]:
            # 写入自身记忆，标记为社交来源
            content = f"[来自{peer_id}] {m.content}"
            self.agent.remember(content, kind="fact", importance=m.importance)
            report["memories_learned"] += 1
        self.social_history.append({"type": "share_memory", **report})
        return report

    def observe_peer(self, peer_id: str) -> dict:
        """观察 peer 的技能表现（不获取记忆/图谱，只学技能数据）。"""
        peer = self.peers.get(peer_id)
        if not peer:
            return {"error": f"未知 peer: {peer_id}"}
        report = {"peer": peer_id, "skills_observed": 0}
        if hasattr(peer, "cognition"):
            for skill_name, skill in peer.cognition.skills.items():
                if skill.mastery > 0.3:
                    # 在自身创建/更新同名技能，设置为基础水平
                    if skill_name not in self.agent.cognition.skills:
                        domain = skill.domain
                        self.agent.cognition.register_skill(skill_name, domain)
                    report["skills_observed"] += 1
        self.social_history.append({"type": "observe_peer", **report})
        return report

    def social_summary(self) -> dict:
        return {
            "peers": list(self.peers.keys()),
            "total_interactions": len(self.social_history),
            "knowledge_shared": sum(1 for h in self.social_history if h.get("type") == "share_knowledge"),
            "memory_shared": sum(1 for h in self.social_history if h.get("type") == "share_memory"),
        }

    def to_dict(self) -> dict:
        return {"social_history": list(self.social_history)}

    @classmethod
    def from_dict(cls, data: dict | None, *, agent=None) -> "SocialLearner":
        obj = cls(agent=agent)
        obj.social_history = list((data or {}).get("social_history") or [])
        return obj
