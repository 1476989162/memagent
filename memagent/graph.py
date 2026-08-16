"""知识图谱：记忆间关系网络，支撑模式提取与概念形成。零依赖纯 Python。"""
from collections import defaultdict


class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, name, **props):
        if name in self.nodes:
            self.nodes[name]["properties"].update(props)
            self.nodes[name]["strength"] += 0.05
            return
        self.nodes[name] = {"name": name, "strength": 0.0, "properties": props, "labels": set()}

    def label_node(self, name, label):
        if name in self.nodes:
            self.nodes[name]["labels"].add(label)

    def add_edge(self, source, target, relation, strength=0.0):
        self.edges.append({"source": source, "target": target, "relation": relation, "strength": strength})
        self.add_node(source)
        self.add_node(target)
        if source in self.nodes:
            self.nodes[source]["strength"] += 0.02 * strength
        if target in self.nodes:
            self.nodes[target]["strength"] += 0.02 * strength

    def find_relations(self, source=None, target=None, relation_type=None):
        r = []
        for e in self.edges:
            if source and e["source"] != source:
                continue
            if target and e["target"] != target:
                continue
            if relation_type and e["relation"] != relation_type:
                continue
            r.append(e)
        return r

    def neighbors(self, name, relation_type=None, direction="out"):
        ns = []
        for e in self.edges:
            if direction == "out" and e["source"] == name:
                if relation_type is None or e["relation"] == relation_type:
                    ns.append((e["target"], e["relation"], e["strength"]))
            elif direction == "in" and e["target"] == name:
                if relation_type is None or e["relation"] == relation_type:
                    ns.append((e["source"], e["relation"], e["strength"]))
            elif direction == "both" and (e["source"] == name or e["target"] == name):
                if relation_type is None or e["relation"] == relation_type:
                    neighbor = e["target"] if e["source"] == name else e["source"]
                    ns.append((neighbor, e["relation"], e["strength"]))
        return ns

    def get_node(self, name):
        return self.nodes.get(name, {"name": name, "strength": 0.0, "properties": {}, "labels": set()})

    def node_names(self):
        return list(self.nodes.keys())

    def top_nodes(self, n=5):
        return sorted(self.nodes.values(), key=lambda x: -x["strength"])[:n]

    def to_dict(self):
        return {
            "nodes": {
                name: {
                    **node,
                    "labels": sorted(node.get("labels", set())),
                }
                for name, node in self.nodes.items()
            },
            "edges": [dict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        data = data or {}
        for name, node in (data.get("nodes") or {}).items():
            obj.nodes[name] = {
                "name": node.get("name", name),
                "strength": float(node.get("strength", 0.0)),
                "properties": dict(node.get("properties") or node.get("props") or {}),
                "labels": set(node.get("labels") or []),
            }
        obj.edges = [dict(edge) for edge in (data.get("edges") or [])]
        return obj
