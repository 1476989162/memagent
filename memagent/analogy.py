"""类比迁移（AnalogyTransfer）：领域A的模式应用到领域B。

类比原理：找到源领域（已知）和目标领域（待解决）的结构相似性，
将源领域的模式/策略迁移到目标领域。

流程：
  1) 接收一个目标领域的问题（domain + 问题描述）
  2) 在所有已知领域中寻找结构相似的模式
  3) 匹配成功 → 返回迁移建议（含置信度）

零依赖。
"""
from typing import Any, Callable
from .embedding import ngrams as _ngrams


def _tokenize(text: str) -> set[str]:
    """字符 n-gram 分词（支持中文，无需空格）。"""
    return set(_ngrams(text, ns=(2, 3)))


class AnalogyTransfer:
    def __init__(self, agent=None):
        self.agent = agent
        self.transfer_history: list = []
        self._analogy_registry: dict[str, list] = {}  # domain -> analogy_rules

    def register_analogy(self, source_domain: str, source_pattern: str,
                         target_domain: str, target_mapping: str, confidence: float = 0.5):
        """显式注册一条类比规则。"""
        key = f"{source_domain}→{target_domain}"
        if key not in self._analogy_registry:
            self._analogy_registry[key] = []
        self._analogy_registry[key].append({
            "source_pattern": source_pattern,
            "target_mapping": target_mapping,
            "confidence": confidence,
        })

    def auto_learn_analogy(self, domain_a: str, domain_b: str):
        """从两个领域的模式相似性中自动学习类比规则。"""
        if not self.agent:
            return []
        growth = self.agent.growth
        patterns_a = [p for p in growth.patterns if p.topic == domain_a]
        patterns_b = [p for p in growth.patterns if p.topic == domain_b]
        if not patterns_a or not patterns_b:
            return []

        learned = []
        for pa in patterns_a:
            best_match = None
            best_sim = 0.0
            for pb in patterns_b:
                # 结构相似性：antecedent/consequent 词汇重叠
                ant_overlap = self._jaccard(set(pa.antecedent.split()), set(pb.antecedent.split()))
                con_overlap = self._jaccard(set(pa.consequent.split()), set(pb.consequent.split()))
                sim = (ant_overlap + con_overlap) / 2
                if sim > best_sim:
                    best_sim = sim
                    best_match = pb
            if best_match and best_sim > 0.2:
                learned.append({
                    "source_domain": domain_a, "target_domain": domain_b,
                    "source_pattern": pa.consequent,
                    "target_pattern": best_match.consequent,
                    "similarity": round(best_sim, 4),
                    "confidence": round(best_sim * pa.confidence, 4),
                })
        self.transfer_history.extend(learned)
        for l in learned:
            key = f"{domain_a}→{domain_b}"
            if key not in self._analogy_registry:
                self._analogy_registry[key] = []
            self._analogy_registry[key].append(l)
        return learned

    def suggest(self, target_domain: str, query: str) -> list:
        """为目标领域问题寻找类比迁移建议。"""
        suggestions = []
        for key, rules in self._analogy_registry.items():
            src, tgt = key.split("→", 1)
            if tgt != target_domain:
                continue
            for rule in rules:
                # register_analogy 的规则只有 target_mapping，auto_learn 的才有 target_pattern
                match_text = rule.get("target_pattern") or rule.get("target_mapping", "")
                if self._jaccard(set(match_text.split()), set(query.split())) > 0.2:
                    suggestions.append({
                        "source_domain": src,
                        "analogy": f"'{rule.get('source_pattern','')}' → '{rule.get('target_mapping', rule.get('target_pattern',''))}'",
                        "confidence": rule.get("confidence", 0.5),
                    })
        return sorted(suggestions, key=lambda x: -x["confidence"])[:5]

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def transfer_summary(self) -> dict:
        return {
            "registered_analogies": {k: len(v) for k, v in self._analogy_registry.items()},
            "total_transfers": len(self.transfer_history),
            "domains": list(set(
                [t["source_domain"] + "→" + t["target_domain"] for t in self.transfer_history]
            )),
        }

    def to_dict(self) -> dict:
        return {
            "transfer_history": list(self.transfer_history),
            "analogy_registry": {
                key: [dict(rule) for rule in rules]
                for key, rules in self._analogy_registry.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict | None, *, agent=None) -> "AnalogyTransfer":
        obj = cls(agent=agent)
        data = data or {}
        obj.transfer_history = list(data.get("transfer_history") or [])
        obj._analogy_registry = {
            str(key): [dict(rule) for rule in rules]
            for key, rules in (data.get("analogy_registry") or {}).items()
        }
        return obj
