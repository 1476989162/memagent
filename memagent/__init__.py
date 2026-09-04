"""memagent —— 模仿人脑分层遗忘机制的记忆系统原型。

设计映射（详见 README.md）：
- Hot 层   ≈ 工作记忆：近期高频使用的记忆，直接注入上下文
- Warm 层  ≈ 长时记忆：完整记忆，带遗忘曲线评分
- Cold 层  ≈ 海马体索引指向的深藏记忆：压缩摘要，命中才唤醒
- 衰减评分  ≈ Ebbinghaus 遗忘曲线
- 检索强化  ≈ 测试效应（retrieval practice）
- 睡眠巩固  ≈ 海马体重放：离线把低频记忆压缩进 Cold 层
"""

from .agent import MemoryAgent
from .memory import Memory, Tier, MemoryStore
from .embedding import embed_text, cosine_similarity

__all__ = ["MemoryAgent", "Memory", "Tier", "MemoryStore", "embed_text", "cosine_similarity"]
__version__ = "0.3.4"
