# memagent 会话上下文

生成时间：2026-08-11 08:44。
用法：开工时把本文件内容作为上下文提供给 agent，或
运行 `python session_memory.py --inject-agents-md` 直接维护 AGENTS.md 顶部。

## 本次注入的决策记忆（memagent 自动生成）

以下为跨会话沉淀的开发决策，按相关性/强度注入；被反复检索的决策越来越强。

1. [skill] 开发决策：按类型再巩固因子：技能 drift 0.15（回忆时高度稳定）、语义 1.0 基准、情景 2.5（易被情境改写）（强度 0.44 · 检索 1 次）
2. [semantic] 开发决策：对照实验靠可注入时钟 now_fn 确定性快进（秒级参数代表数天），不依赖真实 sleep、可精确复现（强度 0.44 · 检索 1 次）
3. [semantic] 开发决策：循环导入坑：profiles ↔ visualize 互相依赖，用函数级导入（运行时再 import）解决（强度 0.44 · 检索 1 次）
4. [semantic] 开发决策：access_count ≥ 2 守护：从未被使用过的新事实不会一觉醒来翻转成情景（强度 0.44 · 检索 1 次）
5. [semantic] 开发决策：零检索对照组的内容与所有查询零 n-gram 共享，避免哈希嵌入泛化命中污染对照（强度 0.44 · 检索 1 次）

> 重新运行 `python session_memory.py --inject-agents-md` 更新本区块。

