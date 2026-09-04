# Changelog

## Unreleased

### 0.3.3 — sensenova 推理模型 max_tokens 地板修复 + MCP/REST/人脑认知模块入主干

- **sensenova 推理模型 max_tokens 修复**：`responder.py` 只对真正开启 reasoning 的
  sensenova 模型（`model_thinking == "enabled"`）强制 16384 地板；非推理模型
  （如 `sensenova-6.8-flash-lite`）保持 1024 默认。用户显式传 `max_tokens`
  时不再被 16384 地板覆盖，空输出走既有 retry 机制兜底。
- **MCP 服务器入主干**：`memagent/mcp_server.py` —— 用官方 mcp SDK 的 lowlevel
  Server 把记忆系统暴露为 MCP 工具，支持 Hermes 等 Agent 原生调用。
- **REST API 入主干**：`memagent/server.py` —— 纯 stdlib `http.server`，零新依赖，
  给不用 Python 的调用方（Node/Go/浏览器）暴露 remember/retrieve/sleep 三个端点。
- **人脑认知模块入主干**：`memagent/human.py`（人类优势增强：REM 联想重组、
  扩散激活、检索诱导遗忘、舌尖现象）与 `memagent/continuity.py`（小说连续性
  审校：事实台账 + 写前节拍表 + 知识状态因果链核对）。
- **可插拔嵌入后端**：`memagent/embedders.py` —— `RemoteEmbedder`（OpenAI 兼容
  /embeddings，纯 urllib）与 `LocalEmbedder`（sentence-transformers，extra
  `embed-local`）；`embedding.embed_text()` 成为单点，`set_embedder()` 可换后端。
- **MCP 落盘节流**：`retrieve` 的测试效应/再巩固改为 5 秒节流保存（显式写
  操作仍强制落盘），避免每次检索全量 JSON 写；`retrieve` 命中附带
  `recall_hint`（Cold 摘要命中时提示 `memagent_recall` 唤醒细节）。
- **归档旧脚本**：一次性诊断/分类/注入脚本统一移入 `scripts/archive/`，根目录
  只留活跃入口。
- **仓库治理**：收紧 `.gitignore`——排除 `novel_studio/`（含 API key 的用户数据）、
  `experiments/`（一次性输出）、`.playwright-cli/`（缓存）、`opencode.json`、
  `clash-verge-*.yaml`（用户代理配置）、`*.bak*` 等；删除 136 个 `.bak*` 垃圾文件。
- `__version__` → 0.3.3。

### 0.3.2 — 语义嵌入可插拔 + MCP 落盘节流（已被 0.3.3 取代，未打 tag）

- **语义嵌入可插拔（突破词汇级上限）**：`embedding.embed_text()` 成为单点，
  `set_embedder()` 可换后端——新增 `memagent/embedders.py`：`RemoteEmbedder`
  （OpenAI 兼容 /embeddings，纯 urllib，退避重试）与 `LocalEmbedder`
  （sentence-transformers，可选 extra `embed-local`）。跨措辞等价
  （「技术栈」↔「Python 程序员」）换后端后即可命中；存量记忆按维度失配
  自动重建迁移（`from_dict`）。MCP 新增 `--embed-base-url / --embed-model /
  --embed-api-key / --embed-local` 参数。
- **MCP 落盘节流**：`retrieve` 的测试效应/再巩固改为 5 秒节流保存（显式写
  操作仍强制落盘），避免每次检索全量 JSON 写；`retrieve` 命中附带
  `recall_hint`（Cold 摘要命中时提示 `memagent_recall` 唤醒细节）。
- 版本号与 extras：`pyproject.toml` 增加 `embed-local`；`__version__` → 0.3.2。

### 0.3.1 — 检索天花板修复 + MCP 生产化

- **嵌入升级（检索天花板修复）**：`embedding.DIM` 256→1024；FNV-1a 哈希加 `_mix`
  低位混洗——旧版直接 `% 256` 取低位、末字符主导，造成跨文本系统性假相关
  （「火锅撞首都」类误命中）。现在无关文本相似度回到 0.00~0.04，短查询的
  「查不到」能被诚实识别。符号哈希的病态（伪碰撞反向抵消真信号）由
  `retrieve()` 的词汇重叠保底兜底（共享 gram 存在但余弦塌陷 <0.03 → 按 0.20
  计分）；存量 256 维向量在 `Memory.from_dict` 按维度失配自动重建迁移。
- **检索置信标注**：`retrieve` 结果带 `relevance` 与 `matched` 判据
  （`RELEVANT_TOTAL` 0.05→0.03 校准；MCP 侧 `REL_CONFIDENT=0.25`），无关命中
  不再假装查得到。
- **MCP 工具面补全**：新增 `memagent_forget` / `memagent_recall`（Cold 唤醒，含
  深藏细节）/ `memagent_find` / `memagent_start`（按主题开工注入）/ `memagent_export`
  （导出 AGENTS.md，dual 同步 CLAUDE.md）；`retrieve` 现在持久化测试效应并输出
  `notice`。服务器改用 mcp SDK 装饰器 API（各版本通用，旧构造器写法在已装
  SDK 上跑不起来）。依赖声明：`pip install memagent-local[mcp]`。
- **CLI/MCP 单一实现**：新增 `memagent/instructions.py`（决策选取、注入块、
  AGENTS.md 导出），`session_memory.py` 委托复用，两条入口不再漂移。
- **包分层文档**：README 新增分层地图（核心记忆 SDK / 认知扩展 / 领域扩展），
  MCP 与 SDK 路径只触核心层。

### 0.3.0 未发布历史

- Added per-call `max_tokens` override to `LLMResponder.respond()` and `call_responder()` (signature-filtered for legacy responder plugins).
- Raised long-form output ceilings: `AgentConfig.llm_long_max_tokens` (4096) is now used for chapter writing; the FoxTable coder passes 4096 for code generation and critique.
- Root-caused the 47% truncation rate: the responder's default `max_tokens=1024` cut VB.NET code blocks mid-flight (279 of 286 truncated cycles produced exactly the 500-char fallback) and capped novel chapters around 1,000 characters.
- Unclosed code blocks now feed the real partial code to critique instead of the first 500 characters of prose.
- Added a consecutive-failure cap for forced retrains: a domain that still truncates after 3 consecutive forced rounds falls back to normal task selection instead of looping forever.

## 0.2.1

- Added immutable release manifests, versioned runtime installs, atomic activation, and offline rollback.
- Added validated named backups with byte-preserving pre-restore snapshots and checksum verification.
- Extended CI to build, verify, clean-install, smoke-test, and retain release artifacts.
- Added proprietary package metadata, release documentation, and dependency update automation.

## 0.2.0

- Added atomic persistence, rolling backup, stale-writer detection, and file locks.
- Persisted interest, graph, growth, cognition, curiosity, analogy, social, and emotion state.
- Prevented chapter overwrites and rejected truncated chapter output.
- Added novel audits, snapshots, overwrite evidence, revision archives, and reviewed-candidate promotion.
- Raised chapter completeness to 90% of the target and rejected incomplete final sentences.
- Added safe work-title migration and fixed chapter-context discovery.
- Fixed curiosity, graph, social-learning, and growth-summary contracts.
- Preserved compatibility with minimal responder plugins.
- Added thinking-disabled retry for reasoning-only model responses.
- Added finite autonomous defaults and consecutive-failure circuit breakers.
- Added installable CLI, health checks, cross-platform CI, and security documentation.
