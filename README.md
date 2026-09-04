# memagent —— 仿人脑分层遗忘记忆系统原型

一个**零第三方依赖**（纯 Python 标准库）的记忆 Agent 原型，把人类记忆的
几个核心机制搬进代码：分层存储、遗忘曲线、检索强化、睡眠巩固、索引唤醒。

## 设计与人脑的映射

| 人脑机制 | 本系统实现 | 位置 |
|---|---|---|
| 工作记忆（前额叶） | **Hot 层**：检索 ≥ 阈值次数的记忆直接注入上下文 | `memory.py` |
| 长时记忆（皮层） | **Warm 层**：完整记忆，参与检索评分 | `memory.py` |
| 海马体索引（只存指针） | **Cold 层**：压缩摘要 + 原始内容索引，命中才唤醒 | `memory.py` |
| Ebbinghaus 遗忘曲线 | 时间衰减 `exp(−Δt/τ)` | `decay.py` |
| 测试效应（检索即复习） | 命中时 `access_count+1` 并刷新时间 | `agent.py::retrieve` |
| 查询同义扩展 | 人称互换+同义词变体，问法与记忆措辞不同也能命中 | `synonyms.py` |
| 短查询子串优先 | 短查询按内容含词重排，消除哈希碰撞噪声（开关/阈值可配） | `agent.py::retrieve` |
| 重要性（情绪/意义标记） | 关键词启发式打分，可替换为 LLM | `memory.py::estimate_importance` |
| 睡眠巩固（海马体重放） | `sleep()`：按时间顺序回放白天经历（再激活）→ 低频旧记忆聚类 → 压缩成 Cold 摘要；中断时未回放的记忆次日更模糊 | `agent.py::sleep` |
| 心游（默认模式网络） | `spontaneous_recall()`：无查询时按强度加权自发想起一条非 Cold 记忆，想起即再激活（touch+采样）并成为当晚回放候选——越想起越牢的闭环 | `agent.py::spontaneous_recall` |
| 场景重建（片段组合回忆） | `compose_scene()`：检索命中时把相关片段按经历顺序拼成连贯场景（共享主题词+时间窗门控），场景片段获得再激活与再巩固；回复以场景呈现 | `agent.py::compose_scene` |
| 回忆即重建 | `recall()`：Cold 摘要 → 重建一条 Warm 记忆（move 语义：原 Cold 移除，往返不增殖；继承 mtype/kind/history/出生时间/originals 深藏细节/再巩固修订日志 revisions，τ 与可塑性学习器都不因唤醒断层；唤醒本身计一次检索；打上 `awakened_at` 复活标记，`/memories` 据此显示「唤醒自Cold(修订=X 历史=Y)」可追溯） | `memory.py::awaken` |
| 记忆再巩固 | 回忆使记忆进入可塑状态，按重要程度微调后重新存储 | `agent.py::_reconsolidate` |
| 按类型内容钩子 | 技能类回忆一致性校验，情景类情境改写 | `checkers.py` |
| 按类型分遗忘 | 技能慢衰减 / 语义中衰减 / 情景快衰减，自动识别类型 | `memory.py::classify_memory` |
| 持续观测验证 | 每轮对话自动采样，实测τ vs 配置τ 验证贴合度 | `visualize.py::fit_report` |
| LLM 回复生成 | 检索结果注入上下文，无记忆时 LLM 直接回答（可选） | `responder.py` |
| 参数自适应 | 按实测 τ 自动校准配置 τ（EMA + 置信度门控；干净段反推 + 唤醒偏差代理两路互补） | `agent.py::learn_tau` |
| 可塑性自适应 | 按修订日志的实测因子 + 唤醒偏差观测（实测偏差相对**类型预期偏差**偏离 → 该类型可塑性越活跃，预期含类型 τ 与压缩时机，单一类型可校准）自动校准再巩固因子 | `agent.py::learn_plasticity` |
| 记忆类型画像 | τ / 再巩固因子 / 压缩阈值统一成一张配置表 | `profiles.py` |
| 情景记忆语义化 | 被反复检索的 episodic 固化为 semantic，低频反向淡化 | `agent.py::_semanticization_score` |
| 遗忘斜率预测 vs 实测 | 观测采样跟踪实际触底时刻，对比预测触底时间 | `visualize.py::floor_verification` |
| 语义化（旧经历融成概括） | 提取式摘要 + 相似记忆合并 | `compression.py` |
| 强度曲线可视化 | 预测曲线（实线）+ 实际采样（圆点）导出 SVG/CSV/JSON | `visualize.py` |

> 上表是机制清单；这些机制如何**共享状态、互相喂数据、形成完整闭环**（在线
> 响应 × 离线自维持 × 学习器参数反馈），见下文
> **「离线处理总设计」** 章节。

## 检索评分公式

```
最终得分 = 语义相似度 × 记忆强度

记忆强度 = ( w_recency·exp(−Δt/τ)
           + w_freq·(1 − exp(−n/κ))     ← n 为检索次数
           + w_importance·importance ) / 权重和
```

- **越久越淡**：时间常数 τ 控制遗忘快慢；
- **越用越牢**：每次检索命中都加分（测试效应），饱和常数 κ 防止无限增长；
- **一次重大事件也能刻进记忆**：importance 维度独立于频率；
- **Cold 摘要用摘要向量参与检索**：命中即"索引触发"，可随时唤醒底层细节。

## 查询同义扩展

字符 n-gram 嵌入对措辞敏感——记忆存"我昨天去吃了火锅"，用户问
"昨天中午用餐了吗"可能漏检。`memagent/synonyms.py::expand_query` 对查询
生成**检索变体**，`retrieve` 对每条记忆取变体相似度的最大值（原始查询恒在
变体里，故 rel 只会升不会降）：

- **人称互换**：疑问句里的"你/您"（指用户自己）→"我"——
  「您叫什么名字」→「我叫什么名字」，与「我叫小林」rel 从 0.17 升到 0.30；
- **同义词替换**：词族内罕见词替换为常见口语词（组首）——
  「用餐」→「吃」、「姓名」→「名字」、「观看」→「看」，方向固定书面→口语；
  命中词已是常见词时不生成无益变体；
- 变体数上限 8；`AgentConfig(query_expansion=False)` 一键关闭（关闭时 rel
  与旧版完全一致，向后兼容）。demo 6.6 段演示：同义词 0.23→0.33、人称互换
  0.17→0.30。

### 短查询子串优先重排（可配置）

短查询的 rel 易被哈希嵌入的**泛化命中**主导——不相干的记忆可能因为
碰撞排到最前面。`retrieve()` 对短查询自动做**子串优先重排**（内容/摘要含
查询词的记忆排最前，组内按 **rel×强度**（total）降序——低相关但高强度的
含词记忆不会压过高相关条目；大小写不敏感，与 n-gram 嵌入的归一化对齐），
所有下游入口（回复引用/对话注入/主题检索）统一受益。两个配置项：

```python
AgentConfig(rerank_short_query=True)    # 总开关，默认开；False = 与旧版一致
AgentConfig(rerank_short_len=5)         # 短词阈值：查询少于该字数视为"短"，默认 3
```

实测（`retrieve("触底时间")`，4 字）：默认阈值 3 不重排、碰撞噪声
（total 0.16）排在含词记忆（0.139）前；`rerank_short_len=5` 后子串优先，
含词记忆排最前。

> 两个机制的分工与适用场景（查询侧扩展 vs 结果侧重排、recall vs 排序、
> 配置组合与调试方法）详见 **`docs/retrieval_enhancement.md`**；其中还含
> 「多源合并簇的摘要检索」示例——sleep 合并后「词只在摘要/只在原始内容」
> 如何影响 `find_memories`（搜索面）与 `retrieve`（检索面）。

## 快速开始

要求：Python 3.10+。

### 作为 SDK 嵌入你的应用（推荐）

```python
from memagent import MemoryAgent
from memagent.llm import LLMClassifier

# api_key="" → 离线关键词分类；配 OPENAI_* 环境变量则自动启用 LLM 分类
agent = MemoryAgent(classifier=LLMClassifier(api_key=""))

agent.remember("用户偏好简洁回复", importance=0.9)   # 写入（自动分类/去重/情绪编码）
hits = agent.retrieve("用户喜欢什么格式")            # 检索（遗忘曲线评分）
report = agent.sleep()                              # 睡眠巩固（回放+分级+压缩）
```

完整示例见 `examples/quickstart.py`（含元认知校准、前瞻记忆，5 分钟读完）。

### 本地运行 / 开发

```bash
# 推荐：以可编辑模式安装产品与开发依赖
python -m pip install -e ".[dev]"

# 1) 跑脚本演示（把 τ 压到秒级，几秒内看完升降级全过程）
python demo.py

# 2) 交互式聊天（记忆自动持久化到 memories.json）
python -m memagent

# 3) 跑测试
python -m pytest tests/ -q

# 4) 上线/升级前健康检查
memagent --check --persist memories.json
```

### 产品版基线（v0.3.2）

- **可安装 CLI**：`pip install -e .` 后可直接运行 `memagent`；
- **可靠持久化**：JSON 同目录原子发布、上一版本 `.bak` 备份、损坏时自动恢复；
- **并发保护**：检测陈旧写入并拒绝覆盖；同一作品只允许一个章节写作事务；
- **完整状态恢复**：记忆、兴趣、图谱、预测、概念、技能、目标、好奇探索和类比历史同步持久化；
- **无人值守保护**：后台默认有限轮次，连续失败自动熔断，避免无限消耗 API；
- **作品保护**：章节哈希清单、全书快照、覆盖证据与修订归档；正文不足目标 90% 或未完整收句时拒绝落盘；
- **发布保障**：Windows/Linux CI、Python 3.10+、安全说明与版本迁移文档。

详细运维边界见 `PRODUCT.md`，v0.2 升级说明见 `docs/migration-0.2.md`。

正式发布与回滚：

```powershell
python -m memagent.release build --output releases
python -m memagent.release install --wheel "releases\v0.3.2\memagent_local-0.3.2-py3-none-any.whl" --runtime .runtime
python -m memagent.release rollback --runtime .runtime
```

持久化命名备份与恢复：

```powershell
memagent-backup create --persist agent_memory.json --output backups
memagent-backup restore --from "backups\备份文件.json" --persist agent_memory.json
```

完整流程见 `docs/releasing.md` 和 `docs/backup-restore.md`。

小说作品可使用独立维护命令：

```powershell
# 只读检查缺号、短章、未收句、标题与定名前异稿
python -m memagent.work_admin audit --work "works\错季锁星"

# 创建全书快照、哈希清单和历史覆盖记录
python -m memagent.work_admin protect --work "works\错季锁星" --log "works\autonomous.log"

# 审核候选稿后，归档旧章并安全发布
python -m memagent.work_admin promote --work "works\错季锁星" --chapter 8 --candidate "候选稿.md"
```

### CLI 命令

```
/help           帮助
/stats          各层记忆数量
/memories       列出全部记忆（层级、强度、重要性星标）；带关键词按内容搜索（多词空格分隔=同时包含）；唤醒自 Cold 的记忆显示「唤醒自Cold(修订=X 历史=Y)」标记（继承的再巩固修订数 + 观测轨迹条数，长生命周期可追溯）
/sleep          手动睡眠巩固（回放白天经历 + 压缩低频记忆 + 情景记忆语义化迁移）
/mind           心游：无查询时按强度加权自发想起一条记忆（再激活测试效应）
/scene <查询>   场景重建：把相关记忆片段拼成连贯场景（片段组合回忆）
/recall <id>    唤醒一条 Cold 摘要记忆
/forget <id>    彻底删除
/plot           导出强度曲线（.svg 主图 + 按类型面板 + .csv 曲线 + .csv 唤醒明细 + .json）并打印贴合度报告
/ploti          导出交互式曲线（单文件 HTML：缩放/点击高亮/层级切换）
/observe        观测一轮（所有记忆采样）并打印当前贴合度
/classify <文本> 用分类器（LLM 或关键词回退）识别记忆类型
/persona        查看当前人设与演化档案（remember_setting 写入的设定记忆）
/models         查看 LLM 模型池状态（429 自动切换次数/最近限流）
/learn          根据观测自动校准各类型 τ 与再巩固因子（睡眠巩固时也会自动触发）
  /tauplot          导出 learn_tau 两路信号（干净段/唤醒偏差）的收敛轨迹图 + 轮次明细
  /types            查看记忆类型画像（各类型 τ / 再巩固因子 / 压缩阈值 / 唤醒信号 / τ 两路信号列：干净段/唤醒方向 + 一致性）
  /signal [近N天]   唤醒信号漂移：对比最近 N 天与更早的方向一致性（默认 30 天）
/save           持久化
/quit           退出（自动保存）
```

## 分层规则

```
Warm ── 检索 ≥ hot_after_access 次 ──▶ Hot（工作记忆，直接进上下文）
Warm ── 超过 cold_after_seconds 未访问 且 低频 ──▶ Cold（压缩摘要，无损降权）
Hot  ── 闲置超过 cold_after_seconds ──▶ 降回 Warm
Cold ── /recall 命中 ──▶ 重建为 Warm
```

默认参数适合长期使用（τ = 7 天）；演示与测试把时间常数压到秒级。

## 工程决策与可替换钩子

- **嵌入**：字符 bigram/trigram 哈希到 256 维向量 + 余弦相似度。零依赖、支持中文。
  生产环境可换成真 embedding 模型，`Memory.embedding` 字段无需改动。
- **重要性打分**：关键词启发式（`estimate_importance`）。可替换为 LLM 判断：
  把对话交给模型，返回 0~1 的重要性分数。
- **回复生成**：默认模板合成（无 LLM 也可运行）；可接 `LLMResponder`
  （`memagent/responder.py`，OpenAI 兼容，与分类器同套环境变量配置）——
  `respond()` 把 `retrieve` 的检索结果注入 prompt 让 LLM **基于记忆回答**，
  无相关记忆时 LLM **直接回答**（不再只会说"不了解"）；未配 key / 网络或
  解析出错时自动回退模板回复，检索链路不受影响。
  `respond(..., max_tokens=N)` 逐调用放宽输出上限——代码生成/写章等长输出
  必须传（默认 1024 会把输出拦腰截断：VB.NET 代码块收不了栏即判截断、
  章节正文被压在 ~1000 字），`AgentConfig.llm_long_max_tokens`（默认 4096）
  已接入写章链路，`call_responder` 按签名过滤兼容不支持该参数的旧 responder。
- **去重**：新记忆与已有记忆相似度 ≥ 0.92 时合并并强化旧记忆，避免
  对话流水重复入库污染检索。
- **对话流水降权**：`kind="turn"` 的记忆检索权重 ×0.5，事实记忆更受重视。

## 记忆再巩固：回忆会修改记忆本身

对应真实记忆的**再巩固（reconsolidation）**机制：回忆使记忆进入
**可塑状态**，随后以修改后的形式重新存储——所以每次回忆都会按当下情境
微调记忆，而不是原样返回。实现（`agent.py::_reconsolidate`）：

- **可塑性 = 1 − 重要性**：重要性越高越稳定；`importance ≥ freeze_importance`
  的记忆完全**冻结**（核心记忆，向量、重要性、文本都不动）；
- **语义漂移**：记忆向量向本次回忆情境靠拢，幅度 = 可塑性 × `content_drift`。
  低重要性记忆越用越"长成"它被回忆的样子，高重要性记忆保持原样；
- **按类型缩放**（`reconsolidation_by_type`）：两个通道（drift / importance）各有
  按类型的乘数因子——**技能类 drift 0.15（回忆时高度稳定）、语义类 1.0（基准）、
  情景类 2.5（容易被情境改写）**，重要性漂移同理（技能 0.2 / 情景 1.5）；
- **重要性微调**：被高度相关的查询命中会巩固（+），弱相关轻微去巩固（−），
  受 `importance_floor` 保护。因强度公式含 importance，这同时微调了强度；
- **可塑窗口**：回忆后进入 `reconsolidation_window` 时长（默认 6 小时）的
  可塑期，期内再次回忆漂移幅度 ×(1+`labile_bonus`)，模拟再巩固窗口；
- **内容级编辑钩子按类型分流**：`content_updaters` 注册表（键为 MemType 或其
  value 字符串，类型专属优先），未注册的类型回退通用 `content_updater`（`fn(记忆,
  触发查询, 可塑性)` 返回新内容，如接入 LLM 把回忆情境融进文本；钩子收到的触发
  查询已归一化——strip + 小写，与 `retrieve()` 打分语义一致）。技能类建议配
  `checkers.consistency_checker()`——**回忆时核对一致性而非情境改写**：结论记入
  `mem.checks`（consistent / unknown / conflict / corrected），内容与向量完全不动、
  不计修订（技能回忆是验证不是吸收情境），只有判定冲突且提供 `rewrite_on_conflict`
  时才真正改写。demo 6.4 段对比：技能内容未变、修订 0 次、校验 2 次；情景内容被
  改写成"……（回忆情境:……）"。不提供任何钩子时只漂移向量，文本无损。

每次微调都会 `revision_count+1` 并记入滚动修订日志；`/memories` 会显示
修订次数与状态徽标（冻结 / 可塑 / 稳定），JSON 导出含修订数据。
可用 `reconsolidate=False` 一键关闭，保持纯检索系统。

## 按类型分遗忘曲线

真实记忆按内容类型衰减速度不同：技能（骑车、弹琴）多年不忘，情景细节
（上周三午饭）几天就淡。系统把记忆分为三类，每类独立 τ，并自动识别：

| 类型 | 默认 τ | 识别线索（示例） |
|---|---|---|
| `skill` 技能 | 60 天 | 学习/练习/学会/步骤/怎么做/做饭/编程/弹琴… |
| `semantic` 语义 | 14 天 | 定义/原理/首都/因为/所以/知识；身份偏好（我叫/我是/我喜欢） |
| `episodic` 情景 | 3 天 | 昨天/今天/去了/吃了/发生/遇到；对话流水（`kind="turn"`） |

- 自动识别：默认用 **LLM 分类器**（OpenAI 兼容接口，`llm.py`），未配置
  或出错时自动回退关键词打分；`remember(..., mtype=...)` 可手动覆盖；
- LLM 分类：设置环境变量 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`、
  `OPENAI_MODEL`）即启用，兼容 OpenAI / DeepSeek / Moonshot / 本地 Ollama 等
  任意 OpenAI 风格端点；LLM 返回 `{"type", "confidence"}` 严格 JSON，
  按内容缓存避免重复调用，对话流水不消耗 LLM 调用；关键词回退自带置信度；
  本仓库 `.env` 已配置 **OpenCode Go**（`https://opencode.ai/zen/go/v1`，
  仅 `deepseek-v4-flash`）：`_default_post` 带浏览器 UA，避免 Cloudflare
  把 urllib 误判为机器人（HTTP 403/1010）；
  验证全链路（agent → 分类/回复生成 → HTTP → 解析 → 入库 + 缓存 + 回退）：
  ```bash
  python llm_classify_demo.py
  # 已设 OPENAI_API_KEY → 直接调真实端点；未设 → 自动启动本地 mock
  # OpenAI 服务走真实 HTTP 传输链路，21 项断言全过则链路验证通过
  ```
  三种等价配置方式：环境变量（推荐，`python -m memagent` 也用它）、
  `LLMClassifier(base_url=..., api_key=..., model=...)` 构造参数、
  `MemoryAgent(classifier=LLMClassifier(...))` 注入；
- CLI 里 `/classify <文本>` 随时查看分类结果与来源（llm / keyword / turn）；
- 睡眠压缩阈值同样按类型推导：闲置超过 `cold_after_tau × τ`（默认 2×τ）才压缩进
  Cold——情景 6 天即可埋藏，技能 120 天才考虑，符合直觉；
- 兼容旧配置：显式设置 `cold_after_seconds` 时仍用绝对秒数；`tau_seconds` 作为
  未覆盖类型的回退值。
- 曲线图标题、CSV/JSON 导出均带类型与按类型 τ；`/memories` 和 `/stats` 显示类型分布。

## 429 自动切换：一直换到不限流的模型

LLM 调用（分类器 `LLMClassifier` 与回复生成器 `LLMResponder`）共用同一个
**多模型池**（`llm.py::ModelPool`）：主模型 429 时自动切换备用模型，
全部限流时等待冷却后重试整个池——直到预算耗尽或出现可用模型：

- **模型池配置**：`OPENAI_MODELS="模型1,模型2,…"`（逗号分隔，主模型
  `OPENAI_MODEL` 优先尝试）；或用 `LLMResponder(models=[...])` /
  `LLMClassifier(models=[...])` 构造参数；
- **单模型 429 → 换下一个**：限流的模型进 `failover_cooldown` 秒冷却（默认
  60s，冷却期内不再碰它），轮转到下一个模型重试，成功即用；
- **全部限流 → 等冷却重试**：整个池都在冷却时，等待**最早冷却结束**（单次
  等待封顶 `all_down_wait_cap`，默认 15s）后重试整个池，最多
  `all_down_retries` 轮（默认 2）——"一直切换到不限流的模型"；预算耗尽仍
  无可用模型时明确报错（分类器此时回退关键词规则，回复生成器抛错回退模板）；
- **可观测**：`/models` 命令（或 `responder.pool_status()`）查看当前模型、
  模型池、429 切换次数与最近限流记录（模型 + 时间）；每次切换自动记录
  `failover_count` / `rate_limited_log`；
- 适用场景：中转聚合站（one-api / new-api 等）一个端点挂多个模型、各模型
  独立限流——模型级切换即可覆盖；跨端点切换不在本设计内（见已知局限）。

## 人设与自主演化：让 agent 成为小说家

`LLMResponder(persona=...)` 支持**人设**：`persona="novelist"`（或
`小说家`/`玄幻小说家`/`仙侠小说家`）映射内置小说家人设，任意自定义文本
直接使用；`OPENAI_PERSONA` 环境变量等效。内置小说家：**擅长长篇玄幻、
仙侠，是成年角色亲密关系的成熟写作者**——可自然描写含蓄暧昧、情欲张力，
直至双方自愿的成年角色之间明确、露骨的性爱场景（写实、符合人物弧线、
服务于剧情）。同时带硬性**铁律**（写进 system prompt，任何情况下不可违反）：
性描写只发生在虚构的**成年（18+）**角色之间且**双方自愿、知情同意**；
严禁任何**未成年人**内容（含擦边暗示与拟未成年变体）；严禁**强迫、胁迫、
迷奸、非自愿、剥削**或权力不对等下的性——一旦察觉立即停止拒绝。
并带创作连续性要求（维护作品名/世界观/人物/境界体系/时间线/伏笔/章节进度）。

**自主演化**——人设不是静态的，它随记忆累积成长：

- `agent.remember_setting("主角：林尘，青州林氏旁支少年")` 写入一条
  `kind="setting"` 的设定记忆（作品/世界观/人物/境界/伏笔/进度都该这样入库）；
- `agent.persona_sheet()` 把设定记忆**按重要性降序**取前 8 条拼成
  "演化档案"；每次 `respond()` 自动把档案注入 system prompt（人设 + 档案 +
  基础提示），**没有设定时零注入**，行为与旧版完全一致；
- 效果：设定写一次，跨会话、跨重启持续生效——创作人设随小说设定
  自主演化，写下一章时模型自动"记得"主角与境界体系；
- 查看：`/persona` 命令打印当前人设 + 演化档案；
  `agent.persona_sheet()` / `agent.remember_setting()` 供编程使用；
- 启动：`OPENAI_PERSONA=novelist python -m memagent` 或
  `python chat.py --persona novelist`；`MemoryAgent(persona="novelist")`
  会自动创建配置好人设的 `LLMResponder`（未配 key 时回退模板回复）。

### 自主演化循环：反思 + 联网 → 新设定入库

`agent.evolve()` 是演化的驱动器：以**当前人设档案 + 最近记忆 + 联网搜索资料**
为上下文，让 LLM 提出新的作品设定（新人物/新伏笔/世界观补全/剧情走向），
再 `remember_setting` 入库（与已有设定自洽去重、只吸收「设定：」行、单次上限
`evolve_max_settings` 条）——下一轮 `persona_sheet()` 即包含新设定，人设随之成长：

- **联网研究**：`_research_query()` 自动取档案里的作品名（无则用类型泛词），
  `search_web()` 查 4 条资料注入演化提示（Bing 首选，DuckDuckGo 备用，纯 stdlib）；
- **触发方式**：`/evolve` 手动一轮；`chat.py --persona novelist` 与
  `python -m memagent`（配了 `OPENAI_PERSONA`）默认**每次睡眠自动演化**
  （`AgentConfig.evolve_on_sleep=True`）；库级默认关闭，避免无意图静默调 LLM；
- **无头自主模式**：睡觉前挂上 `python chat.py --auto 10 --persona novelist`，
  连续跑 10 轮 {自主演化(含联网) + 睡眠巩固}，每轮打印演化报告，收工即成长；
- 未配 LLM 时 `evolve()` 静默跳过（ok=False），不影响检索/回复主链路。

### 联网搜索

`/web <查询>` 或 `search_web(query)`：Bing → DuckDuckGo 备用，浏览器 UA
绕 Cloudflare，失败返回空列表不抛错（探索能力不影响主链路）。

离线演示（不联网，注入假 HTTP 客户端验证人设注入 + 429 切换全链路）：

```bash
python novelist_demo.py
```

## 参数自适应学习器：自动调 τ

贴合度报告能反推实测 τ，那就不必手动调参——`agent.py::learn_tau()`
把校准做成闭环：

- **信号（两路互补）**：① fit_report 从干净衰减段反推的实测 τ，与配置 τ 的
  偏差即"预测偏差"；② 唤醒偏差代理（`_tau_awakening_estimate`）——唤醒时实测
  跳升深于该类型预期偏差（`dev > expected`，expected 由模型信念 τ 在同一事件
  算出）→ 该类型衰减比信念快 → τ 下调：`τ_est = τ × (expected/dev)^gain`。
  两路按各自置信度加权合并（干净段按观测时长占比、唤醒按事件数占比）；
- **门控**：任一源充足即可更新（干净段 ≥ `tau_min_segments` 或唤醒观测 ≥
  `tau_min_awakenings`，默认 3）且偏差 > 0.5%；唤醒链路全是干扰段（无干净段）
  时，仅唤醒观测也能驱动更新；触底段（强度在 0.2 下限）不会污染估计——
  深埋唤醒实测与预期同饱和，比值 → 1 → 保守无信号；
- **EMA 更新**：`τ_new = (1−α)·τ_old + α·τ_est`，
  `α = tau_learning_rate × 置信度`，置信度 = 两源置信度之和（封顶 1，
  观测越充分越敢动，稀疏数据不反应过度）；
- **持久化**：学习结果写入 `store.meta["learned_tau"]` 随记忆落盘，
  重启自动应用；`tau_learning=False` 可关闭，保留显式配置。

睡眠巩固时会自动触发学习；CLI 里 `/learn` 随时手动执行并打印更新明细。
（demo 第 9 段：配置 τ=6s、真实 2s，学习器从观测中把 τ 单调收敛到 ≈3s；随后唤醒偏差
第二观测源演示——3 次 Cold↔Warm 往返、无干净段，仅唤醒观测把 episodic τ 从 3 天推向
真实 2 天。）

### 收敛轨迹图：两路信号按轮次互相印证

每轮学习历史记录**两路的独立估计**（干净段 τ_est / 唤醒 τ_est）与唤醒中位
比值 `dev/expected`——`/tauplot` 或 `agent.plot_tau_convergence()` 导出
**`tau_convergence.svg` + `tau_convergence.csv`**：

- 每个有学习记录的类型一张面板：上子图 τ（对数轴）画配置 τ 的 **EMA 轨迹**
  （实线，轮次 0 = 初始信念）、干净段 τ_est（紫虚线）、唤醒 τ_est（橙虚线）
  与真实 τ（灰虚线参考线，配置了 `true_tau_by_type` 时）；
- 下子图画**唤醒中位比值**随轮次逼近 1（灰虚线 = 与真实一致）——比值 > 1
  表示唤醒比类型预期剧烈（τ 应下调），趋 1 = τ 已校准；
- **互相印证的判读**：两条 τ_est 线都向真实 τ 收敛、比值同步趋 1，即两路
  独立观测（干净衰减段 vs 唤醒深度）给出同一结论——学习不是单源自说自话。

**信号方向写入学习历史（可复盘）**：每次更新把当时实际使用的唤醒信号原始
值记进历史行——learn_tau 历史 11 列（`_learn_history`：…/比值/dev/expected）、
learn_plasticity 历史 10 列（`_plasticity_history`：…/dev/expected/比值），
`tau_rounds()` / `tau_convergence.csv` 同步带 `awakening_dev` /
`awakening_expected` 列。dev > expected = 该类型埋得比信念深 → 下调 τ（或
上调可塑性）的方向依据，比值随校准趋 1——每轮为何动、动多大幅度都可追。

实测（demo 第 9.5 段：真实 2 天、信念 3 天，10 轮）：EMA 轨迹
`[2.93 → 2.14] 天`、比值 `[1.181 → 1.102]`，末轮干净段 τ_est=2.00 天 /
唤醒 τ_est=2.00 天——两路同时指向真实值；信号复盘：首轮 `dev=0.455 vs
预期 0.385`（> → 下调）→ 末轮 `dev=0.443 vs 预期 0.394`（趋 1 = 已校准）。
旧格式学习历史（无独立源列）自动降级（源列缺省，EMA 轨迹仍可画）。

### τ↔可塑性联合估计：一次唤醒事件同时更新 τ 与 drift

唤醒跳升（dev）同时编码**τ 失准**与**可塑性**：dev 按该类型实测可塑性因子
缩放（`awakening_plasticity_gain`），expected 按信念因子缩放——比值 =
[τ 分量] × [可塑性分量]，单事件两个未知数学上不可分离。`_joint_awakening_estimates`
用**双向跨轮耦合**拆信号：

- **τ 通道**：比值先剥掉**上一轮估计的可塑性因子**再反演 τ（可塑性收敛 →
  剥因子 → 实测刻度 → 比值变纯 τ）——纯可塑性失准不再被误读为 τ 失准；
- **drift 通道**：拿 τ 解释不了的残余——用去可塑性后的 dev 按衰减公式**精确
  反演 τ 参考**，重算校正后的预期跳升 → 残余 → 调制反演回 p_est。纯 τ 失准
  时残余归零（消除旧独立代理的双计数——旧代理把整个比值同时判给 τ 与 drift）；
- **互相加速**：每轮 `learn_tau` + `learn_plasticity` 同时更新——τ 的收敛
  （跨轮 τ 参考）清洗 drift 残余，drift 的收敛（跨轮可塑性估计）清洗 τ 的
  比值。观测层唤醒元组扩为 6 列（埋藏时长 Δt、埋藏时检索次数）供精确反演。

实测（demo 第 9.7 段：真实 τ=2 天 + drift=3.5，信念 3 天 + 1.0，8 轮）：
τ EMA `[2.81 → 2.17] 天`、唤醒比值 `[2.686 → 1.158]`、drift EMA
`[1.51 → 3.16]`——两路同时逼近真实值。识别边界（文档化的诚实记录）：单事件
无法完全分离两路，首轮无上一轮可塑性知识时退化为顺序归因；可塑性估计滞后
时短暂互扰，随收敛消融。`joint_awakening=False` 回退两路独立代理（保留旧
双计数语义供对照）。

## 再巩固因子自适应：自动调可塑性

τ 学习器解决"遗忘多快"，再巩固因子学习器解决"回忆时改多狠"——
`agent.py::learn_plasticity()` 与 `learn_tau` 同构，把可塑性校准也做成闭环：

- **信号**：修订日志现在每行记录事件发生时的类型与实际应用的 drift/importance
  因子（`reconsolidation_by_type` 的隐藏真实版本 `true_reconsolidation_by_type`
  生效时即为真实环境值）——它偏离配置因子的程度即"预测偏差"；
- **第二信号——唤醒偏差观测**：`recall()` 每次唤醒时记录实测跳升 − 模型延续
  预测的偏差（`awakenings` 四元组 [时间戳, 实测偏差, 类型预期偏差, 类型]）。
  学习器换算成漂移因子代理样本喂给同一事件池：**实测偏差相对该类型预期偏差**
  （同一唤醒事件、同一状态，只把 τ 换成模型信念算出的预期跳升——含类型 τ
  与压缩时机，技能类预期轻、情景类预期重）偏离 → 唤醒比类型预期更剧烈 →
  该类型可塑性上调，反之下降。预期来自模型而非跨类型对比，**单一类型也能
  校准**（旧全局中位数锚的限制已移除）；深埋时实测与预期同饱和于强度下限
  → 比值趋 1 → 保守无信号（无偏）。唤醒观测随 Cold↔Warm 往返继承，多次
  唤醒的信号累积进事件池（`plasticity_from_awakenings=False` 可关）；
- **门控**：每类型每通道事件数 ≥ `plasticity_min_events`（默认 3）且偏差 > 1%
  才更新；旧格式修订日志（无因子记录）自动跳过；
- **EMA 更新**：`factor_new = (1−α)·factor_old + α·median(实测)`，用中位数抗
  离群事件，`α = plasticity_learning_rate × 置信度`，置信度 =
  `min(1, 事件数 / (2×min_events))`；
- **安全网**：因子夹在 `[plasticity_min, plasticity_max]`（默认 0~5）；
- **持久化**：只落盘显式配置过的通道到 `store.meta["learned_plasticity"]`，
  重启自动应用，未学习过的类型保持默认；`plasticity_learning=False` 可关闭。

与 `true_tau_by_type` 同理：未设置 `true_reconsolidation_by_type` 时实际即模型
（自洽、无可学），设置后即可验证并校准配置因子。睡眠巩固自动触发，CLI `/learn`
同时打印 τ 与因子的更新明细。
（demo 第 11 段：真实 drift 2.5 / importance 1.5、配置 1.0/1.0，6 次回忆后学习
4 轮，drift 轨迹 `[1.45, 1.76, 1.99, 2.14]` 单调逼近真实值。）

## 睡眠回放：白天经历按时间顺序再激活

对应真实记忆的**锐波涟漪（sharp-wave ripple）**——睡眠时海马体按时间顺序
重放白天的轨迹。系统在 `sleep()` 第一阶段做回放（先于迁移/压缩）：

- **候选**：非 Cold、非对话流水、`last_access` 在 `replay_window_seconds`
  （默认 24 小时）内活跃过的记忆——它们才是"白天经历"；
- **顺序**：按 `last_access` 升序（经历时间从早到晚）重放；
- **每次重放 = 一次再激活**：`access_count +1` + 记录观测采样——强度微调上升、
  语义化评分获得使用事件贡献（被反复回放的情景经历固化为语义）、检索次数
  可能因此超过压缩上限而逃过本次压缩（被回放 = 更牢）；
- **重放不改 `last_access`**：否则每次睡眠都重置衰减时钟，记忆变得不朽；
- **睡眠中断**：`sleep(duration=秒)` 传入实际睡眠时长，按
  `replay_per_second`（默认每秒 1 条）折算重放预算，只回放按时间顺序靠前的
  部分——**未回放的候选次日更模糊**（`importance × replay_fog_factor`，默认
  0.9；设 1.0 可只强化不惩罚）。`replay=False` 一键关闭回放。

`/sleep` 报告回放 / 未回放 / 模糊条数；demo 第 3 段演示完整睡眠回放与中断模糊
（2 条白天经历只睡 0.5s → 预算 0 条 → 全部未回放，重要度 0.30 → 0.27）。

## 心游：无查询时的自发想起

对应真实记忆的**默认模式网络（DMN）/ 心游**——安静时大脑自发采样记忆库，
不靠外部线索。`spontaneous_recall()` 按**强度加权**随机想起一条非 Cold 记忆：

- **权重 = 当前强度**：越牢的记忆越容易被想起；触底记忆（强度 0.2）偶尔也
  会冒出来（同权参与，只是概率低）；
- **被想起 = 一次再激活**：`touch` 刷新时间 + `access_count +1` + 观测采样——
  "反复突然想到的事情记得特别牢"的测试效应；无查询上下文，不触发再巩固
  内容改写、不升级 Hot；
- **闭环**：想起刷新时间 → 当晚成为睡眠回放候选 → 回放再激活 → 更牢——
  心游 → 想起 → 再激活 → 回放 的自增强循环；
- **Cold 深藏记忆不参与**（需线索才能唤起，见 `recall()`）；`rng` 可注入
  （测试 / 对照实验确定性），空记忆库返回 `None`。

测试锚定：500 次全新 agent 采样，强度 ≈0.88 vs 触底 0.2 的记忆被想起频率比
> 2（理论 ≈4.4）；被想起的记忆次数 +1 / 时间刷新 / 采样 +1，随后进入当晚回放。

## 场景重建：把相关片段拼成连贯场景

人脑回忆的是**片段组合**（场景重建）而非单条事实——安静时的场景闪回、
睡前对白天的运转，都是把多条相关记忆按经历顺序重新组合。`compose_scene()`
在检索命中时做同样的组合（`respond()` 自动启用，命中场景时回复直接展示
连贯场景；CLI 可用 `/scene <查询>` 手动重建）：

- **种子**：检索命中（排除对话流水）；
- **扩展**：与任一种子"相关"的其它记忆——相关度 = 嵌入余弦与 n-gram 共享
  度取大（字符嵌入对措辞不同的相关片段相似度低，共享主题词补信号），且
  **必须共享至少一个 n-gram**（哈希余弦在 0.1~0.25 有碰撞噪声，不要求
  共享词会把措辞无关的片段拉进场景），并满足 `scene_time_window` 时间窗
  （跨年片段不属于同一场景）；
- **组合**：按经历顺序（`created_at`）从早到晚排序，赋予 开头/中间/结尾
  时序角色，叙事用时序连接词拼接（先是…；接着…；最后…）；
- **整体度量**：`strength` = rel 加权平均强度（场景显著性），`coherence` =
  片段间平均最大相关度（连贯性）；
- **测试效应**：被纳入场景的扩展片段也获得 touch + 观测采样（回忆即强化），
  `scene_reconsolidates=True`（默认）时同样走再巩固——每次重建都是对片段
  的一次微调（可塑窗口语义）；Cold 摘要片段标注 `[Cold 摘要]`，叙事用摘要
  文本，可 `/recall` 唤醒细节；
- **可配置**：`scene_reconstruction`（总开关）/ `scene_similarity`（相关阈值）/
  `scene_max_fragments`（上限，种子优先）/ `scene_time_window` /
  `scene_reconsolidates`。相关片段不足 2 条返回 None——单条命中只是记忆，
  不是场景。

测试锚定：同主题片段（共享「西湖」）拼成 3 片段场景且无关记忆被排除；
叙事按 created_at 排序；阈值抬高后弱相关片段退出场景；5 年前的片段被
时间窗排除（相似度门控本会放行）；扩展片段 touch/采样/再巩固精确可测。

## 类型迁移：情景记忆语义化

真实记忆不是静止的：**被反复回忆的情景会逐渐固化成语义**（"我经常去爬山"
最终替代 50 次具体的爬山经历），而不再被使用的语义又会淡回情景。系统在
每次 `sleep()` 时做双向迁移：

- **语义化评分**（`agent.py::_semanticization_score`）：无需额外状态，直接从
  观测历史推导——相邻快照间 `access_count` 增大即发生一次"使用"（检索命中/
  去重强化/升级），按距离当前的时间指数衰减加权，评分 = 近期检索事件的加权和；
- **episodic → semantic**：评分 ≥ `semanticize_threshold`（默认 3.0，≈ 近期
  3 次检索）→ 固化为语义类，τ 由情景的快常数换成语义的中常数（记忆更持久），
  `mtype_confidence` 清空（类型改由迁移决定）；
- **semantic → episodic**：评分 < `desemanticize_threshold`（默认 0.8）且
  `access_count ≥ 2` → 淡化为情景类。`access_count ≥ 2` 保证**从未被使用过
  的新事实不会一觉醒来就翻转**；
- **双阈值滞回**（3.0 / 0.8）避免评分在阈值附近来回振荡；
- 对话流水（`kind="turn"`）和 Cold 摘要不参与迁移；`semanticize=False` 一键关闭。

每次迁移记入 `Memory.migrations` 日志并随 JSON 导出；`/memories` 显示迁移次数，
`/sleep` 报告迁移条数，仪表盘详情面板显示语义化评分。
（demo 第 10 段：情景记忆被检索 4 次后睡眠 → 固化为 semantic、τ 8s→30s、强度回升；
停止检索 8 秒后再次睡眠 → 淡回 episodic，完整双向迁移。）

## 离线处理总设计：在线响应 × 离线自维持的完整闭环

人脑的记忆处理大量发生在**离线状态**——静息时心游（默认模式网络）、睡眠时
锐波涟漪重放——而 memagent 最初是纯在线系统（有查询才检索、有对话才记录）。
三个离线机制（心游 / 睡眠回放 / 场景组合）把"静息期处理"补了进来。本节的
核心是：它们不是三个孤立功能，而是与在线检索 / 压缩 / 学习器共享同一套
状态、互相喂数据的**完整闭环**。

### 分工原则

| | 在线（响应式） | 离线（自维持） |
|---|---|---|
| 触发 | 查询/对话来才动（被动） | 系统自己运转（主动）：静息想起、定时睡眠、收工整理 |
| 职责 | 读状态做响应式决策：检索排序、注入回复、记录对话 | 写状态做批量整理：再激活、压缩、迁移、校准参数 |
| 代表 | `retrieve` / `respond` / `remember` | `spontaneous_recall` / `sleep`（回放+迁移+压缩）/ `compose_scene` |
| 学习器角色 | 无（只消费） | `sleep()` 末尾自动跑 `learn_tau` / `learn_plasticity` |

场景组合是**半在线**：由查询触发，但做的是离线式重组（扩展相关片段、按经历
顺序重排）——它是连接两条路径的桥。

### 共享状态脊：所有机制读写同一组字段

在线与离线之所以能闭环，是因为它们**读写的是同一份记忆状态**，没有各自的
私库：

| 字段 | 谁写 | 谁读 |
|---|---|---|
| `access_count` / `last_access` | 检索命中、心游想起、睡眠回放、场景扩展、`recall` 唤醒、去重强化 | 检索排序、压缩门槛（低频才压缩）、语义化评分 |
| `history`（观测轨迹） | `_record_sample`：创建/检索/回放/场景/唤醒/每轮 `_observe` | 强度曲线、`fit_report`（干净段反推 τ）、语义化评分、学习器 |
| `importance` | 再巩固漂移、睡眠中断模糊（×fog 系数）、去重取大 | 强度公式、再巩固幅度（1−importance）、冻结判定 |
| `tier`（Hot/Warm/Cold） | 升级（检索次数达标）、降级（闲置）、压缩、唤醒 | 检索加成（Hot）、压缩候选、`recall` 候选、心游排除（Cold 不参与） |
| `created_at` | 写入时 | 场景时序排序、时间窗 |

### 六个闭环

```
        在线路径（响应式）                    离线路径（自维持）
   ┌────────────────────────────┐      ┌─────────────────────────────┐
   │ 用户输入→retrieve→命中      │      │ 心游：想起→touch→采样        │
   │ →touch/采样/再巩固/Hot升级  │      │   ↓（时间刷新）              │
   │ →compose_scene→场景叙事回复  │      │ 当晚回放候选                 │
   │ →remember(对话)→_observe    │      │   ↓                         │
   └─────────────┬──────────────┘      │ sleep：回放→迁移→Hot降级     │
                 │                      │   →压缩成Cold→_observe       │
                 └──────────┬───────────┘   →learn_tau→learn_plasticity│
                            │              └─────────────┬───────────────┘
                    ┌───────▼────────────────────────────▼───────┐
                    │   共享状态脊 + 观测流（history/awakenings/   │
                    │   revisions）→ 学习器 → 参数 → 影响下一轮行为  │
                    └─────────────────────────────────────────────┘
```

**环 A 测试效应（在线自增强）**：命中 → `touch`（次数+1、时间刷新）→ 强度↑ →
排序更靠前 → 更容易再次命中——"越用越牢"。

**环 B 心游自增强（离线自增强）**：`spontaneous_recall` 按强度加权想起 →
`touch` → 强度↑ → 权重↑ → 更容易再次想起；且 `touch` 刷新 `last_access` →
记忆进入当晚回放窗口 → 回放再激活 → 更牢。心游 → 想起 → 再激活 → 回放，
解释了"反复突然想到的事情记得特别牢"。

**环 C 睡眠巩固（在线→离线→压缩）**：白天的在线使用沉淀在状态字段里 → 晚间
`sleep()` 按经历时间序回放（再激活）→ **未回放的候选次日模糊**（睡眠中断）→
闲置超过 2×τ 且低频的 Warm 聚类压缩成 Cold 摘要（空间释放）。回放先于压缩：
被回放的记忆可能因次数达标而**逃过本次压缩**（被回放 = 更牢）。

**环 D 唤醒往返（Cold→Warm→Cold 不增殖）**：查询命中 Cold 摘要 → `recall()`
以 **move 语义**唤醒（原 Cold 移除、继承 history/revisions/awakenings/
`awakened_at` 复活标记）→ 观测采样 + 唤醒偏差记录 → 再次闲置 → 再次压缩回
Cold。往返不产生记忆增殖，且唤醒信号随往返继承累积（学习器不断层）。

**环 E 观测→参数（两个学习器）**：所有在线/离线活动都写观测流（`history` 采样
+ `awakenings` 四元组 + `revisions` 修订日志）→ `sleep()` 末尾自动跑：
`learn_tau` 合并两路信号（干净段反推 + 唤醒偏差：实测跳升深于类型预期 → τ
下调）、`learn_plasticity` 合并修订日志 + 唤醒偏差（可塑性代理）→ EMA 更新
参数 → 参数**反哺**：τ 决定衰减曲线、压缩时机（`cold_after_tau×τ`）、唤醒深度
（类型预期偏差）；因子决定再巩固漂移幅度——观测 → 参数 → 行为 → 新观测。

**学习器响应演示（收工验证第 ⑨ 节）**：`--sync --eval` 扫描到高一致性信号
（某类型两路方向一致且非持平）时，用**真实记忆库**跑一次 `sleep()`——末尾
自动触发两个学习器，逐类型打印校准前后对比（`τ 3.000 → 2.850 天（Δ-0.150）·
drift 2.500 → 2.500（Δ+0.000）`）；真实库信号不足（最常见：决策记忆库几乎
不产生唤醒/干净段观测）时回退**受控合成 agent**（真实 τ=2 天 vs 配置 3 天、
真实 drift=3.5 vs 信念 1.0 + 4 次唤醒观测），演示同一机制：单次 sleep 即
`τ 3.000 → 2.749（Δ-0.251 → 逼近真实 2.0 天）· drift 1.000 → 1.676
（Δ+0.676 → 逼近真实 3.5）`——信号 → 响应 → 参数变化的完整闭环可见。
sleep 的改动只发生在内存（`--sync` 已把沉淀结果落盘在前），不污染真实库。

**环 F 语义化（类型演化）**：在线检索 / 回放 / 心游的使用事件都会让
`access_count` 增长（`_semanticization_score` 从 history 推导）→ `sleep()` 时
情景固化为语义（τ 变小变慢）→ 衰减慢、强度高 → 检索更靠前、更难压缩——
"越用越概括、越概括越持久"，且 `semanticize` 与 `replay`、`tau_learning` 等
开关互相独立（实验脚本可分别隔离）。

### 调度与触发

| 时机 | 动作 |
|---|---|
| 每轮对话 | 在线检索 + 场景组合 + `_observe` 全记忆采样；每 `sleep_interval_turns`（默认 8）轮自动 `sleep()` |
| 手动 | CLI `/mind`（心游）、`/sleep`（回放+迁移+压缩+学习器）、`/scene <查询>`（场景重建）、`/recall <id>`（唤醒）、`/learn`（只跑学习器） |
| 收工 | `session_memory.py --sync --eval`：决策沉淀 + AGENTS.md 导出 + 唤醒链路连续性检查 + awakenings 信号统计 + **τ 学习器健康检查**（两路信号方向一致性）+ **学习器响应演示**（高一致性信号 → sleep() 校准前后对比） |
| 验证 | `recall_curve_check.py`（合成判别场景 + `--real` 真实持久化场景：21 条真实决策走完整生命周期验证曲线无缝衔接）；`experiment.py` 控制变量 |

### 与在线检索 / 压缩 / 学习器的接口矩阵

| 离线机制 | 入口 | 对记忆库的动作 | 喂给学习器 | 被谁消费 |
|---|---|---|---|---|
| 心游 | `spontaneous_recall()` | 想起一条非 Cold：touch + 采样 | 间接（access 增长 → 干净段 / 语义化评分） | 当晚回放候选 |
| 睡眠回放 | `sleep()` 阶段 0 | 时间序再激活（access+1+采样）；中断 → 未回放模糊 | 采样进干净段（`fit_report`） | 逃压缩、语义化评分 |
| 场景组合 | `compose_scene()` | 扩展片段 touch + 采样 + 再巩固（可关） | `revisions` 修订日志（learn_plasticity） | 回复叙事（模板/LLM 注入） |
| 压缩 | `sleep()` 阶段 3 | Warm→Cold：聚类 + 摘要 + originals 无损保留 | 无直接 | `retrieve` 摘要命中 → `recall` 唤醒 |
| 唤醒 | `recall()` | Cold→Warm（move 语义 + 全量继承） | `awakenings` 四元组 → learn_tau 第二源 + learn_plasticity 唤醒池 | 再压缩（往返不增殖）、曲线连续性 |

**一句话总结**：在线机制把"当下"写进状态，离线机制把"过去"整理进状态，
学习器把"状态的历史"翻译成"未来的参数"——三者共享同一份记忆，因此任何
一次检索 / 一次想起 / 一夜睡眠都会沿着状态脊影响后续的一切。

## 类型行为对照实验

`experiment.py` 是一个控制变量实验脚本：**同一批记忆**（技能/语义/情景/低频
各一条）、**同一检索序列**、**同一时间线**，分别在 4 组类型参数下跑完整生命
周期（写入 → 检索 → 衰减 → 再巩固 → 睡眠压缩 → 语义化迁移），输出对比报告：

```bash
python experiment.py            # 打印报告
python experiment.py --save experiment_report.md  # 同时导出 md
```

四组对照：**A 基线**（技能慢/语义中/情景快 + 因子 0.15/1.0/2.5）、**B 无区分**
（全 τ 相同、因子全 1.0）、**C 全冻结**（因子全 0）、**D 反转**（τ 方向错置）。

- **可注入时钟**：`MemoryAgent(now_fn=...)` 替换内部 `time.time()`（含
  `Memory.touch`/`store.add`/`awaken` 的时间戳），实验用模拟时钟确定性快进，
  秒级参数代表"数天"，不依赖真实 sleep、可复现；
- **报告内容**：每组逐记忆生命周期摘要（类型/强度/检索/修订/重要性/层级/迁移）
  + 睡眠报告（压缩条数/迁移条数）+ 组×指标对照表 + 自动生成的结论；
- **实验设计要点**：零检索记忆的内容与所有查询零 n-gram 共享（避免哈希嵌入的
  泛化命中污染对照组）；查询全部用与目标高重叠的完整句子（命中不依赖再巩固
  漂移，四组行为对称）；语义化阈值校准到 2.5（避开 3 次检索评分≈3.0 的浮点边界）。

实测结论示例：类型区分让技能/情景遗忘速度拉开（A 技能 0.46 vs B 0.29）；
冻结因子后修订归零但遗忘曲线不变（修改与遗忘正交）；τ 反转时技能反而最快
遗忘（0.20）；压缩阈值随 τ 缩放（A 埋藏闲置情景、D 埋藏闲置技能）；慢衰减
记忆因强度高更易被泛化命中而自我强化。

## 记忆类型画像

三种类型各自独立的三条行为轴——**遗忘多快**（τ）、**回忆改多狠**（再巩固
drift / importance 因子）、**闲置多久埋藏**（压缩阈值）——在真实大脑里是
同一类记忆的内在属性，`memagent/profiles.py` 把它们聚合成一张"画像"：

| 类型 | τ | drift | importance | 压缩阈值 |
|---|---|---|---|---|
| 技能 | 60 天 | 0.15 | 0.20 | 120 天（2×τ） |
| 语义 | 14 天 | 1.00 | 1.00 | 28 天（2×τ） |
| 情景 | 3 天 | 2.50 | 1.50 | 6 天（2×τ） |

- `profiles.type_profiles(cfg)` 生成全部类型的 `TypeProfile`（τ 取学习器校准
  后的有效值，压缩阈值按绝对模式或 `cold_after_tau × τ` 推导并给出倍数）；
- **唤醒信号并入画像列**：`type_profiles(cfg, awakening_signal)` 可选传入
  `awakening_signal_stats(agent)`（memagent/agent.py 的单一事实源，CLI /
  types、仪表盘、收工验证共用）——每类型追加**「唤醒信号（实测）」列**：
  方向箭头 + 一致性 + 事件数（`↑上调·100%（3条）` / `↓下调·100%（2条）` /
  `无观测`），配置画像与实测信号同表对照，一眼看出"配置 vs 行为"是否相符；
- CLI `/types` 打印画像表格（含唤醒信号列 + **τ 两路信号列**：干净段 / 唤醒
  方向箭头 + 一致性徽章 ✔一致 ✘冲突 △单源 —无信号——与仪表盘、CSV 合表同源，
  终端 / 仪表盘 / CSV 三处输出一致）；demo 第 12 段展示；
- 仪表盘"记忆类型画像"面板（七列表格：类型 / τ 遗忘速度 / drift /\
  importance / 压缩阈值 / 埋藏时机 / 唤醒信号——方向着色：上调红 / 下调青 /
  持平灰 / 无观测浅灰），数据嵌入仪表盘 JSON；
- **τ 两路信号健康检查合表**：画像面板再追加三列——`干净段` / `唤醒` 方向
  箭头（↓红=应下调 / ↑青=应上调 / =灰=已校准 / —=无数据）+ `一致性` 徽章
  （✔一致绿 / ✘冲突红 / △单源灰 / —无信号浅灰），tooltip 带干净段 n / 唤醒
  n / 实测τ与配置τ / 中位比值 / 行动建议（τ↓ τ↑ 需检查 需补观测 已校准）——
  单一事实源 `agent.tau_learner_health`（与 `--export-signals` 的 CSV 合表
  同源），配置画像、实测信号、学习器健康三表同屏对照；
- **行动徽章列**：画像面板再加 `行动` 列——suggest_adjust 徽章（`τ↓`红 /
  `τ↑`青 / `⚠需检查`橙 / 其余灰，tooltip 带语义与置信度）；**点击徽章与下方
  信号漂移行同类型条目双向高亮联动**（再点 / Esc 取消）——"该类型建议怎么调"
  与"信号是否随时间漂移"一眼对照；
- **一致性徽章 → 主图类型唤醒联动**：点击 `✘ 冲突`（或任一一致性徽章）→
  主图高亮该类型**全部唤醒点**（非该类型压暗至 0.22），悬浮 callout 逐条列出
  该类型唤醒事件（记忆 / ratio / 方向），**与干净段方向相反的事件标橙 + 「← 与
  干净段相反」**——直接定位是哪几起事件造成两路冲突；事件行可点击展开单条
  双条，再点 / Esc 取消；
- **冲突类型 ⚠ 行 + 两路证据展开**：`health.warnings` 非空的类型，画像行左侧
  描橙边 + 底色高亮 + ⚠ 标记——点击 ⚠ / 整行展开隐藏的两路证据行（`① 干净段
  evidence ② 唤醒 evidence → 排查建议`，与告警 JSON 同源）；点击该类型一致性
  徽章联动主图时证据行**同步展开**（`linkTypeAwakenings` 内 `setWarnEv`），再点
  / Esc 全部收起——"冲突类型该往哪调"与"是哪两路证据矛盾"同屏对照；证据行末尾
  列出**冲突成因事件明细**（记忆预览 + `[行 k]` CSV 行号（行号算法收敛于
  agent.py 单一实现，导出 JSON / 仪表盘 / 终端打印三处同源）+ 比值 + 方向
  箭头），**点击任意事件定位主图对应唤醒点**（展开 dev vs expected 双条 + 信号
  方向 callout），callout 同时附**原始 CSV 行预览**（与导出 events CSV 同列：
  memory_id, mtype, ts, ts_relative_seconds, dev, expected, ratio, dt_seconds,
  retrievals_before，标注 `[行 k]`；六元组事件才有后两列）；**Shift 点击多选事件**
  → 聚合面板实时显示
  选中事件方向分布 + 干净段方向 + 移除后剩余事件中位比值/方向，判定「✔ 移除后
  两路一致——冲突消除 / ✘ 移除后仍冲突 / — 观测不足」——直接在仪表盘验证"去掉
  这批事件后两路信号是否一致"；聚合面板带**全选 / 选反向 / 清空**快捷按钮，
  方向占比用**三段色条**可视化（↑青 / ↓红 / ＝灰按占比等比例显示，取代纯文本
  百分比；图例保留计数、悬浮显示精确 `n/N (pct%)`）——**双条显示**：全体条 + **选中集
  分布条**（选中事件的方向占比，0 选中显示无数据），全体条上选中覆盖到的方向段
  叠加**蓝色描边**（inset 环 + 悬浮标注「选中 n 起」），选反向/全选/清空/Shift 逐条
  同步变化；**每个色段可点击**（`data-dir`，悬浮提示「点击色段只圈出该方向事件」）
  ——点击只圈出该方向的事件（清空重选，与「选反向」的 `dir ≠ clean` 判定互补），
  移除此方向后两路是否一致即刻可见——**选反向**一键圈定与
  干净段相反方向的事件（`dir ≠ clean`，与一致性徽章 callout 的 clashes 标橙
  共用 `_evDir` 同一判定），冲突成因两步操作变一步；全选/清空/选反向/Shift
  逐条共用同一刷新路径（`refreshConflictSel`）——warn-ev-row 高亮、聚合面板与
  callout 的 CSV 行预览（已选/未选徽章）三处同步更新，面板内嵌**选中集 CSV
  行预览**逐条列出选中事件的 CSV 行（行号对应导出 events CSV），Esc 清空多选
  并重置面板基线；**选反向圈定的选择集一键生成 `--exclude-events` 参数串**
  （`memory_id:序号,...`，面板实时显示 + 证据行顶部**「复制 --exclude-events」
  按钮**写剪贴板）——仪表盘圈定 → CLI 剔除重判 → JSON 落盘全程免手抄；
- 静态 JSON 导出（`memories_curves.json`）同样带 `profiles` 字段（含
  `awakening_signal` 原始统计 + `signal_text` 列文本），并**与仪表盘一致带
  顶层 `health` 合表**（`by_type` 含干净段/唤醒方向、一致性、`suggest` 行动建议、
  `confidence` 置信度 + `summary`）——配置画像、实测信号、健康检查三表同屏可查。

**时间窗与信号漂移对比**：`awakening_signal_stats` 支持时间窗参数——
`window_seconds`（只看最近 N 秒，相对 now）或绝对窗口 `since`/`until`（事件
时刻为 awakenings[0]，可组合出任意的时段切片）；`awakening_signal_periods
(agent, recent_seconds)` 把唤醒历史切成**最近 N 天 vs 更早**两段，逐类型对比
方向一致性，判定信号是否随时间漂移：

| 判定 | 含义 |
|---|---|
| 稳定 | 两段主导方向一致，一致性差 < 0.2 |
| 方向翻转 | 早期 ↑上调 → 近期 ↓下调（或反之）——类型行为发生了真实变化，需重新审视配置 |
| 一致性变化 | 方向未变但一致性差 ≥ 0.2 |
| 仅近期/仅早期有观测 | 单段有事件，无法对比 |

- CLI 新增 `/signal [近N天]`（默认 30）打印漂移对比表——早期与近期结论一致 =
  校准方向稳定，翻转 = ⚠ 需关注；
- 仪表盘画像面板下方新增「信号漂移（近30天 vs 更早）」提示行（近↑75% 早↓100%
  ⚠方向翻转，漂移红 / 稳定绿），数据嵌入仪表盘 JSON；
- `memories_curves.json` 新增顶层 `signal_drift` 字段（含 now / 分界时刻 /
  各类型两段统计与判定）。

**信号统计导出（--export-signals）**：`session_memory.py --export-signals 基名`
（可单独运行，或配合 `--sync --eval` 在收工验证后追加）把第 ⑦ 节的信号统计
写成三份文件——`{基名}.json`（完整 `stats` + `periods` 漂移对比 + `events`
逐事件明细 + `health` 健康检查，含 now / 窗口）、`{基名}.csv`（每类型一行
全字段自包含：事件数、dev/expected 三数（min/中位/max）、比值、方向计数
与主导方向、一致性、近期 vs 早期两段统计、漂移判定与一致性差、**τ 两路信号
方向一致性列**（干净段 n / 实测 τ / 配置 τ / 方向、唤醒方向、`tau_consistency`
= agree 一致 / conflict 冲突 / one_sided 单源 / no_data 无信号——与收工验证
第 ⑧ 节健康检查同源，信号导出与健康检查合表）、**`suggest_adjust` 行动建议列**
（agree 同向非 flat → `τ↓`/`τ↑` 直接给调整方向、agree 双 flat → `已校准`、
conflict → `需检查`（先排查观测污染再调参）、one_sided → `需补观测`、
no_data → `无信号`——外部工具按此列过滤即可得到待处理行动清单）、
`{基名}_events.csv`
（**事件级明细**：每条唤醒事件一行——来源记忆 id、唤醒时刻类型、绝对时间戳
ts、相对时间、dev / expected / 比值（>1 = 唤醒比类型预期剧烈）、六元组日志
的埋藏时长与检索次数，按事件时刻排序）——外部工具（Excel / pandas / R）
无需连表即可直接分析类型可塑性与信号随时间的变化；事件级 CSV 保留绝对
时间戳与来源记忆 id，可按**任意时间窗自行重切片**（如只看某类型近 7 天、
按记忆聚合、按埋藏时长分桶），不受聚合行的窗口限制。

**导出 → 验证闭环**：`recall_curve_check.py --awakened {基名}.json` 直接吃
`--export-signals` 的导出文件——自动识别顶层 `events`，从导出 JSON 挑一条
**多次唤醒**记忆（事件最多）做逐次标注：逐次打印 dev vs expected + 比值 + 信号
方向（含埋藏时长/检索次数），并渲染 `recall_curve_awakened_export.svg`
（事件时间线：红条 dev / 青条 expected 从基线升起 + 顶部菱形 + 信号徽章，
无强度轨迹时的降级标注图）。导出 → 验证一步闭环，无需回查记忆库；导出中无
多次唤醒记忆时明确跳过（不合成，避免误导）。

**冲突类型自动告警**：`--sync --eval --export-signals`（或单独
`--export-signals`）导出时若检测到某类型两路信号**冲突**（干净段说应下调、
唤醒说应上调），打印 `⚠ 需排查类型（两路信号冲突）` 告警并附**两路原始证据行**
——干净段（n 条 / 实测τ vs 配置τ / 方向）+ 唤醒（n 条 / 中位比值 / 方向）+
排查提示（检查观测污染 / 事件注入）——收工验证直接指出需排查的类型；无冲突
类型时静默。**同一份告警同时写进导出 JSON**：`health.warnings` 数组（`mtype`
+ `clean_evidence` / `awakening_evidence` 两路原始证据 + `suggestion` 排查建议）
——CI / 外部工具按 `health.warnings` **非空直接判定红灯**，无需解析终端文本；
`--export-signals` 的 `{基名}.json`、`memories_curves.json` 静态导出与交互仪表盘
数据三处同源，`_warn_conflict_types` 终端告警也改为读该数组渲染，杜绝两处文案漂移。

**--strict 退出码**：加 `--strict` 后（配合 `--export-signals`，独立运行或
`--sync --eval --export-signals` 均可），退出码**区分两种红灯**——

| 退出码 | 条件 | 语义 |
|---|---|---|
| **1 需排查** | `health.warnings` 非空（冲突类型） | 先检查观测污染 / 事件注入，**不自动调参**——最高优先级（同时存在时覆盖） |
| **2 需校准** | 无冲突但存在 `τ↓`/`τ↑` 行动项 | 非阻塞红灯，可直接 `--apply-suggestions` 执行校准 |
| 0 | 两者皆无（仅无信号 / 需补观测） | 通过 |

收工脚本按退出码分流（`python session_memory.py --sync --eval --export-signals
--strict`；`$?=1` → 排查冲突、`$?=2` → 跑 `--apply-suggestions` 校准、`=0` →
放行）；未开 `--strict` 时恒返回 0（向后兼容，默认不阻塞）。

**行动清单（suggest_adjust）**：导出后**读回 `{基名}.csv` 的 `suggest_adjust`
列**（导出 → 读回 → 行动，顺带验证落盘产物），检测到 `τ↓` / `τ↑` / `需检查`
时打印逐类型行动项（`episodic: τ↓（弱 · 配置偏大 · 忘得比信念快）→ 跑 sleep() 让
学习器实际校准`；`需检查` 引用上方告警证据并**打印该冲突类型的唤醒事件明细**
（`#n [行 k] 记忆 <id> · 相对时间 · dev vs 预期 · 比值 + 方向`，行号直接对应
`{基名}_events.csv`——排障按行号翻文件，不用再全表扫描）），并提示 `sleep()` 末尾自动触发
`learn_tau` + `learn_plasticity` 按当前信号实际校准（CLI `/sleep` 或
`MemoryAgent.sleep()` 后查看 `cfg.tau_by_type`）；无行动项时静默。**同一份行动
清单同时写进导出 JSON**：`health.actions` 数组（`{mtype, suggest, confidence,
reason}`——`suggest ∈ {τ↓, τ↑, 需检查}` 的类型，与终端行动清单 / CSV
`suggest_adjust` 列同一视图、单一事实源 `agent.py::_ADJUST_CN` 语义）——CI /
外部工具按 `health.actions` **非空直接判定需处理**，与 `warnings`（冲突告警）
互补：`actions` 给出全量待办、`warnings` 给出需排查的具体证据。**`需检查` 条目
额外携带 `events` 冲突成因清单**：每起唤醒事件 `{row（对应 events CSV 行号）,
memory_id, dev, expected, ratio, direction, csv}`——direction 与终端明细打印同
闸门（ratio>1.05 → down / <0.95 → up / 其余 flat / 无比值 legacy）——CI 读导出
JSON 直接拿冲突成因，无需再解析终端文本或翻 events CSV。

**一键执行校准（--apply-suggestions）**：加 `--apply-suggestions` 后（配合
`--export-signals`，独立运行或 `--sync --eval --export-signals` 均可），按行动项
**批量执行**：行动项 = `suggest ∈ {τ↓, τ↑}` 的类型（agree 且方向明确——可信可执行）；
冲突（需检查）需先排查、单源（需补观测）观测不足、无信号无可调——均不自动执行。
确认提示（`--yes` 跳过）后跑一次 `sleep()`——末尾自动触发 `learn_tau` +
`learn_plasticity` 按当前观测信号**实际校准**（学习器门控不足的类型自然跳过），
打印**校准前后各类型 τ/drift 对比**（实测 `episodic τ 3.000 → 2.900 天（Δ-0.100）·
drift 2.500 → 2.500（Δ+0.000）`）；校准结果写入 `store.meta`（`learned_tau` /
`learned_plasticity`）并落盘——**真正的 apply**：下次会话重启即加载校准后的参数。

**冲突剔除重算（--exclude-events）**：加 `--exclude-events memory_id:序号,...`
（配合 `--export-signals`；`序号` = 事件在该记忆 `_awakening_events` 过滤列表中的
位置——仪表盘证据行 Shift 多选即可读到此序号），排除指定唤醒事件后**重算唤醒统计
/漂移 / health**——冲突剔除假设检验落地：仪表盘圈出可疑事件 → 终端排除重判。
排除后统计、事件明细、CSV 行号、warnings/actions 全部按剩余事件重算；实测冲突库
排除两条冲突侧事件后 health 从 `conflict` 变为 `agree`（warnings 清零、行动项变为
`τ↓ 需校准`），打印 `⚠ 已排除 N 起事件（…）→ 唤醒统计/漂移/health 按剩余事件重算`。

**自动剔除冲突成因（--exclude-clashes）**：加 `--exclude-clashes`（配合
`--export-signals`）——**按 `dir ≠ clean` 自动圈定冲突成因事件并排除后重判
health 并导出**，与仪表盘「选反向」**同一判定**（`_event_ratio_dir` 1.05/0.95
事件方向 + clean 方向存在且 `dir != clean`，含 `flat ≠ clean`；判定收敛于
agent.py `clash_event_keys` 单一实现，选反向在 CLI 侧的一键执行）：免手写
memory_id，直接 `--export-signals sig --exclude-clashes` → 自动排除冲突侧事件 →
重判后 `conflict → agree`（warnings 清零、行动项变 `τ↓`）。`excluded_clashes`
（与 `excluded`）写入导出 JSON，终端打印 `⚠ 已按 dir ≠ clean 自动排除 N 起冲突成因
事件（…）→ 重判后: 一致 N · 冲突 0 · …`；`--strict` 按重判后 health 给退出码。

**剔除前后对比块（health.exclude_compare）**：`--exclude-events` 或
`--exclude-clashes` 有实际排除时，导出 JSON 的 `health` 自动附
`exclude_compare` 块——每类型 **before/after** 并列一致性、干净段/唤醒方向与建议
（before = 无排除基线，after = 重判后），顶层附 `summary`（一致/冲突计数）与
warnings/actions 条数：CI 读 `exclude_compare.before.by_type.episodic.consistency
= "conflict"` vs `after...= "agree"` 即可判定“剔除这批事件后两路是否一致”，无需
再解析排除逻辑。终端同步打印 `== 剔除前后对比 ==`（`episodic: conflict → agree
（干净段 down → down · 唤醒 up → down · 建议 需检查 → τ↓）` + summary 行）；无
排除时不带此块。

**聚合结论回放（--aggregations）**：加 `--aggregations TYPE:key,key`（可多次）——
把仪表盘 **Shift 多选聚合结论回放为 `health.aggregations`** 数组：每条约目含事件
子集（`events: ['memory_id:序号', ...]`）、选中方向分布、全体/移除后中位比值与方向、

**免手写选择集（--aggregations-file）**：仪表盘聚合面板的**「导出聚合 / 复制」按钮**
一键把全部类型当前 Shift 多选状态导出为 `[{"mtype", "events": ['memory_id:序号',
...]}, ...]`——**导出聚合**下载为 `aggregations.json`，**复制**把同一 payload 写入
剪贴板（clipboard API + `execCommand('copy')` 回退，覆盖无下载权限的嵌入式环境）；
CLI 直接 `--aggregations-file aggregations.json` 读取（也接受 `{"aggregations": [...]}`
包裹；非法条目跳过、缺文件/坏 JSON 打印提示），与手写 `--aggregations` 结果完全
一致，仪表盘圈定 → CLI 回放 → 证据包全程免手抄 memory_id。
干净段方向与判定（`verdict`: `resolved` 冲突消除 / `still_conflict` 仍冲突 /
`insufficient` 观测不足 / `unknown`；`verdict_text` 中文）——与仪表盘 JS 判定同闸门，
同一记忆库上导出与仪表盘结论逐字一致，CI 直接读 JSON 即可回放人工排查结论（实测
选两条冲突侧事件 → `resolved ✔ 移除后两路一致——冲突消除`）。

**方向占比分布（复刻色条）**：每条约目附 `all_dist`（全体各方向计数）+ `all_n`
（有比值事件总数，占比分母）+ `all_dist_pct`（全体占比 `{up: 67, down: 33, flat: 0}`）
与 `selected_dist_pct`（选中集占比）+ `selected_n_ratio`（选中有比值数）+ 中位比值
`all_median_ratio`——与仪表盘色条同口径（以有比值事件为基数），外部工具读 JSON
即可直接复刻「全体 / 选中集」两根三段色条，无需再解析事件明细。

**resolved 自动附带剔除后证据包**：`verdict == "resolved"` 的聚合条目自动附带
`recomputed` 块（`--exclude-events` 同链路）——把该聚合圈出的事件子集作为排除集
重算 `stats` / `periods` / `health`（`excluded` 键带排除清单），人工在仪表盘圈定的
冲突成因**一步变成剔除后健康证据包**：CI 读 `recomputed.health` 直接拿到排除后的
`consistency`（conflict → agree）、`suggest`（需检查 → τ↓）、`warnings` 清零与
`actions` 行动项；终端同步打印「已自动附带剔除后重算证据包（exclude N 起）: …」。
`still_conflict` / `insufficient` 未解决不附带（无可信剔除依据）。

**仪表盘回放历史聚合结论**：`agent.plot_interactive("dash.html", aggregations=[
{"mtype": "episodic", "events": ["记忆id:序号", ...]}, ...])`（与 `--aggregations`
同格式）→ `health.aggregations` 嵌入仪表盘数据，画像面板下方新增**「历史聚合结论」
区**——每条 verdict 徽章（`✔ 冲突消除` 绿 / `✘ 仍冲突` 红 / `— 观测不足` 灰）+ 事件
子集摘要（已选方向分布 · 干净段 · 移除后中位与方向）+ resolved 自动附带剔除后证据包
（consistency / suggest / warnings，与 `--export-signals` 同链路、逐字一致）；**点击
该行在主图高亮对应事件子集**（◇ / 红条 / 青条精确匹配，再点 / Esc 取消），callout
逐条列出子集事件并可点开单事件 dev vs expected 双条。聚合结论判定已下沉到
agent.py（`aggregation_for` / `aggregation_recompute` 单一事实源），导出与仪表盘
共用同一实现。

**建议置信度（suggest_confidence）**：CSV 再追加 `suggest_confidence` 列，
agree 时按两路观测条数给证据强度 **强 / 中 / 弱**——强 = 干净段 ≥ 2×
`tau_min_segments` **且** 唤醒 ≥ 2×`tau_min_awakenings`（两路充分采样）；
中 = 两路都过各自门控；**弱 = agree 但唤醒观测不足（< `tau_min_awakenings`，
如单条观测）——避免单条观测就建议调参，先积累观测再校准**；冲突/无信号 → `—`、
单源 → `弱`。仪表盘画像列与行动清单同步显示置信度。

## memagent 与主流 agent 的定位关系

一句话定位：**memagent 不是又一个 agent，而是 agent 的"记忆机制研究模型"**——
它是"零件"（模拟人脑海马体/皮层的动态记忆），Codex / Claude Code / OpenCode 等
是"整机"（接 LLM、读代码、改代码、跑命令的任务执行工具）。两者不是同类，
但可以互补：memagent 的记忆层可以嵌进任何真实 agent。

主流 coding agent 的记忆实现现状（共同架构 = 静态文件 + 线性历史）：

| 工具 | 长期记忆 | 上下文管理 | 语义检索 |
|---|---|---|---|
| Codex | `AGENTS.md` 层级合并加载 + Memories（从旧对话提炼） | 搜索/读文件优先窗口开头 | 原生无，靠 Mem0/Hindsight 等 |
| Claude Code | `CLAUDE.md` 全量加载 + Auto memory（从纠正中学） | `/compact` 总结旧对话 | 原生无，靠插件 |
| OpenCode | `opencode.json` instructions + SQLite 会话持久化 | 自动压缩（按窗口大小） | 原生无，靠 opencode-mem 等向量插件 |

它们共同的工作方式：**工作记忆 = 上下文窗口（线性历史），长期记忆 = 手工维护的
指令文件（每会话全量加载、不检索），持久层 = 会话存档**。缺的正是 memagent 的
"记忆动态机制"：

| memagent 机制 | 主流 agent 现状 |
|---|---|
| 遗忘曲线（时间衰减） | ❌ 无——记忆要么永存占满窗口，要么被压缩丢弃，没有中间态 |
| 重要性 / 冻结 | ❌ 无——所有记忆等权加载 |
| 分层自动升降级（Hot/Warm/Cold） | 部分——工作/长期之分是手动的 |
| 检索式注入 | 部分——原生不检索，靠第三方向量插件 |
| 睡眠巩固（聚类压缩） | 部分——`/compact` 是线性总结，无低频埋藏 |
| 再巩固（回忆改写记忆） | ❌ 无（Auto memory 只追加笔记，不改原记忆） |
| 按类型分化（技能慢忘/情景快忘） | ❌ 无 |
| 测试效应（检索强化） | ❌ 无 |
| 情景→语义迁移 | ❌ 无 |
| 观测 / 参数自适应 | ❌ 无 |

**结论**：主流 agent 已解决"记忆的存储"（文件 + 会话存档 + 插件检索），memagent
解决"记忆的演化"（衰减 / 巩固 / 再巩固 / 迁移 / 自适应）——遗忘、巩固、再巩固、
类型迁移、自适应调参这五个动态机制，目前没有任何主流 agent 原生内置。

## 把 memagent 接进真实 LLM agent（remember_agent.py）

`remember_agent.py` 把 memagent 作为**长期记忆层**接到 OpenAI 兼容的 LLM
agent 上，演示完整数据流（记忆层 → 编排层 → LLM 层）：

```
用户输入 → 记忆层 retrieve()（遗忘曲线评分 × 相似度 × 同义扩展）
        → 编排层把命中记忆注入 system prompt（带类型与强度）
        → LLM 基于记忆回答（LLMResponder；无相关记忆时直接回答）
        → 对话写入记忆层（持久化 memories_agent.json）
```

```bash
python remember_agent.py                                 # 交互式对话（每轮注入）
python remember_agent.py --remember "我叫小林，用 Python"  # 手动写事实（可多次）
python remember_agent.py --show-memories                 # 查看记忆层
python remember_agent.py --demo                          # 自动演示跨轮注入
python remember_agent.py --reset                         # 清空记忆
```

- 记忆是"活的"：`--remember` 写入的事实（姓名/技术栈/偏好）跨对话长期保留，
  被反复检索的记忆越来越强（测试效应，demo 里可见强度 0.51→0.53→0.55），
  长期不用自动衰减、对话流水自动降权——这些正是主流 agent 上下文里缺失的机制；
  **短查询子串优先重排已下沉到核心 `retrieve()`**（
  `memagent/synonyms.substring_priority_order` 单点实现，开关/阈值见
  `AgentConfig.rerank_short_query` / `rerank_short_len`）——所有下游入口
  （本脚本的对话注入、session_memory 的 `--start --topic`、respond 回复引用）
  统一受益，短词查询不再被哈希碰撞噪声干扰；
- 无 key 时降级为"记忆层演示"：仍检索并打印注入内容，只是不生成 LLM 回复
  （记忆层独立于 LLM，先验证记忆再配 LLM）；配 `OPENAI_API_KEY` 即完整；
- 与 `demo.py` 的区别：demo 演示记忆机制本身，remember_agent 演示**嵌入方式**
  ——把 retrieve 结果拼进 system prompt 这一行，就是真实 agent 接记忆的全部接口。
- **嵌入任意 agent 的三种通用模式**：① prompt 注入（本脚本的做法，通用性最强，
  任何 OpenAI 兼容 agent 都适用）；② 外部记忆工具（把 memagent 包成 MCP/CLI 工具，
  agent 按需调用检索——不占主上下文）；③ 指令文件生成器（已实现：
  `--export-agents-md`，见下）。
- **`--export-agents-md`**：把记忆层"固化知识"（重要 or 被反复检索确认的事实/
  决策/偏好，对话流水排除）导出成 AGENTS.md / CLAUDE.md 风格指令文件：
  ```bash
  python remember_agent.py --export-agents-md            # 默认同时生成 AGENTS.md + CLAUDE.md（内容同步）
  python remember_agent.py --export-agents-md CLAUDE.md  # 指定文件名则只写该文件
  ```
  按类型分组（semantic=事实偏好 / skill=经验方法 / episodic=事件），每条标注
  重要性与检索次数（越靠前越可靠）；文件头注明生成时间与更新方式——
  Codex 读 AGENTS.md、Claude Code 读 CLAUDE.md，两者逐字节一致，任何全量加载
  指令文件的 agent 无需改造就能吃到 memagent 沉淀的固化知识。

## 会话记忆：跨会话沉淀开发决策（session_memory.py）

把 memagent 用作项目的"外置会话记忆"——收工时沉淀决策，开工时自动注入：

```bash
python session_memory.py --record [--since "2 hours ago"] [--note "补充决策"]  # 收工：从 git log 提炼提交 + 手动补充
python session_memory.py --start [--topic 关键词] [--k 5]                       # 开工：终端打印注入块（默认模式）
python session_memory.py --write-context [文件] [--topic 关键词]               # 开工：生成独立注入 prompt 文件（默认 session_context.md）
python session_memory.py --inject-agents-md [文件] [--topic 关键词]            # 开工：维护 AGENTS.md 顶部动态记忆区块（默认 AGENTS.md）
python session_memory.py --export-agents-md [文件]                             # 全量导出决策记忆为 AGENTS.md 风格文档（不带文件名 → 同时生成 AGENTS.md + CLAUDE.md，内容同步）
python session_memory.py --eval-agents-md [文件]                               # 实测全量加载效果（内容覆盖 + 可选 LLM 问答）
python session_memory.py --sync [--note "补充"] [--eval] [--export-signals 基名]  # 一键闭环：git log 提炼 → 沉淀 → 刷新 AGENTS.md + CLAUDE.md（--eval 追加收工验证：加载评估 + 唤醒链路连续性 + 唤醒信号统计 + τ 学习器健康检查 + 学习器响应演示；--export-signals 把第 ⑦ 节信号统计导出为 JSON + CSV + 事件明细 CSV，也可单独运行）
python session_memory.py --show [--topic]     # 查看记忆层
python session_memory.py --reset              # 清空
```

**开工流程集成**：`--inject-agents-md` 把注入块维护到 AGENTS.md 顶部——文件不存在则创建，
已有 `<!-- memagent-injection -->` 区块则整体替换（marker 单一，不重复堆积），
否则顶部插入（原内容保留）。Codex / Claude Code 等全量加载指令文件的 agent
开工时自动吃到最新注入的决策；`--write-context` 生成独立 prompt 文件供手动粘贴。
`--sync` 把整条链闭合成一条命令（收工时跑一次即可）：
```bash
python session_memory.py --sync --note "本次补充决策" --eval
# ① git log 提炼提交 → ② 沉淀决策（去重强化）→ ③ 刷新 AGENTS.md + CLAUDE.md → ④ 收工验证（加载评估 + 唤醒链路连续性 + 唤醒信号统计 + τ 学习器健康检查 + 学习器响应演示）→ ⑤ --export-signals 导出信号统计 JSON + CSV + 事件明细 CSV
```

**一键接入脚本**：`setup_agents.ps1`（PowerShell，中文 Windows 直接可用）把上面
整套流程收成一条命令——① 全量导出 AGENTS.md + CLAUDE.md（双格式同步，
Codex / Claude Code / Hermes 开工自动加载）；② 安装 git post-commit 钩子，之后
**每次提交自动 `--sync`**（提炼新提交 → 沉淀决策 → 刷新双格式导出，去重设计可重复触发；
非 git 目录跳过钩子并提示）；③ 打印开工/收工说明。
```powershell
powershell -ExecutionPolicy Bypass -File setup_agents.ps1
```
重复运行安全：重新导出、钩子已存在时先备份 `post-commit.memagent.bak` 再覆盖。

> 完整的接入教程（三种方式：指令文件 / 动态记忆循环 / memagent 自身接 LLM，含
> Codex / Claude Code / Hermes 各自的加载方式与 10 条常见问题）详见
> **`docs/agents-integration.md`**。

本仓库根目录已有 `AGENTS.md` + `CLAUDE.md`（**全量 21 条决策，逐字节一致**，
按类型分组、带重要性与检索标注）与 `session_context.md` 示例。**加载效果实测**（`--eval-agents-md`）：
8 个主题问题（中位数抗离群/双阈值滞回/触底时间/技能校验/同义扩展/时钟注入/
贴合度公式/再巩固冻结）的答案关键词在 AGENTS.md 中 8/8 全部可及——全量加载
agent 开工读取后信息完整；配置 `OPENAI_API_KEY` 后评估会追加 LLM 问答环节，
验证真实提取质量（链路已用 mock 验证）。

- **收工沉淀**：提交信息（`开发决策：xxx`）与 `--note` 手动补充全部写入记忆层；
  非 git 仓库自动降级（只记 `--note`）；
- **开工注入**：输出可直接粘贴给 agent 的上下文块（`==== memagent 决策记忆注入 ====`），
  `--topic` 按主题检索（rel 排序），无 topic 按当前强度取 top-k——重要决策
  长期保留、被遗忘的自动沉底；  **短主题词自动子串优先重排**：已下沉到核心 `retrieve()`
  （`synonyms.substring_priority_order` 单点实现，本入口与 remember_agent 共用；
  阈值可配 `AgentConfig.rerank_short_len`，默认 3 字以下），
  内容含主题词的决策排前面（组内仍按强度），消除哈希嵌入泛化命中把无关记忆
  顶到前面的干扰（实测「触底」从误伤"可注入时钟"修正为精准命中"触底时间"），
  并提示建议加长；
- **动态记忆语义**（与静态 CLAUDE.md/AGENTS.md 的区别）：同一决策重复沉淀会被
  去重合并并**测试效应强化**（检索次数可见增长），决策按类型自动分类（skill/
semantic），长期不检索的自动衰减——"被反复确认的决策越来越强"。
- **本仓库已回填**：历次实现的关键决策（学习器用中位数抗离群、双阈值滞回防振荡、
  触底验证从观测推导、查询同义扩展 rel 只升不降、可注入时钟、循环导入坑等 20 条）
  已沉淀到 `memories_session.json`——新会话跑 `python session_memory.py --start
  [--topic 主题]` 即可恢复上下文；实测 5 个主题检索 4 个 top1 精准命中
  （短词「触底」受哈希碰撞干扰，印证了"短查询泛化命中会误伤"那条决策）。

## MCP 服务器：把记忆层接给任意 Agent（mcp_server.py）

把 memagent 作为 **MCP 工具服务**暴露——Hermes / Claude / Codex 等支持 MCP 的
agent 无需改代码，原生调用记忆层。核心零依赖不变，MCP 路径按需安装：

```bash
pip install "memagent-local[mcp]"        # 或 pip install mcp>=1.2
memagent-mcp --persist memories.json     # stdio 服务器（也可 python -m memagent.mcp_server）
```

**语义嵌入（可选，突破词汇级上限）**——默认哈希嵌入零依赖、按 n-gram 重叠打分；
跨措辞等价（「技术栈」↔「Python 程序员」）需要真语义嵌入，二选一：

```bash
# ① OpenAI 兼容远程端点（任何 /embeddings 服务，纯 stdlib 调用）
memagent-mcp --persist memories.json --embed-base-url https://api.example.com/v1 \
             --embed-model text-embedding-3-small --embed-api-key sk-xxx
# ② 本地 sentence-transformers（pip install "memagent-local[embed-local]"）
memagent-mcp --persist memories.json --embed-local paraphrase-multilingual-MiniLM-L12-v2
```

换后端后存量记忆在加载时按维度失配自动重建向量；词汇重叠保底、子串重排等
文本级逻辑不依赖后端。SDK 内调用 `embedding.set_embedder(RemoteEmbedder(...))`
同样生效（`embed_text` 是唯一接入点）。

注册（Hermes 示例）：

```bash
hermes mcp add memagent --command python --args -m memagent.mcp_server --args --persist /path/to/memories.json
```

九个工具（与 CLI / 交互模式对齐，不只是检索一个薄面）：

| 工具 | 作用 |
| --- | --- |
| `memagent_remember` | 写入记忆（importance ≥ 0.8 冻结为核心记忆） |
| `memagent_retrieve` | 检索（带置信标注：`matched=false` 或 `notice` = 大概率查不到，别硬编） |
| `memagent_forget` | 彻底删除一条记忆（CLI `/forget`） |
| `memagent_recall` | 唤醒 Cold 摘要 → 完整记忆 + 深藏细节（CLI `/recall`） |
| `memagent_find` | 关键词定位记忆，拿 id 供 forget / recall 用 |
| `memagent_start` | 开工注入：按主题取相关决策组成上下文块（`--start` 的 MCP 版） |
| `memagent_export` | 导出 AGENTS.md（dual=true 同步 CLAUDE.md；`--export-agents-md` 的 MCP 版） |
| `memagent_sleep` | 睡眠巩固（回放 / 分级 / 压缩） |
| `memagent_stats` | Hot/Warm/Cold 各层统计 |

注入 / 导出与 `session_memory.py` **共用同一实现**（`memagent/instructions.py`），
CLI 与 MCP 两条入口不会漂移。检索结果带 `relevance` 与 `matched` 标注：
符号哈希嵌入（见下）让无关查询的相似度回到 0 附近，服务端据此明确说
"无高置信命中"，而不是让调用方在碰撞噪声里硬猜。

## v0.3.2 发布验证

当前版本已包含可选语义嵌入、MCP stdio 服务与零依赖 REST 服务。发布前建议执行：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m memagent --check --persist agent_memory.json
python -m memagent.release build --output releases
python -m memagent.release verify --wheel releases/v0.3.2/memagent_local-0.3.2-py3-none-any.whl
python -m memagent.release install `
  --wheel releases/v0.3.2/memagent_local-0.3.2-py3-none-any.whl `
  --runtime .runtime
python -m memagent.release run --runtime .runtime -- --version
```

发布包应只包含 `memagent` 包和标准元数据，不应包含 `.env`、记忆 JSON、
`works/`、日志、实验产物或本地虚拟环境。

## 已知局限

- **默认检索仍是词汇级的**：符号哈希 n-gram 嵌入（v0.3.1）消除了加性碰撞
  偏置——无关文本相似度回到 0 附近，「火锅撞首都」类误命中不再发生，短查询
  的「查不到」能被诚实识别；但它本质是**字符 n-gram 重叠**，跨措辞的语义
  等价（「技术栈」↔「Python 程序员」）默认查不到。**已可插拔**：`embedding.set_embedder()`
  / MCP `--embed-*` 接入真语义嵌入（OpenAI 兼容远程或本地 sentence-transformers），
  向量按维度失配自动迁移（见 MCP 章节）；
- 提取式摘要在短文本上压缩率有限（摘要 ≈ 原文），此时 Cold 的意义主要在"埋藏"而非"省空间"；
- 类型识别默认 LLM 分类、关键词回退，长句含多类信号时仍可能误判（可手动 `mtype` 覆盖）；
- 429 切换是**模型级**（同一端点多个模型各有限流）；跨端点（如 DeepSeek 429 → 切本地 Ollama）需要多个 base_url 的池，当前未支持；
- 情绪目前来自关键词启发式三轴模型，不等同于真实情感理解；多感官输入仍未支持；
- v0.2 是单机、单租户产品基线；团队 SaaS 场景仍需外置数据库、认证、租户隔离和审计服务。

## 项目结构

**分层地图**——只把 memagent 当「记忆 SDK / 记忆 MCP」用时，只需要第一层：

| 层 | 模块 | 说明 |
| --- | --- | --- |
| **核心记忆 SDK**（零依赖） | `embedding` `decay` `memory` `synonyms` `compression` `agent`（检索/巩固部分）`instructions` `mcp_server` `server` `llm` `responder` `io_utils` | 记忆层本体 + 服务入口。MCP / SDK / HTTP 三条接入路径只走这一层 |
| 认知扩展（零依赖） | `emotion` `interest` `graph` `growth` `cognition` `curiosity` `analogy` `social` `human` `checkers` `profiles` `visualize` `interactive` | 检索评分的增益项（情绪一致性/情境加成/兴趣等），随核心一起加载 |
| 领域扩展（写作 / FoxTable 编码） | `architecture` `critique` `work_admin` `continuity` `reader_postproc` `literary` | 仅被根目录 `autonomous_writer.py` / `autonomous_coder.py` 等脚本调用；`MemoryAgent` 内的写作方法（`write_chapter` 等）同样只服务写作入口——记忆 MCP 的调用路径不会触及 |

```
memagent/
  embedding.py   哈希 n-gram 嵌入 + 余弦相似度（v0.3.1：1024 维 + 符号哈希；v0.3.2 可插拔后端）
  embedders.py   语义嵌入后端（OpenAI 兼容远程 / 本地 sentence-transformers）
  decay.py       Ebbinghaus 遗忘曲线评分
  memory.py      Memory 模型、三层存储、类型/重要性启发式、JSON 持久化（旧向量自动迁移）
  synonyms.py    查询同义扩展（人称互换 + 同义词变体）
  llm.py         LLM 类型分类器 + ModelPool（429 多模型自动切换/全限流等待重试）
  responder.py   LLM 回复生成器（检索结果注入上下文；人设 persona + 演化档案注入；无 key 回退模板）
  websearch.py   自主联网搜索（Bing 首选 / DuckDuckGo 备用，纯 stdlib，失败返回空列表）
  compression.py 提取式摘要 + 相似记忆聚类合并
  agent.py       MemoryAgent：检索、回复、睡眠巩固、CLI
  checkers.py    按类型分流的内容钩子（技能类一致性校验）
  instructions.py 开工注入 / AGENTS.md 导出（CLI 与 MCP 共用的单一实现）
  mcp_server.py  MCP 服务器（stdio，九个工具；需 pip install memagent-local[mcp]）
  server.py      本地 HTTP 服务（纯 stdlib）
  visualize.py   强度曲线可视化（纯 Python 生成 SVG）+ CSV/JSON 导出
  interactive.py 多视图仪表盘（单文件 HTML：曲线/气泡图/分布/Top列表联动）
  profiles.py    记忆类型画像（τ / 再巩固因子 / 压缩阈值配置表）
  emotion.py     三轴情绪模型与编码/遗忘/检索调制
  interest.py    可持久化兴趣向量与主题检测
  graph.py       可持久化知识图谱
  growth.py      预测验证、模式提取、概念形成和自主提问
  cognition.py   技能、长期目标和认知边界
  curiosity.py   好奇驱动探索闭环
  analogy.py     跨领域类比迁移
  social.py      多 agent 显式知识/技能/记忆交换
  io_utils.py    文件锁、原子写入和备份
  compat.py      可插拔 responder 接口兼容层
  architecture.py 小说大纲/人物/世界文档与作品迁移
  critique.py    写作自评、对标和改进规则沉淀
  cli.py         Windows GBK 终端 UTF-8 输出适配
demo.py          脚本演示（秒级时间常数）
novelist_demo.py 小说家人设自主演化 + 429 多模型自动切换演示（离线，假 HTTP 客户端）
experiment.py    类型行为对照实验（同一批记忆 × 多组参数，输出对比报告）
llm_classify_demo.py  LLM 分类+回复生成链路最小示例（真实 HTTP 验证，无 key 时自动用 mock）
recall_curve_check.py   唤醒链路曲线连续性验证（导出唤醒前后 SVG，断言 Cold↔Warm 无缝衔接；预测线 vs recorded 叠加图直观看唤醒点偏差：红条=实测跳升、青条=类型预期偏差，基线连线=信号幅度，附信号方向——实测>类型预期 → τ↓ · 可塑性↑；--real 追加真实持久化场景：从 memories_session.json 加载真实决策记忆，老化→sleep 压缩→唤醒→临时持久化往返，验证真实数据连续性且只读原文件；--awakened [路径] 多次唤醒记忆检查：从真实库挑一条 awakenings > 1 的记忆，逐次标注全部唤醒事件（dev vs 类型预期双条 + 信号方向徽章，比值趋 1 = learn_tau 已校准）——真实库无可选对象时合成一条；路径若是 --export-signals 导出的 JSON（顶层 events）则直接从导出文件挑多次唤醒记忆做逐次标注（导出 → 验证闭环，事件时间线 SVG））
remember_agent.py     memagent 记忆层 + 真实 LLM agent 编排（检索→注入→回复→写入）
session_memory.py     会话记忆：收工沉淀开发决策（git log + --note），开工自动注入
mock_openai_server.py 本地 mock OpenAI 服务（离线验证 LLM 分类/回复生成链路）
tests/           pytest 核心行为测试
docs/retrieval_enhancement.md  检索增强说明（expand_query × substring_priority_order 分工 + 合并簇摘要检索示例）
docs/agents-integration.md    Codex / Hermes 接入指南（真实命令 + 推荐工作流 + 常见问题）
```

## 可视化：记忆强度曲线

Agent 会在**每次写入、检索命中、升降级**时给记忆记录一条状态快照
`(时间, 观测强度, 最后访问, 检索次数, 重要性)`，并且**每轮对话结束和每次
睡眠巩固后自动观测一轮**（`_observe()`，给所有记忆采样，不只检索命中的）——
这就是持续观测：每条记忆的真实遗忘轨迹都被跟踪。`/plot` 或
`agent.plot_curves()` 据此导出四种文件：

- **`memories_curves.svg`** —— 主图：一条曲线对应一条记忆：
  - **实线** = 按遗忘曲线公式外推的预测强度；
  - **灰色虚线** = 实际观测轨迹（穿过所有采样点），偏离实线处即发生
    检索/再巩固干扰，或 τ 配置失准；
  - **圆点** = 观测采样（悬停可看记忆 ID 与内容）；
  - 颜色按层级区分（红=Hot / 蓝=Warm / 灰=Cold），黄虚线标出强度下限 0.2，
    黑色虚线是"现在"时刻；标题显示各类型 τ，第二行副标题显示贴合度摘要；
  - **唤醒事件标注（全部历史）**：每条记忆的每次唤醒画一个 ◇ 菱形（唤醒后
    实测强度）+ 红条（实测跳升 dev）+ 青条（类型预期 expected）+ **信号徽章**
    （比值 dev/expected，颜色 = 校准状态：红 >1 = 唤醒比类型预期剧烈（τ 应
    下调）、灰 ≈1 = 已校准、青 <1 = 偏温和）——learn_tau 校准过程中徽章红色
    渐渐转灰/青、比值趋 1，直观展示信号随学习轮次衰减收敛；横轴窗口自动左扩
    覆盖记忆创建之后的全部唤醒历史（不只 now 起），负时间轴上的菱形即历史
    唤醒事件（多次 Cold↔Warm 往返的曲线连续性由此可追溯）；
  - **类型参考曲线**（绿=技能/紫=语义/橙=情景的虚线）：配置 τ 的"典型遗忘"
    预期斜率（一条从横轴起点创建、重要0.1、零检索记忆的衰减）——与全部记忆
    曲线同图对照：曲线落在参考线上方 = 该记忆比"典型"忘得慢（被检索/重要度高），
    下方 = 忘得快；交互版主图同样叠加（跟随缩放/平移，不参与点击高亮）。
- **`memories_curves_by_type.svg`** —— 按类型分面板：技能/语义/情景各一张子图，
  **共享横轴**可直接对比遗忘斜率；每张子图标注配置 τ / 实测 τ / 贴合度，并画
  一条灰色虚线**参考曲线**（该类型典型遗忘，重要0.1、零检索），面板间坡度差异
  （技能平缓 / 情景陡降）一目了然；无此类记忆时显示占位。
- **`memories_curves.csv`** —— 长格式数据（row_type 区分采样/唤醒；memory_id,
  tier, mtype, 时间, 强度...）。**每条记忆的唤醒事件随主表同行导出**：
  `row_type="awakening"` 的行 strength 列 = 实测偏差 dev、content 列 =
  `expected=… ratio=…`、mtype 取唤醒时刻类型——过滤 row_type 即可在同一张表
  里同时分析曲线与唤醒明细。
- **`memories_curves_awakenings.csv`** —— **唤醒事件明细**：每行一次唤醒
  （memory_id, 唤醒时刻类型, 相对时间, 实测偏差 dev, 类型预期偏差 expected,
  **比值 dev/expected**），与曲线 CSV 按 memory_id 连接。外部工具按 mtype
  分组分析比值：> 1 = 该类型被唤醒得比自身模型预期剧烈（τ 配置偏大 / 可塑性
  配置偏小，学习器开启时会自动校准）；比值仅在 dev、expected 都 > 0 时给出
  （与学习器门控一致），旧格式唤醒日志（无类型预期）不导出。
- **`memories_curves.json`** —— 结构化数据（预测序列 + 实际采样 + 贴合度报告）；
  每条记忆额外带 `awakening_events`（{时间, 实测, 预期, 比值, 唤醒时刻类型}）
  与原始 `awakenings` 四元组，供直接分析类型可塑性；顶层带 `profiles` +
  `signal_drift` + `health` 三表（与仪表盘一致），静态 JSON 三表同屏可查。

预测窗默认：演示用小 τ 时取 6×最大τ，生产配置取 14 天。

## 多视图仪表盘（浏览器版）

`/ploti` 或 `agent.plot_interactive()` 导出**单文件 HTML 仪表盘**（内联 SVG +
原生 JS，零依赖、离线可开），四个**联动**视图：

- **强度曲线主图**：线宽 ∝ 重要性、空心环标 = 检索事件；滚轮缩放、拖动平移；
- **记忆地图**（气泡图）：x=检索次数、y=重要性、气泡大小=**触底倒计时**（遗忘斜率，
  默认：不触底=最大、已触底=最小，可一键切回"强度"模式）；点击气泡 → 详情面板
  **实时触底倒计时**（每秒滴答："41.2秒后触底"），悬停提示带倒计时；
- **层级×类型分布条**：Hot/Warm/Cold 按 skill/semantic/episodic 分色，点层级名切换；
- **最强记忆 Top5** + 顶部统计条（记忆数/分层/平均重要性/检索事件总数）；
- **类型对比视图**：技能/语义/情景三列子图**共享横轴**（同宽 = 同时长，坡度直接
  对比），每列画该类型记忆曲线 + 灰色虚线**典型遗忘参考曲线**（按该类型 τ），
  面板标题标注条数/τ/贴合度——把静态的 `memories_curves_by_type.svg` 也接进
  了仪表盘。点击子图曲线 → 高亮并显示**遗忘斜率数值**（详情面板：每τ强度下降 +
  触底时间 vs 参考 → "持久 2.8 倍"/"快 33% 触底"/"≈ 典型"/"持久（不触底）"），
  子图曲线悬停提示同样带斜率。
- **自定义时间窗**：类型对比视图完全由 JS 按窗口渲染——数据只嵌入记忆状态，
  JS 用与 `decay.py` 同款的强度公式**按窗口自适应生成曲线点**（窄窗密、宽窗疏，
  任意窗口都平滑）。控件："过去 N 天 / 未来 M 天"数字输入 + 预设按钮
  （过去7/30天、未来7/30天）+ 重置；窗口切换时观测点/轨迹自动裁剪（未来窗
  无观测点、过去窗无预测段），"现在"竖线在窗口含当前时刻时显示，点击联动与
  层级切换在窗口切换后依然生效。

**遗忘斜率**（`visualize.py::forgetting_slope`）：每 τ 的预测强度下降量 + 按模型
预测的**触底时间**（衰减到 0.2 下限还需多久）。对比用触底时间而非斜率比——
斜率归一化到每 τ 后纯 recency 衰减跨类型相同，且触底钳制会扭曲斜率比（同
learn_tau 的"触底段不参与反推"教训）；触底时间区分度大且直观：检索/重要性抬高
基线的记忆"持久（不触底）"，已触底记忆 slope≈0。

**触底验证**（`visualize.py::floor_verification`）：把遗忘斜率从纯预测升级为
**预测 vs 实际**。观测采样（`_record_sample`）按"真实"τ（`true_tau_by_type`）
记录强度并钳到下限 0.2，所以历史里第一个强度 ≤ 0.2 的采样点 = **实测触底时刻**；
对比 `forgetting_slope` 从同一状态（该衰减段起始的最后访问）按模型 τ 预测的
触底时长：`ratio = 实测/预测`，<1 衰减快于预期、>1 慢于预期、≈1 贴合。仪表盘
详情面板新增"触底验证"行（未触底时显示"尚未实测触底（预测 X）"，触底后显示
"实测 4秒，比预测快 85%（衰减快于预期）"这类结论）；类型对比子图与静态 by_type
SVG 的曲线悬停提示同样带验证结论。demo 第 13 段用模拟时钟演示：模型信念 τ=30s、
真实环境 4s → 预测 27 秒触底、实测 4 秒（比预测快 85%）。

点击任意面板（主图/气泡/列表/类型对比子图）中的曲线/观测点 → **全局高亮**
（其余变暗 + 详情面板）；Hot/Warm/Cold 按钮的显示/隐藏对五个视图同时生效
（含类型对比子图的曲线与观测点）；Esc/空白取消，重置视图还原。
静态 SVG（`/plot`）同样叠加了线宽=重要性、环标=检索事件、类型参考曲线。

主图标注**全部唤醒事件**（窗口左扩覆盖历史，不只 now 起）：每条一个 ◇
菱形（唤醒后实测强度）+ 红条（实测跳升 dev）+ 青条（类型预期 expected）
——两条都结束于实测点高度，红条长于青条即"唤醒比类型预期剧烈"（比值 > 1）；
每条带**信号徽章**（比值 dev/expected，颜色随校准状态变化：红 >1 = τ 应
下调、灰 ≈1 = 已校准、青 <1 = 偏温和）——learn_tau 收敛时徽章颜色由红转
灰/青、比值趋 1，直观展示信号随学习轮次衰减收敛。悬停显示精确双值与比值，
点击参与全局高亮；统计条追加「唤醒事件」计数。

**唤醒点交互展开**（点击 ◇ 菱形或任一双条）：
- 主图在该点旁**展开更长的 dev/expected 双条**（端点带数值标签）+ **信号方向
  箭头**（红 ↓ = τ 应下调、青 ↑ = 应上调、灰 ✓ = 已校准）；
- 右上角**悬浮 callout**：事件时间、记忆内容、dev vs expected 比例条、比值
  与方向解释（"实测跳升深于类型预期 → 埋得比信念深 → τ 应下调（或可塑性
  配置偏小）"等）；
- **与类型面板联动**：展开时该记忆在类型对比视图的曲线被高亮，曲线上的
  唤醒事件显示为彩色小菱形（颜色同方向语义）——点击类型面板里的菱形会
  在主图展开**同一事件**（两个视图双向可达）；
- Esc / 空白 / 点击其他唤醒点收起。

`demo.py` 结尾会自动导出 `memories_dashboard.html`。

## 贴合度验证：实测τ vs 配置τ

把观测数据与预测对比（`visualize.py::fit_report`），回答"遗忘参数调得准不准"：

- 相邻两次观测构成一段：段内发生过**检索**（last_access 前进）或**再巩固**
  （重要性变化）记为干扰段，否则为"干净衰减段"；
- 干净衰减段里，用段首状态回放预测段末强度（模型 τ），同时从实际衰减
  反推该段的**实测 τ**；按类型聚合（按时长加权），与配置 τ 对比得贴合度
  `fit = 1 − |实测τ − 配置τ| / 配置τ`；
- 默认观测与预测用同一 τ，贴合度恒 ≈ 100%（自洽性检查）；要真实验证
  校准，设置 `true_tau_by_type` 模拟"隐藏的真实遗忘速度"——观测按真 τ
  采样、预测仍按配置 τ，τ 失配会立刻暴露（demo 第 7 段演示：配置 30 秒、
  真实 12 秒，报告反推出 12 秒并给出 40% 贴合度）。
- CLI 里 `/observe` 随时观测一轮并打印报告，`/plot` 导出图表并打印报告；
  JSON 导出含逐记忆残差与干扰统计。

```bash
python -m memagent   # 对话后输入 /plot
python demo.py       # 演示结尾自动导出 memories_curves.svg
```
