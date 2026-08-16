# memagent × Codex / Hermes 接入指南

memagent 是模拟人脑记忆机制的**记忆层**，本指南回答一个问题：怎么让真实 agent
（Codex / Claude Code / Hermes 等）用上它的跨会话记忆。三种接入方式按改造量递增：

| 方式 | 一句话 | 改造量 | 适合 |
|---|---|---|---|
| ① 指令文件 | 导出 AGENTS.md / CLAUDE.md，agent 开工自动加载 | 零改造 | 一次性接入、静态固化知识 |
| ② 动态记忆循环 | 收工 `--sync` 沉淀、开工 `--start` 注入、钩子全自动 | 一条命令 | 日常开发、跨会话决策记忆 |
| ③ memagent 自身接 LLM | 配 `OPENAI_*` 环境变量，类型分类 + 回复生成走 LLM | 配置环境变量 | 需要智能分类/回答时 |

三种方式可以叠加：**① 保证开工即拥有记忆，② 保证记忆持续更新，③ 让 memagent
自己的判断也智能化**。全部命令均为 Windows PowerShell 与 Linux/macOS bash 通用
（PowerShell 下用 `$env:X = "…"`，bash 下用 `export X=…`）。

---

## 方式一：指令文件（零改造，推荐先做这个）

### 原理

Codex 启动时**自动读取** `AGENTS.md`（OpenAI 的 agent 指令约定），Claude Code
读 `CLAUDE.md`。memagent 把记忆层固化的知识按这两种约定导出——agent 无需任何
改造，开工就把记忆库当"它的长期指令"加载。

### 真实命令

```powershell
# ① 开发决策记忆库（项目级：从 git log 沉淀的决策，按 semantic/skill/episodic 分组）
python session_memory.py --export-agents-md
#    → 同时生成 AGENTS.md + CLAUDE.md（同一份内容，逐字节一致，内容同步）

# ② 用户事实/偏好记忆库（个人级：记得你是谁、项目细节）
python remember_agent.py --export-agents-md
#    → 同样双格式生成；指定文件名则只写该文件，例如：
python remember_agent.py --export-agents-md project.md
```

不带文件名时默认双写（本仓库根目录现有 `AGENTS.md` + `CLAUDE.md`，均为 21 条决策、
逐字节一致——Codex 读前者、Claude Code 读后者，不存在"文件过期"问题）。

### 各 agent 如何吃到文件

| agent | 加载方式 | 说明 |
|---|---|---|
| OpenAI Codex | 自动读项目根 `AGENTS.md` | 零操作；也可 `codex --instruction-file AGENTS.md` 显式指定 |
| Claude Code | 自动读项目根 `CLAUDE.md` | 零操作 |
| Hermes（agent 工具） | 视实现：指令文件 / `--system` 粘贴 / 手动贴入对话 | 用 `--write-context` 生成独立 prompt 文件最通用 |
| 任意 agent | 无约定时手动粘贴 | `python session_memory.py --start` 打印注入块，直接粘贴 |

### 一键接入脚本（PowerShell）

```powershell
powershell -ExecutionPolicy Bypass -File setup_agents.ps1
```

一步完成：① 双格式导出 AGENTS.md + CLAUDE.md → ② 安装 git post-commit 钩子
（之后每次提交自动 `--sync`）→ ③ 打印开工/收工说明。重复运行安全：钩子已存在
时先备份 `post-commit.memagent.bak` 再覆盖。

---

## 方式二：动态记忆循环（日常使用）

静态导出是快照；日常用这套循环让记忆持续更新：

### 收工：沉淀一次

```powershell
# 一键闭环：git log 提炼新提交 → 沉淀决策（去重强化）→ 刷新 AGENTS.md + CLAUDE.md
python session_memory.py --sync --note "本次关键决策" --eval
#   --note 可多次指定；--eval 追加加载评估（8 个主题问题检查导出质量）
```

非 git 仓库时 `--sync` 自动降级：提炼 0 条提交，仅沉淀 `--note` 内容并照常导出
（有测试锚定该行为）。

### 开工：注入一次

三种形态任选：

```powershell
python session_memory.py --start --topic "再巩固"   # ① 终端打印注入块，直接粘给 agent
python session_memory.py --write-context ctx.md     # ② 生成独立 prompt 文件（agent 读取/加载）
python session_memory.py --inject-agents-md         # ③ 维护 AGENTS.md 顶部动态区块（Codex 自动加载）
```

`--inject-agents-md` 与 `--export-agents-md` 的分工：前者是**顶部 top-k 动态区块**
（每次开工注入最新热点，marker 单一不堆积）；后者是**全量决策库**（完整导出）。
日常用前者热更新、隔段时间用后者重新全量导出。

### 全自动：git 钩子

`setup_agents.ps1` 安装的 post-commit 钩子让收工沉淀变成零操作——每次 `git commit`
自动运行 `--sync`（去重设计使重复触发安全，无新提交时静默跳过）：

```bash
# 钩子实际执行的内容（了解即可）
git commit -m "feat: ..."      # 提交后自动：提炼 → 沉淀 → 刷新双格式导出
```

---

## 方式三：memagent 自身接 LLM（Hermes / DeepSeek / Ollama）

如果 "Hermes" 指 Nous Research 的 Hermes 模型（或任意 OpenAI 兼容端点），
memagent 的两个智能环节都可以走它——只读三个环境变量，零代码改造：

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `OPENAI_BASE_URL` | 端点地址（OpenAI / DeepSeek / Moonshot / Ollama / vLLM 均可） | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | 密钥（Ollama 本地可填任意占位） | 无 → 自动离线降级 |
| `OPENAI_MODEL` | 模型名 | `gpt-4o-mini` |

```powershell
# 本地 Ollama 跑 Hermes 3（示例）
$env:OPENAI_BASE_URL = "http://127.0.0.1:11434/v1"
$env:OPENAI_API_KEY  = "ollama"
$env:OPENAI_MODEL    = "hermes3:latest"
python -m memagent     # 交互对话：类型识别 + 回复生成都走 Hermes

# DeepSeek（云端）
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
$env:OPENAI_API_KEY  = "sk-..."
$env:OPENAI_MODEL    = "deepseek-chat"
```

配置后生效的两个环节：

1. **记忆类型分类**（`LLMClassifier`）：写入记忆时由 LLM 判断 skill/semantic/episodic
   并给出置信度（返回严格 JSON，容忍代码围栏与措辞别名）；未配 key、网络错误、
   解析失败时**自动回退关键词规则**——分类永不阻塞写入；
2. **回复生成**（`LLMResponder`）：`respond()` 把检索结果作为上下文注入 prompt——
   有相关记忆 → 基于记忆回答；无相关记忆 → 凭常识直接回答（不再是"我不知道"）；
   出错/无 key → 回退模板回复。回复生成是可替换钩子，不影响检索链路。

实测链路验证：`tests/` 里有 FakePost mock 全链路用例（分类 JSON 解析、注入上下文、
降级路径），配置真实 key 后即可替换为真实端点。

---

## 推荐工作流（三合一）

```
项目初始化    → setup_agents.ps1（双格式导出 + 钩子 + 说明）
每次会话结束  → python session_memory.py --sync --note "..." --eval
每次开工前    → python session_memory.py --inject-agents-md
agent 启动    → Codex/Hermes 自动加载 AGENTS.md / CLAUDE.md → 拥有跨会话决策记忆
可选增强      → 配置 OPENAI_* → memagent 自身分类/回复也智能化
```

---

## 常见问题

**Q1：AGENTS.md 和 CLAUDE.md 为什么有两份？内容会不一致吗？**
双格式是给不同 agent 的加载约定——Codex 读 AGENTS.md、Claude Code 读 CLAUDE.md。
两者由**同一份内容**写出（一次生成、双写），逐字节一致，不存在漂移；任何一次
`--export-agents-md` / `--sync` 都会同时刷新两者。

**Q2：`--export-agents-md` 和 `--inject-agents-md` 到底什么区别？**
- `--export-agents-md`：**全量导出**整个决策库（默认 50 条上限），适合一次性完整加载；
- `--inject-agents-md`：只维护 **AGENTS.md 顶部的动态区块**（top-k 热点注入），
  用 `<!-- memagent-injection -->` marker 整体替换、不重复堆积，适合每次开工热更新。
推荐组合：开工用 `--inject-agents-md`，隔段时间跑一次 `--export-agents-md` 全量重导。

**Q3：memagent 的记忆文件会不会和 Codex/Hermes 的状态互相干扰？**
不会。`memories_session.json`（session_memory 用）和 `memories_agent.json`
（remember_agent 用）是 memagent 自己的持久化，与 agent 的会话状态完全独立；
删掉文件即重置记忆层，不影响 agent 本身。

**Q4：没配 OPENAI_API_KEY 会怎样？**
一切照常。类型分类回退关键词规则，回复生成回退模板；记忆的写入/检索/遗忘/
巩固等核心机制完全不依赖 LLM（这是 memagent 的设计前提——记忆层零依赖）。

**Q5：`--sync` 在非 git 仓库会报错吗？**
不会。`git log` 拉取失败时降级为 0 条提交，只沉淀 `--note` 内容并照常导出，
退出码仍为 0（有测试锚定）。

**Q6：怎么验证 agent 真的"吃到"了记忆？**
```powershell
python session_memory.py --eval-agents-md AGENTS.md
```
离线模式对 8 个主题问题检查 AGENTS.md 是否含答案关键词（本仓库实测 8/8 全可及）；
配置 `OPENAI_API_KEY` 后追加 LLM 问答环节，验证真实提取质量。

**Q7：想让每个项目的记忆互相隔离？**
`--persist` 指定各自文件即可：
```powershell
python session_memory.py --sync --persist ../proj-a/memories_session.json
```
git 钩子已在项目目录内运行，天然按项目隔离。

**Q8：中文控制台输出乱码？**
脚本/CLI 已内置 UTF-8 处理；在较旧的 Windows 终端可显式设置：
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

**Q9：post-commit 钩子想关掉/改掉？**
删除 `.git/hooks/post-commit` 即关；重新运行 `setup_agents.ps1` 会先备份原钩子
再覆盖（备份名 `post-commit.memagent.bak`）。

**Q10：memagent 与主流 agent 的记忆机制差异在哪？**
主流 coding agent 的记忆本质是"静态文件 + 线性历史"；memagent 补的是五个动态
机制——遗忘曲线、睡眠巩固、记忆再巩固、类型迁移、参数自适应（详见 README
《memagent 与主流 agent 的定位关系》对照表）。本指南的三种方式是"把动态层接到
静态 agent 上"的工程化路径。

---

## 相关入口速查

```powershell
python session_memory.py --export-agents-md          # 双格式全量导出（AGENTS.md + CLAUDE.md）
python session_memory.py --sync [--note] [--eval]    # 一键闭环：提炼 → 沉淀 → 刷新 → 评估
python session_memory.py --start [--topic]           # 开工注入块（粘贴给 agent）
python session_memory.py --write-context [文件]      # 生成独立 prompt 文件
python session_memory.py --inject-agents-md          # AGENTS.md 顶部动态区块
python session_memory.py --eval-agents-md [文件]     # 加载效果评估
python remember_agent.py --export-agents-md          # 固化知识双格式导出（个人级）
python remember_agent.py --demo                      # 记忆层注入演示（无 key 降级可跑）
python -m memagent                                   # 交互对话（/memories /plot /learn …）
powershell -ExecutionPolicy Bypass -File setup_agents.ps1   # 一键接入脚本
```

更深的机制细节（检索增强、合并簇摘要检索、唤醒链路）见
[`docs/retrieval_enhancement.md`](retrieval_enhancement.md)。
