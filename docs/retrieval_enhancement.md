# 检索增强说明：expand_query（查询侧）与 substring_priority_order（结果侧）

memagent 的检索核心是**字符 n-gram 哈希嵌入 + 余弦相似度**（见 `memagent/embedding.py`），
它无需训练、无第三方依赖，但有两个固有弱点：

1. **对措辞敏感**——记忆里存的是「我昨天去吃了火锅」，用户问「昨天中午用餐了吗」，
   字符重叠不足会漏检；
2. **短查询的 rel 不可信**——1~2 个字的查询 n-gram 极少、向量稀疏，余弦相似度里
   哈希碰撞的成分压过真实语义重叠，不相干的记忆可能排到最前面。

`memagent/synonyms.py` 用两个机制分别补这两个洞，它们**作用在流水线的两端**：

| | `expand_query` | `substring_priority_order` |
|---|---|---|
| 作用位置 | **查询侧**（打分之前，改查询） | **结果侧**（排序之后，改顺序） |
| 解决的问题 | 措辞不同 → **漏检**（recall） | 短查询碰撞 → **错序**（precision） |
| 机制 | 生成查询变体，对每条记忆取变体相似度的 **max** | 内容/摘要**字面包含**查询词的记忆排最前（大小写不敏感，组内按 rel×强度） |
| 对 rel 的影响 | 真实相关记忆的 rel **只升不降** | 不改 rel，只改返回顺序 |
| 关键参数 | 词表（18 词族 + 人称互换） | 短词阈值（默认 3 字以下） |
| 配置项 | `AgentConfig.query_expansion` | `AgentConfig.rerank_short_query` / `rerank_short_len` |
| 成本 | 每查询最多 8 次嵌入 | 短查询时一次 O(n log n) 排序 |

一句话分工：**`expand_query` 决定「哪些记忆能进入候选」，「substring_priority_order`
决定「候选里谁排最前」**——前者是召回，后者是排序。

---

## 机制一：expand_query——查询侧扩展

### 要解决的问题

`retrieve()` 对每条记忆算 `cos(查询向量, 记忆向量)`。n-gram 向量只在**字符重叠**处
产生信号：

- 书面语 vs 口语：「用餐」vs「吃」——零重叠；
- 第二人称问句：「您叫什么名字」vs 记忆「我叫小林」——「您/你」指用户自己，与
  「我」字开头的记忆不重叠。

措辞差得越远，真实相关记忆的 rel 越低，越容易被挤出 top-k——**漏检**。

### 机制

`expand_query(text)` 把查询扩展为一组**变体**（原始查询恒在首位）：

```python
expand_query("昨天中午用餐了吗，请问您的姓名")
# → ['昨天中午用餐了吗，请问您的姓名',     ← 原文
#    '昨天中午用餐了吗，请问我的姓名',     ← 您的 → 我的（人称互换）
#    '昨天中午吃了吗，请问您的姓名',       ← 用餐 → 吃（同义词）
#    '昨天中午用餐了吗，请问您的名字']     ← 姓名 → 名字
```

两类变体：

1. **人称互换**（`PRONOUN_SWAPS`）：疑问句里的「您/你」→「我」——
   「您昨天去爬山了吗」→「我昨天去爬山了吗」，与事实记忆「我昨天去爬山」直接重叠；
2. **同义词替换**（`SYNONYM_GROUPS`，18 个词族）：词族内**罕见词替换为组首常见
   口语词**（方向固定书面→口语）——「用餐」→「吃」、「姓名」→「名字」、
   「观看」→「看」。命中词已是常见词时不生成无益变体，把变体额度留给其他命中组。

`retrieve()` 对每条记忆取**所有变体相似度的最大值**：

```python
rel = max(cosine_similarity(q, mem.embedding) for q in qvs) * boost
```

原始查询恒在变体列表里，所以 max 只会**提高**真实相关记忆的 rel、不会降低任何
记忆——这是「扩展不会变差」的不变量。

### 实测效果

| 场景 | 关闭 | 开启 | 提升 |
|---|---|---|---|
| 「您叫什么名字」vs「我叫小林」 | rel 0.17 | rel 0.30 | 1.8× |
| 「昨天中午用餐了吗」vs「我昨天去吃了火锅」 | rel 0.23 | rel 0.33 | 1.4× |

### 适用场景

- 问法是**书面语/正式措辞**，记忆是口语化表达（或反之）——词族覆盖的日常词；
- 问句以「您/你」开头指代用户自己——人称互换直接命中的场景；
- 任何措辞可能不一致的对话检索（`respond()`、`remember_agent` 对话注入）。

### 不适用 / 局限

- **超出词表**：18 个词族只覆盖日常高频词，冷门同义词（如「攀岩」vs「爬山」不在
  词族内）不生效；
- **不连续命中**：子串替换要求词连续出现——「昨天中午**用了什么餐**」里「用餐」
  被「了什么」隔开，`"用餐" in text` 为 False，不替换。这本身是 n-gram 检索
  对措辞敏感的另一个写照；
- **变体数上限 8**：命中过多时截断，先到先得；
- 它解决的是**召回**，不解决短查询的**排序噪声**——那是下一个机制的事。

---

## 机制二：substring_priority_order——结果侧重排

### 要解决的问题

短查询（默认 < 3 字）的嵌入向量只有少量 bigram 参与，向量稀疏且任意两个稀疏
向量都可能因哈希碰撞产生不可忽略的余弦值。实测：查询「触底」（2 字）时，完全
不相干的「对照实验靠可注入时钟」rel 高达 0.23，排在真正含「触底」的记忆前面。

问题不在 rel 本身，而在**短查询的 rel 无法区分谁更相关**——此时按 total
（rel × 强度）排序，噪声记忆会顶到最前面，且**测试效应会优先强化噪声记忆**
（被回忆 → touch → 再巩固漂移），污染记忆本身。

### 机制

对短查询（`len(查询.strip()) < short_len`），把命中列表重排为：
**内容/摘要字面包含查询词的记忆排最前**。子串检查**大小写不敏感**（两侧都
lower）——与 n-gram 嵌入的归一化语义对齐（`ngrams()` 里 `text.lower()`），
「ai」查「AI 分类」、英文记忆「GPU 显存」查「gpu」都能命中。

```python
topic_l = topic.strip().lower()
sorted(hits, key=lambda h: (topic_l not in h.memory.content.lower(), -score(h)))
# score = score_of(h)；retrieve() 传 score_of=lambda h: h.total（rel×强度）
```

- **组内按 rel×强度（total）降序**——`retrieve()` 通过 `score_of` 传入：
  低相关但高强度的含词记忆（如碰巧含词、实际无关）不再压过更高相关（但强度
  略低）的含词记忆；缺省 `score_of=None` 时回退纯强度（与旧版一致）；
- 不含词的记忆排后面（同样按 rel×强度 降序）——碰撞噪声被压下去，但不是被丢弃；
- 长查询（≥ 阈值）**完全不重排**，行为与旧版一致；
- 子串检查同时覆盖 `content` 与 Cold 摘要（`summary`），压缩过的记忆也能被字面命中；
- 它发生在 **测试效应循环之前、截断 top-k 之前**，所以：命中强化跟随子串优先
  （返回给你看的记忆才是被强化的），且含词记忆即使按 total 排在 k 名之外也会被召回。

### 为什么短查询需要特殊对待

| 查询长度 | 参与 n-gram | 向量 | rel 的可靠性 |
|---|---|---|---|
| 1~2 字 | 仅 bigram | 稀疏 | 碰撞主导，不可信 |
| ≥ 3 字 | bigram + trigram | 较稠密 | 真实重叠占优，可信 |

阈值 3 是「trigram 开始参与」的分界线——这既是经验值也是机制依据，可通过
`rerank_short_len` 自定义（见下）。

### 实测效果

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 「触底」top1 | ❌ 碰撞噪声「可注入时钟」 | ✅ 含词记忆「触底时间」 |
| 「触底时间」（4 字，默认阈值 3） | 噪声 total 0.16 排前 | （默认不重排，维持原样） |
| 同上，`rerank_short_len=5` | — | ✅ 含词记忆（total 0.139）排最前 |

### 适用场景

- **短词主题检索**：`session_memory --start --topic` 的 1~2 字主题（如「触底」、
  「τ 学习」）；
- **短词对话注入**：`remember_agent` 对话里出现短查询；
- **任何核心 `retrieve()` 调用**——重排已下沉到 `retrieve()` 内部，所有下游入口
  （回复引用 / 对话注入 / 主题检索）统一受益，无需各自实现。

### 不适用 / 局限

- **长查询**：rel 已可信，重排反而可能破坏 total 排序的合理性，不触发；
- **无条目含查询词**：退化为纯强度降序（此时它只做了无害的排序）；
- **字面匹配的局限**：子串包含是字符级、且重排只对短查询触发——不过
  `retrieve()` 在打分前已做单点查询归一化（strip + 小写，见流水线 ⓪），
  rel 计算与所有文本级检查全链路大小写无关，英文大小写差异不再是问题；
  组内排序已兼顾 rel 信号（score_of = total），但**组边界仍绝对**：含词记忆
  无论 total 多低都排在全部不含词记忆之前——这是子串优先的核心不变量；
  它也不是模糊匹配——「触底」查不到「触底时间」之外的近似表达；
- 它解决**排序**，不解决措辞不同的**漏检**——查询侧才是召回的关键。

---

## 两个机制如何协作（retrieve 流水线）

```
query
  │ ⓪ 查询归一化（retrieve 内单点）：strip + lower——与 n-gram 嵌入的
  │    小写归一化对齐，所有文本级检查与 rel 计算全链路大小写无关；
  │    用户原句保留在调用方（respond/对话注入），此处只做检索内部归一化
  ▼
  │ ① expand_query() → 变体 [q0, q1, …]（仅 query_expansion=True 时）
  ▼
对每条记忆  rel = max(cos(变体, 记忆向量)) × Hot 加成 × turn 惩罚
  ▼
total = rel × strength
  ▼
按 total 降序排序
  │ ② substring_priority_order()（仅短查询且 rerank_short_query=True 时）
  │    ——含词记忆排前，组内按 total（rel×强度）降序 = ① 组内保持原 total 序
  ▼
测试效应（touch / 采样 / 再巩固）→ 截断 top-k → 返回
```

- **① 在打分之前**：决定每条记忆的 rel 有多高 → 决定谁能进 top-k（召回）；
- **② 在排序之后、强化与截断之前**：决定返回顺序与强化优先级（排序）。

两者互补：`expand_query` 负责「措辞不同也能**找到**」，`substring_priority_order`
负责「短查询时**排对**」。一个典型的组合场景：用户问「触底时间」——扩展变体
把同义词可能引入的措辞差消掉，若查询够短（自定义阈值后）再按字面包含重排。

### 配置速查

```python
from memagent import MemoryAgent
from memagent.agent import AgentConfig

MemoryAgent(cfg=AgentConfig(
    query_expansion=True,          # 查询侧扩展，默认开；False = rel 与旧版完全一致
    rerank_short_query=True,       # 结果侧重排，默认开；False = 返回顺序与旧版完全一致
    rerank_short_len=3,            # 短词阈值（= synonyms.SHORT_QUERY_LEN），可放宽到 5
))
```

三个开关**相互独立**，可任意组合：例如只信字面匹配（扩展关 + 重排开）、或只做
措辞宽容（扩展开 + 重排关）。

### 调试建议：判断是哪个机制在起作用

| 现象 | 排查方法 |
|---|---|
| 措辞不同但相关的记忆没被召回 | 关 `query_expansion` 对比 rel——若明显下降，是扩展在起作用 |
| 短查询返回了不相干的记忆且排最前 | 关 `rerank_short_query` 对比顺序——若顺序恢复 total 序，是重排在起作用 |
| 两者都不生效 | 词表外词汇（扩展）或查询 ≥ 阈值（重排）——考虑调阈值或扩词族 |

---

## 延伸：多源合并簇的摘要检索（find_memories / /memories / retrieve）

`sleep()` 把相似的 Warm 记忆聚类合并成 Cold 簇：`cl[0]` 降级为 Cold，
**content 只留第一源原文、summary 是各源合并后的抽取摘要、originals 保留全部
源的原文**（无损降权）。下面用真实合并簇说明「词只在摘要」如何影响搜索。

### 合并簇长什么样

三条相似记忆合并成一簇（实测）：

| 记忆 | 内容 | 合并后去向 |
|---|---|---|
| m1（簇首） | 我昨天去学了 python | → Cold **content** |
| m2 | 我昨天去学了 python 和 **AI** 编程 | → **summary** |
| m3 | 我昨天去学了 python 和 **GPU** 部署 | → 只进 **originals** |

压缩后 Cold 记忆：

```
content : 我昨天去学了 python
summary : 我昨天去学了 python； 我昨天去学了 python 和 AI 编程
originals: {m1: …python…, m2: …AI 编程…, m3: …GPU 部署…}
```

注意摘要上限 2 句（`extractive_summary(max_sentences=2)`）——m3 的「GPU 部署」
被挤出摘要，只留在 originals。

### 三个搜索面对应三种命中

`find_memories` / `/memories <词>` 的搜索面 = **content + summary + originals**
（全部小写、空格多词 = 同时包含），实测：

| 关键词 | 命中面 | find_memories | retrieve（嵌入检索） |
|---|---|---|---|
| python | content | ✅ 1 条 | ✅ 命中（摘要也含） |
| AI | **summary** | ✅ 1 条 | ✅ rel 0.097（摘要嵌入含 AI） |
| GPU / 部署 | **originals** | ✅ 1 条 | ❌ rel 0.000（摘要嵌入零重叠） |
| 火锅 | 三个面都不在 | ❌ 0 条 | ❌ |

### 关键行为差异：搜索面 ≠ 检索面

- **搜索（find_memories / /memories）**：子串扫描三个面，能"找到"压缩进 originals
  的任何词——包括被摘要挤掉的细节（GPU 依然可搜）；
- **检索（retrieve）**：只按**摘要嵌入**算相似度——"想不起来"只在 originals 里的
  词（GPU rel 0.000，低于 0.05 相关门槛，语义检索不可及）。

还有第三面：**注入/展示面**——Cold 命中（via_summary）在模板回复、LLM 注入
（`_generate_reply` / `remember_agent.injected_from` / `session_memory` 注入块）
里统一显示**摘要文本**而非深藏 content（content 可能不含命中词）；
`/memories` 列表与可视化同样用 `summary or content`。

这正是 Cold 层的设计分工：**摘要 = 索引（可检索），originals = 深藏（可搜索、
可唤醒）**。找回"只在 originals 的词"的路径：`/memories GPU` 定位 →
`/recall <id>` 唤醒（content 重建为摘要文本）→ 之后 GPU 才进入检索面。

`/recall` 是 **move 语义**：唤醒后原 Cold 记忆从仓库移除、重建的 Warm 记忆
**继承 originals**（深藏细节随行）——Cold↔Warm 往返（再次闲置又被压回 Cold）
不产生记忆增殖，且 `demote_to_cold` 对 originals 采用**合并**而非覆盖，
往返不丢深藏词（实测：合并簇 2 条 → 唤醒 1 条 → 压回 1 条，originals 由 2 变 3
——前一轮深藏 + 本轮自身，"味道"始终可搜）。

### 摘要重嵌入对 rel 与重排的影响（实测）

`demote_to_cold` 会把嵌入向量**重建为摘要嵌入**（`embed_text(summary)`）——
同一记忆的 rel 随摘要丢词而变：

| 查询词 | rel（content 嵌入，Warm） | rel（摘要嵌入，Cold） |
|---|---|---|
| GPU / 部署（被摘要丢掉） | 0.285 / 0.164 | **0.000**（跌破 0.05 门槛） |
| python（摘要保留） | 0.493 | **0.655**（摘要更短更稠密，反升） |

重排对此有一个**兜底**：`substring_priority_order` 的检查面是 content + summary，
而 Cold 的 content 仍保留第一源原文——所以"content 含词但摘要丢弃"的记忆（rel≈0）
仍被子串优先找回排最前（实测 total 0.000 压过碰撞噪声 0.238）；但兜底不越权：
同含词的组内仍按 rel×强度 排序，被兜底的 Cold（rel≈0）排在真实命中之后。

### 与两个机制的关系

- `substring_priority_order` 的重排检查面是 **content + summary**（不含
  originals）——与 retrieve 一致：只在 originals 的词既不参与 rel、也不触发
  子串优先，检索侧两个机制对"深藏词"同样不可见（但 content 中的词会被
  content+summary 检查面兜底，见上表）；
- `find_memories` 的 originals 覆盖是**搜索面专属**——这是两个检索入口
  （对话检索 vs 内容搜索）最实质的分工差异。

---

## 与更高级方案的边界

这两个机制都是**零依赖启发式**：词表固定、字面包含、无语义理解。它们的价值是
在无 LLM 环境下用最小成本修掉 n-gram 检索的两个系统性弱点；真正的语义级检索
（同义不同字、概念联想）仍属于 LLM 分类/回复生成器的范畴——本项目里
`LLMClassifier` 负责类型识别、`LLMResponder` 负责回复生成，检索层保持轻量。
