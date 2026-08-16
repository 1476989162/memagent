# MemAgent Local 0.2

MemAgent Local 是一个本地优先、单租户的长期记忆与自主学习引擎。它适合作为桌面 agent、个人知识助理和受控内容生产后台的嵌入式记忆层。

## 产品边界

- 本地数据所有权：默认只写本机 JSON、作品目录和日志。
- 离线可用：记忆写入、检索、遗忘、睡眠和可视化不依赖 LLM。
- 可选模型：通过 OpenAI 兼容端点启用分类、回复、演化和写作。
- 当前部署模型：单机、单租户、单个主写进程。

多用户 SaaS 不应直接共享一个 JSON 文件。该场景需要数据库事务、认证、租户隔离、配额、审计与数据删除 API。

## 可靠性承诺

- 记忆保存使用同目录临时文件、`fsync` 和原子替换。
- 更新现有记忆库时保留一个 `.bak` 快照；主文件损坏会尝试读取备份。
- 两个进程基于同一旧快照保存时，后写者收到 `ConcurrentWriteError`，不会静默覆盖。
- 章节写作使用作品级锁且拒绝覆盖已有章节。
- 章节正文低于目标字数 90% 或没有完整句末时拒绝落盘。
- 作品维护工具生成 SHA-256 清单、全书快照、覆盖事件记录和逐章修订档案。
- 自主后台默认运行 10 轮，连续失败 3 次熔断。
- 发布包带 SHA-256 清单并安装到版本化虚拟环境；激活指针原子切换，可离线回滚。
- 命名备份在恢复前后都校验结构与摘要，原目标按原始字节留存。

## 运维

```powershell
python -m pip install -e ".[dev]"
memagent --check --persist agent_memory.json
python -m memagent.work_admin audit --work "works\作品名"
python -m memagent.release status --runtime .runtime
python -m pytest -q
```

生产环境应使用独立系统账户运行，限制目录权限，仅开放必要的模型与研究域名，并通过进程管理器收集退出码和日志。

## 数据恢复

1. 停止所有写入该记忆库的进程。
2. 保留损坏文件用于调查。
3. 将同目录的 `<name>.json.bak` 复制为新的主文件。
4. 运行 `memagent --check --persist <path>`。

应用正常加载时会自动尝试备份，但不会自动删除损坏的主文件。

正式备份、恢复及发布回滚操作见 `docs/backup-restore.md` 与 `docs/releasing.md`。

## 发布门槛

- 全量测试在 Windows 和 Linux 上通过。
- `memagent --check` 通过。
- `.env`、真实记忆和作品产物未进入发布包。
- API 密钥已轮换并设置额度告警。
- 恢复演练和并发启动演练通过。
