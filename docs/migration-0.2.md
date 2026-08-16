# v0.2 migration

## Persistence

旧 JSON 无需转换。首次通过 `MemoryAgent.save()` 保存后，会新增 `meta.agent_state`，用于恢复兴趣、图谱、成长、认知与探索状态，同时生成 `.bak`。

如果另一个进程在当前实例加载后修改了同一 JSON，保存会抛出 `ConcurrentWriteError`。重新加载后再执行操作，不要覆盖该异常。

## Autonomous jobs

后台默认从无限循环改为 10 轮，连续失败 3 次退出。确需无限运行时显式传入 `--cycles 0`，并由进程管理器负责重启和预算监控。

## Work title migration

定名前已有章节时可执行：

```powershell
memagent --migrate-work 未命名作品 新书名 --works-dir works
```

若新书名目录没有章节，旧目录会迁入新目录；若新目录已经有章节，旧作品会归档到 `新书名/legacy/未命名作品`，绝不覆盖现有章节。

## Responder plugins

`respond()` 的 `timeout`、`memories` 与 `persona_extras` 均为可选能力。v0.2 会检查插件签名，只传入其支持的关键字参数。
