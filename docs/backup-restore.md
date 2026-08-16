# Persistence backup and restore

MemAgent saves use a same-directory temporary file, `fsync`, atomic replace,
and stale-writer detection. The adjacent `.bak` file is a fast single-step
recovery aid. Named backups provide longer-lived recovery points.

## Create and verify a backup

```powershell
memagent-backup create --persist agent_memory.json --output backups
memagent-backup verify --file backups/agent_memory-<time>-<hash>.json
```

Each named backup has a manifest containing its SHA-256 digest, byte size,
record count, source path, and creation time.

## Restore

Stop every process writing the target store, then run:

```powershell
memagent-backup restore `
  --from backups/agent_memory-<time>-<hash>.json `
  --persist agent_memory.json

memagent --check --persist agent_memory.json
```

Restore validates the source before taking the target lock. The existing
target is preserved byte-for-byte in `restore-backups/`, the selected backup
is atomically published, and the resulting checksum is verified.

Do not manually merge memory JSON files. A process loaded before a restore
will detect the changed file signature and refuse a stale save.
