# Contributing

MemAgent Local is maintained as a proprietary local-first product. Changes
require authorization from the product owner.

## Development setup

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q memagent
```

Do not use real memory stores, API keys, generated novels, or production logs
as test fixtures. Tests must use temporary directories and fake responders.

## Change requirements

- Preserve backward compatibility for existing persistence JSON unless a
  migration is documented and tested.
- Add regression tests for storage, locking, release, and recovery changes.
- Never swallow `ConcurrentWriteError`, checksum failures, or corrupted store
  errors by overwriting the affected file.
- Update `CHANGELOG.md` for user-visible behavior.
- Run the release build and clean-install smoke test before tagging.

The repository does not accept generated artifacts, runtime data, `.env`
files, release caches, or local virtual environments.
