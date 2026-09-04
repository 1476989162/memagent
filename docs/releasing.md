# Release and rollback

MemAgent releases are immutable wheel artifacts. A version is identified by
its semantic version and SHA-256 digest. Never replace an artifact already
published under the same version; increment `memagent.__version__` instead.

## Build（v0.3.2 示例）

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m memagent.release build --output releases
python -m memagent.release verify --wheel releases/v0.3.2/memagent_local-0.3.2-py3-none-any.whl
```

The command creates:

```text
releases/v0.3.2/
  memagent_local-0.3.2-py3-none-any.whl
  manifest.json
  SHA256SUMS
```

The manifest records the package version, artifact size, digest, build time,
and Git revision when one is available. Private memory, work, environment, and
log files cause verification to fail. The current release smoke test also runs
against a clean managed runtime and checks the installed CLI version.

## Install into a managed runtime

```powershell
python -m memagent.release install `
  --wheel releases/v0.3.2/memagent_local-0.3.2-py3-none-any.whl `
  --runtime .runtime

python -m memagent.release status --runtime .runtime
python -m memagent.release run --runtime .runtime -- --version
```

Each artifact is installed into its own virtual environment under
`.runtime/versions/`. Installation completes and passes a package smoke test
before `state.json` is atomically switched to the new version.

## Roll back

```powershell
python -m memagent.release rollback --runtime .runtime
python -m memagent.release run --runtime .runtime -- --version
```

Rollback only changes the atomic active-version pointer. Installed version
directories are retained, so rollback does not depend on the network or on
rebuilding an old package.

Package rollback does not roll back user data. Use the persistence backup
workflow separately and stop all writers before restoring data.

## Tagging checklist

1. Update `memagent.__version__` and `CHANGELOG.md`.
2. Run the full test suite on Windows and Linux through CI.
3. Build and verify the immutable release directory.
4. Install the wheel into a clean managed runtime.
5. Exercise rollback to the previous installed artifact.
6. Create a signed Git tag only after all checks pass.

## Optional service smoke tests

REST 使用标准库即可验证：

```powershell
python -m memagent.server --host 127.0.0.1 --port 8399 --offline --persist smoke.json
```

MCP 需要额外安装服务依赖；可先验证 CLI 和工具注册：

```powershell
python -m pip install "memagent-local[mcp]"
python -m memagent.mcp_server --help
```

完整的 REST/MCP 端到端回归分别见 `tests/test_server_e2e.py` 与
`tests/test_mcp_server.py`。
