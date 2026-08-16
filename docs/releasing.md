# Release and rollback

MemAgent releases are immutable wheel artifacts. A version is identified by
its semantic version and SHA-256 digest. Never replace an artifact already
published under the same version; increment `memagent.__version__` instead.

## Build

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m memagent.release build --output releases
```

The command creates:

```text
releases/v0.2.1/
  memagent_local-0.2.1-py3-none-any.whl
  manifest.json
  SHA256SUMS
```

The manifest records the package version, artifact size, digest, build time,
and Git revision when one is available. Private memory, work, environment, and
log files cause verification to fail.

## Install into a managed runtime

```powershell
python -m memagent.release install `
  --wheel releases/v0.2.1/memagent_local-0.2.1-py3-none-any.whl `
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
