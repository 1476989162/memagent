"""Build, verify, install, and roll back immutable MemAgent releases."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import venv
import zipfile
from pathlib import Path

from .cli import enable_utf8
from .io_utils import FileLock, atomic_write_json, atomic_write_text


class ReleaseError(RuntimeError):
    """A release artifact or runtime transition is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_wheel(wheel: str | Path, expected_sha256: str | None = None) -> dict:
    path = Path(wheel).resolve()
    if not path.is_file():
        raise ReleaseError(f"wheel not found: {path}")
    digest = _sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ReleaseError(f"checksum mismatch for {path.name}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise ReleaseError("wheel must contain exactly one METADATA file")
            metadata = email.parser.Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
    except (zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise ReleaseError(f"invalid wheel: {path}") from exc

    lowered = [name.replace("\\", "/").lower() for name in names]
    forbidden = [
        name for name in lowered
        if name.endswith("/.env")
        or name == ".env"
        or name.startswith("works/")
        or "agent_memory" in name
        or "memories_session" in name
        or name.endswith(".log")
    ]
    if forbidden:
        raise ReleaseError(f"private runtime data found in wheel: {forbidden[0]}")
    name = metadata.get("Name", "")
    version = metadata.get("Version", "")
    if name != "memagent-local" or not version:
        raise ReleaseError(f"unexpected package metadata: {name} {version}")
    return {
        "name": name,
        "version": version,
        "filename": path.name,
        "sha256": digest,
        "size": path.stat().st_size,
        "path": str(path),
    }


def _release_manifest_checksum(wheel: Path) -> str | None:
    manifest_path = wheel.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = manifest["artifact"]
        if artifact["filename"] != wheel.name:
            raise ReleaseError(f"release manifest does not describe {wheel.name}")
        return str(artifact["sha256"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ReleaseError(f"invalid release manifest: {manifest_path}") from exc


def _source_revision(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_digest(project_root: Path) -> str:
    candidates = [
        project_root / "pyproject.toml",
        project_root / "README.md",
        project_root / "LICENSE",
        *sorted((project_root / "memagent").glob("**/*.py")),
    ]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_release(project_root: str | Path, output: str | Path) -> dict:
    project = Path(project_root).resolve()
    destination_root = Path(output).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="memagent-build-") as temp_name:
        temp = Path(temp_name)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(temp)],
            cwd=project, check=True,
        )
        wheels = list(temp.glob("memagent_local-*.whl"))
        if len(wheels) != 1:
            raise ReleaseError(f"expected one wheel, found {len(wheels)}")
        info = verify_wheel(wheels[0])
        release_dir = destination_root / f"v{info['version']}"
        release_dir.mkdir(parents=True, exist_ok=True)
        target = release_dir / info["filename"]
        if target.exists() and _sha256(target) != info["sha256"]:
            raise ReleaseError(
                f"release v{info['version']} is immutable and already has a different artifact"
            )
        if not target.exists():
            pending = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(wheels[0], pending)
            os.replace(pending, target)
        info = verify_wheel(target)
        manifest = {
            "schema_version": 1,
            "built_at": _utc_now(),
            "source_revision": _source_revision(project),
            "source_tree_sha256": _source_digest(project),
            "artifact": {key: value for key, value in info.items() if key != "path"},
        }
        manifest_path = release_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_artifact = existing.get("artifact", {})
            if existing_artifact.get("sha256") != info["sha256"]:
                raise ReleaseError(f"release manifest conflict: {manifest_path}")
            manifest = existing
        else:
            atomic_write_json(manifest_path, manifest)
            atomic_write_text(
                release_dir / "SHA256SUMS",
                f"{info['sha256']}  {info['filename']}\n",
                overwrite=False,
            )
        return {"release_dir": str(release_dir), "manifest": manifest}


def _load_state(runtime: Path) -> dict:
    state_path = runtime / "state.json"
    if not state_path.exists():
        return {"schema_version": 1, "current": None, "history": [], "versions": {}}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ReleaseError(f"invalid runtime state: {state_path}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("versions"), dict):
        raise ReleaseError(f"invalid runtime state schema: {state_path}")
    state.setdefault("history", [])
    return state


def _install_wheel_runtime(wheel: Path, pending: Path) -> None:
    venv.EnvBuilder(with_pip=True).create(pending)
    python = _venv_python(pending)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", str(wheel)],
        check=True,
    )
    subprocess.run([str(python), "-m", "memagent", "--version"], check=True)


def install_release(wheel: str | Path, runtime_dir: str | Path) -> dict:
    wheel_path = Path(wheel).resolve()
    info = verify_wheel(wheel_path, _release_manifest_checksum(wheel_path))
    runtime = Path(runtime_dir).resolve()
    versions = runtime / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    release_id = f"v{info['version']}-{info['sha256'][:12]}"
    final = versions / release_id
    with FileLock(runtime / ".release.lock", timeout=30.0):
        state = _load_state(runtime)
        if not final.exists():
            pending = versions / f".{release_id}.{uuid.uuid4().hex[:8]}.pending"
            try:
                _install_wheel_runtime(wheel_path, pending)
                atomic_write_json(
                    pending / "release.json",
                    {**{key: value for key, value in info.items() if key != "path"}, "installed_at": _utc_now()},
                )
                pending.rename(final)
            except Exception:
                shutil.rmtree(pending, ignore_errors=True)
                raise
        previous = state.get("current")
        if previous and previous != release_id:
            history = state.setdefault("history", [])
            if not history or history[-1] != previous:
                history.append(previous)
        state["current"] = release_id
        state.setdefault("versions", {})[release_id] = {
            "version": info["version"],
            "sha256": info["sha256"],
            "path": str(final),
            "activated_at": _utc_now(),
        }
        atomic_write_json(runtime / "state.json", state, backup=True)
    return {"current": release_id, "previous": previous, "runtime": str(runtime)}


def rollback_release(runtime_dir: str | Path) -> dict:
    runtime = Path(runtime_dir).resolve()
    with FileLock(runtime / ".release.lock", timeout=30.0):
        state = _load_state(runtime)
        current = state.get("current")
        history = state.setdefault("history", [])
        target = None
        while history:
            candidate = history.pop()
            if (runtime / "versions" / candidate).is_dir():
                target = candidate
                break
        if target is None:
            raise ReleaseError("no previous installed release is available")
        state["current"] = target
        state["versions"][target]["activated_at"] = _utc_now()
        atomic_write_json(runtime / "state.json", state, backup=True)
    return {"current": target, "rolled_back_from": current, "runtime": str(runtime)}


def release_status(runtime_dir: str | Path) -> dict:
    runtime = Path(runtime_dir).resolve()
    state = _load_state(runtime)
    current = state.get("current")
    current_path = runtime / "versions" / current if current else None
    return {
        **state,
        "runtime": str(runtime),
        "healthy": bool(current_path and _venv_python(current_path).is_file()),
    }


def run_current(runtime_dir: str | Path, arguments: list[str]) -> int:
    status = release_status(runtime_dir)
    if not status["healthy"]:
        raise ReleaseError("no healthy active release")
    root = Path(status["runtime"]) / "versions" / status["current"]
    return subprocess.run([str(_venv_python(root)), "-m", "memagent", *arguments]).returncode


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="MemAgent immutable release manager")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--project-root", default=".")
    build.add_argument("--output", default="releases")
    verify = sub.add_parser("verify")
    verify.add_argument("--wheel", required=True)
    verify.add_argument("--sha256")
    install = sub.add_parser("install")
    install.add_argument("--wheel", required=True)
    install.add_argument("--runtime", default=".runtime")
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--runtime", default=".runtime")
    status = sub.add_parser("status")
    status.add_argument("--runtime", default=".runtime")
    run = sub.add_parser("run")
    run.add_argument("--runtime", default=".runtime")
    run.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "build":
        result = build_release(args.project_root, args.output)
    elif args.command == "verify":
        result = verify_wheel(args.wheel, args.sha256)
    elif args.command == "install":
        result = install_release(args.wheel, args.runtime)
    elif args.command == "rollback":
        result = rollback_release(args.runtime)
    elif args.command == "status":
        result = release_status(args.runtime)
    else:
        run_args = args.args[1:] if args.args[:1] == ["--"] else args.args
        return run_current(args.runtime, run_args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
