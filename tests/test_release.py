"""Immutable release verification and activation history tests."""

import zipfile
import json
from pathlib import Path

import pytest

from memagent import release


def _wheel(tmp_path: Path, version: str, *, private: bool = False) -> Path:
    path = tmp_path / f"memagent_local-{version}-py3-none-any.whl"
    dist = f"memagent_local-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("memagent/__init__.py", f"__version__ = '{version}'\n")
        archive.writestr(
            f"{dist}/METADATA",
            f"Metadata-Version: 2.1\nName: memagent-local\nVersion: {version}\n",
        )
        archive.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        if private:
            archive.writestr("works/private.log", "secret")
    return path


def _fake_install(_wheel_path: Path, pending: Path) -> None:
    python = release._venv_python(pending)
    python.parent.mkdir(parents=True)
    python.write_text("fake", encoding="utf-8")


def test_verify_wheel_metadata_and_checksum(tmp_path):
    wheel = _wheel(tmp_path, "0.1.0")
    info = release.verify_wheel(wheel)
    assert info["name"] == "memagent-local"
    assert info["version"] == "0.1.0"
    assert release.verify_wheel(wheel, info["sha256"])["sha256"] == info["sha256"]
    with pytest.raises(release.ReleaseError):
        release.verify_wheel(wheel, "0" * 64)


def test_verify_wheel_rejects_private_runtime_data(tmp_path):
    with pytest.raises(release.ReleaseError):
        release.verify_wheel(_wheel(tmp_path, "0.1.0", private=True))


def test_install_rejects_adjacent_manifest_checksum_mismatch(tmp_path):
    wheel = _wheel(tmp_path, "0.1.0")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifact": {"filename": wheel.name, "sha256": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError):
        release.install_release(wheel, tmp_path / "runtime")


def test_install_and_rollback_switch_version_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "_install_wheel_runtime", _fake_install)
    runtime = tmp_path / "runtime"
    first = release.install_release(_wheel(tmp_path, "0.1.0"), runtime)
    second = release.install_release(_wheel(tmp_path, "0.2.0"), runtime)
    assert second["previous"] == first["current"]
    assert release.release_status(runtime)["healthy"] is True
    rolled_back = release.rollback_release(runtime)
    assert rolled_back["current"] == first["current"]
    assert rolled_back["rolled_back_from"] == second["current"]


def test_rollback_requires_previous_release(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "_install_wheel_runtime", _fake_install)
    runtime = tmp_path / "runtime"
    release.install_release(_wheel(tmp_path, "0.1.0"), runtime)
    with pytest.raises(release.ReleaseError):
        release.rollback_release(runtime)
