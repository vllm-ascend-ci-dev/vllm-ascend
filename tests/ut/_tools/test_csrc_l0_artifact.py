# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

from .build_cache_test_utils import REPO_ROOT


@pytest.fixture
def transport():
    path = REPO_ROOT / ".github/workflows/scripts/csrc_l0_artifact.py"
    spec = importlib.util.spec_from_file_location("csrc_l0_artifact", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "producer"
    package = root / "vllm_ascend"
    custom_ops = package / "_cann_ops_custom"
    custom_ops.mkdir(parents=True)
    (custom_ops / "kernel.o").write_bytes(b"custom kernel")
    library = package / "extension.so"
    library.write_bytes(b"shared library")
    library.chmod(0o755)
    (package / "alias.so").symlink_to(library.name)
    (package / "__init__.py").write_text("not an L0 artifact")
    cache = root / "csrc/build_cache"
    cache.mkdir(parents=True)
    (cache / "entry").write_bytes(b"not for transport")
    return root


def test_complete_l0_round_trip_preserves_symlinks_modes_and_excludes_l1(transport, source: Path, tmp_path: Path):
    artifact = tmp_path / "artifact"
    consumer = tmp_path / "consumer"
    transport.pack(source, artifact)
    transport.unpack(consumer, artifact)

    package = consumer / "vllm_ascend"
    assert (package / "extension.so").read_bytes() == b"shared library"
    assert (package / "extension.so").stat().st_mode & 0o777 == 0o755
    assert (package / "alias.so").is_symlink()
    assert (package / "alias.so").readlink() == Path("extension.so")
    assert (package / "_cann_ops_custom/kernel.o").read_bytes() == b"custom kernel"
    assert not (consumer / "csrc/build_cache").exists()
    assert not (package / "__init__.py").exists()


@pytest.mark.parametrize("missing", ["extension.so", "_cann_ops_custom/kernel.o"])
def test_incomplete_outputs_cannot_be_published(transport, source: Path, tmp_path: Path, missing: str):
    (source / "vllm_ascend" / missing).unlink()
    with pytest.raises(ValueError, match="complete L0 requires"):
        transport.pack(source, tmp_path / "artifact")


def test_corrupt_artifact_fails_before_extracting(transport, source: Path, tmp_path: Path):
    artifact = tmp_path / "artifact"
    consumer = tmp_path / "consumer"
    transport.pack(source, artifact)
    (artifact / transport.ARTIFACT_NAME).write_bytes(b"corrupt transfer")
    with pytest.raises(ValueError, match="checksum mismatch"):
        transport.unpack(consumer, artifact)
    assert not consumer.exists()


def test_missing_artifact_fails_without_falling_back_to_compilation(transport, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        transport.unpack(tmp_path / "consumer", tmp_path / "absent")


def test_stale_checkout_cannot_mask_an_empty_artifact(transport, source: Path, tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    archive = artifact / transport.ARTIFACT_NAME
    with tarfile.open(archive, "w:gz"):
        pass
    (artifact / f"{transport.ARTIFACT_NAME}.sha256").write_text(transport.archive_sha256(archive))
    with pytest.raises(ValueError, match="complete L0 requires"):
        transport.unpack(source, artifact)
    assert (source / "vllm_ascend/extension.so").read_bytes() == b"shared library"


def test_gitkeep_is_not_a_custom_kernel(transport, source: Path, tmp_path: Path):
    custom = source / "vllm_ascend/_cann_ops_custom"
    (custom / "kernel.o").unlink()
    (custom / ".gitkeep").touch()
    with pytest.raises(ValueError, match="non-empty _cann_ops_custom"):
        transport.pack(source, tmp_path / "artifact")


@pytest.mark.parametrize("name", ["../outside.so", "/outside.so", "vllm_ascend/__init__.py", "csrc/build_cache/entry"])
def test_non_csrc_paths_cannot_be_restored(transport, tmp_path: Path, name: str):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    archive = artifact / transport.ARTIFACT_NAME
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    (artifact / f"{transport.ARTIFACT_NAME}.sha256").write_text(transport.archive_sha256(archive))
    with pytest.raises(ValueError, match="outside the csrc output set"):
        transport.unpack(tmp_path / "consumer", artifact)


def test_escaping_symlink_cannot_be_restored(transport, tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    archive = artifact / transport.ARTIFACT_NAME
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("vllm_ascend/extension.so")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside.so"
        tar.addfile(member)
    (artifact / f"{transport.ARTIFACT_NAME}.sha256").write_text(transport.archive_sha256(archive))
    with pytest.raises(tarfile.FilterError):
        transport.unpack(tmp_path / "consumer", artifact)
