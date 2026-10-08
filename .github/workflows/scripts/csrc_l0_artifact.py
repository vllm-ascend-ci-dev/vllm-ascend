# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Transfer complete csrc outputs without following symlinks or shipping L1."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ARTIFACT_NAME = "csrc-l0.tar.gz"
CSRC_DIRECTORIES = ("_cann_ops_custom", "lib", "include")


def verify_outputs(source_root: Path) -> None:
    package = source_root / "vllm_ascend"
    if not any(path.is_file() for path in package.glob("*.so")):
        raise ValueError("complete L0 requires a top-level csrc shared library")
    custom_ops = package / "_cann_ops_custom"
    if not custom_ops.is_dir() or not any(
        path.is_file() and not path.name.startswith(".") for path in custom_ops.rglob("*")
    ):
        raise ValueError("complete L0 requires non-empty _cann_ops_custom outputs")


def archive_sha256(archive: Path) -> str:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(source_root: Path, artifact_dir: Path) -> None:
    verify_outputs(source_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    archive = artifact_dir / ARTIFACT_NAME
    package = source_root / "vllm_ascend"
    paths = sorted(package.glob("*.so"))
    paths += [package / name for name in CSRC_DIRECTORIES if (package / name).exists()]
    # upload-artifact itself does not preserve Unix permissions or symlinks.
    # Archive only the existing L0 outputs, never the source tree or L1 cache.
    with tarfile.open(archive, "w:gz", compresslevel=1, dereference=False) as transport:
        for path in paths:
            transport.add(path, arcname=path.relative_to(source_root).as_posix())
    (artifact_dir / f"{ARTIFACT_NAME}.sha256").write_text(archive_sha256(archive), encoding="utf-8")


def is_csrc_member(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "vllm_ascend":
        return False
    if len(path.parts) == 2 and path.suffix == ".so":
        return True
    return len(path.parts) >= 2 and path.parts[1] in CSRC_DIRECTORIES


def unpack(source_root: Path, artifact_dir: Path) -> None:
    archive = artifact_dir / ARTIFACT_NAME
    expected = (artifact_dir / f"{ARTIFACT_NAME}.sha256").read_text(encoding="utf-8").strip()
    if archive_sha256(archive) != expected:
        raise ValueError("L0 archive checksum mismatch; CPU artifact transport failed")
    # Validate in a fresh directory: stale checkout outputs must not make an
    # incomplete incoming artifact appear complete.
    with tempfile.TemporaryDirectory(prefix="csrc-l0-") as temporary:
        stage = Path(temporary)
        with tarfile.open(archive, "r:gz") as transport:
            if not all(is_csrc_member(member.name) for member in transport.getmembers()):
                raise ValueError("L0 archive contains paths outside the csrc output set")
            # Reject path traversal, escaping links and special device files.
            transport.extractall(stage, filter="data")
        verify_outputs(stage)
        shutil.copytree(stage / "vllm_ascend", source_root / "vllm_ascend", symlinks=True, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack", "unpack", "verify"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "verify":
            verify_outputs(args.source_root.resolve())
            return
        if args.artifact_dir is None:
            parser.error("--artifact-dir is required for pack and unpack")
        operation = pack if args.operation == "pack" else unpack
        operation(args.source_root.resolve(), args.artifact_dir.resolve())
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"L0 artifact error: {error}\n")


if __name__ == "__main__":
    main()
