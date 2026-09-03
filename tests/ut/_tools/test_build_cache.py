# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE = REPO_ROOT / "csrc" / "scripts" / "build_cache.py"
_KEY_RE = re.compile(r"\bkey=([0-9a-f]{64})\b")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_builder(tmp_path: Path) -> Path:
    builder = tmp_path / "fake_builder.py"
    builder.write_text(
        '''from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
counter = Path(sys.argv[2])
mode = sys.argv[3]
artifact_name = sys.argv[4]

count = int(counter.read_text()) if counter.exists() else 0
count += 1
counter.write_text(str(count))

output_dir.mkdir(parents=True, exist_ok=True)

if mode == "symlink":
    target = output_dir / "ascend_protoc"
    target.write_text("#!/bin/sh\\necho fake-protoc\\n", encoding="utf-8")
    target.chmod(0o755)

    link = output_dir / "protoc"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to("ascend_protoc")
else:
    artifact = output_dir / artifact_name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    content = "fixed-artifact" if mode == "fixed" else f"artifact-{count}"
    artifact.write_text(content, encoding="utf-8")
''',
        encoding="utf-8",
    )
    return builder


def _run_cache(
    *,
    cache_root: Path,
    prepared_inputs: list[Path],
    output_dir: Path,
    builder: Path,
    counter: Path,
    recipe_values: list[str] | None = None,
    environment_values: list[str] | None = None,
    domain: str = "custom_operator",
    operator_source: Path | None = None,
    artifact_includes: list[str] | None = None,
    builder_mode: str = "counted",
    artifact_name: str = "kernel.o",
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ENGINE),
        "run",
        "--cache-root",
        str(cache_root),
        "--domain",
        domain,
        "--unit",
        "test_unit",
        "--output-dir",
        str(output_dir),
        "--environment-profile",
        "ascendc" if domain == "custom_operator" else "host-cxx",
        "--environment-tool",
        sys.executable,
    ]

    for path in prepared_inputs:
        command.extend(["--prepared-input", str(path)])

    for value in recipe_values or ["recipe=stable"]:
        command.extend(["--recipe-value", value])

    for value in environment_values or ["abi=test"]:
        command.extend(["--environment-value", value])

    for pattern in artifact_includes or []:
        command.extend(["--artifact-include", pattern])

    if domain == "custom_operator":
        if operator_source is None:
            operator_source = prepared_inputs[0]
        command.extend(
            [
                "--soc",
                "ascend910b",
                "--operator",
                "test_operator",
                "--action",
                "TestOperator-0",
                "--operator-source",
                str(operator_source),
            ]
        )

    command.extend(
        [
            "--",
            sys.executable,
            str(builder),
            str(output_dir),
            str(counter),
            builder_mode,
            artifact_name,
        ]
    )

    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _assert_success(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, (
        f"returncode={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def _extract_key(proc: subprocess.CompletedProcess[str]) -> str:
    matches = _KEY_RE.findall(proc.stdout)
    assert matches, f"no cache key in stdout:\n{proc.stdout}"
    assert len(set(matches)) == 1, f"multiple cache keys in stdout: {matches}"
    return matches[-1]


def _find_entries(cache_root: Path, domain: str, final_key: str) -> list[Path]:
    # Deliberately locate entries from manifest contents rather than assuming
    # the cache directory layout. This is what the old T6 integration check got wrong.
    domain_root = cache_root / domain
    entries: list[Path] = []
    if not domain_root.exists():
        return entries

    for manifest_path in domain_root.rglob("manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("domain") == domain and payload.get("final_key") == final_key:
            entries.append(manifest_path.parent)

    return sorted(entries)


def _only_entry(cache_root: Path, domain: str, final_key: str) -> Path:
    entries = _find_entries(cache_root, domain, final_key)
    assert len(entries) == 1, (
        f"expected exactly one entry for domain={domain} key={final_key}, "
        f"got {len(entries)}: {entries}"
    )
    return entries[0]


def _manifest(entry: Path) -> dict:
    return json.loads((entry / "manifest.json").read_text(encoding="utf-8"))


def _artifact_kind(artifact: dict) -> str:
    return artifact.get("kind", "file")


def _first_file_artifact(entry: Path) -> tuple[dict, Path]:
    manifest = _manifest(entry)
    files = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if _artifact_kind(artifact) == "file"
    ]
    assert files, f"entry has no regular-file artifacts: {entry}"
    artifact = files[0]
    return artifact, entry / "artifacts" / artifact["path"]


def _fresh_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _make_operator_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.cpp").write_text("int source = 1;\n", encoding="utf-8")

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "kernel.cpp").write_text("int prepared = 1;\n", encoding="utf-8")
    return source, prepared


def test_custom_operator_miss_then_hit_and_restore(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout
    assert "[build-cache] SAVED" in first.stdout
    assert counter.read_text() == "1"

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(second)
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"
    assert (output / "kernel.o").read_text(encoding="utf-8") == "artifact-1"


def test_prepared_input_change_invalidates_and_revert_hits_history(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    original = (prepared / "kernel.cpp").read_text(encoding="utf-8")

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    key_original = _extract_key(first)

    (prepared / "kernel.cpp").write_text("int prepared = 2;\n", encoding="utf-8")
    _fresh_dir(output)

    changed = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(changed)
    key_changed = _extract_key(changed)
    assert "[build-cache] MISS" in changed.stdout
    assert key_changed != key_original
    assert counter.read_text() == "2"

    (prepared / "kernel.cpp").write_text(original, encoding="utf-8")
    _fresh_dir(output)

    reverted = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(reverted)
    assert "[build-cache] HIT" in reverted.stdout
    assert _extract_key(reverted) == key_original
    assert counter.read_text() == "2"


def test_recipe_change_invalidates_cache(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["optimization=-O2"],
    )
    _assert_success(first)
    key_a = _extract_key(first)

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["optimization=-O0"],
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert _extract_key(second) != key_a
    assert counter.read_text() == "2"


def test_compiler_environment_change_invalidates_cache(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        environment_values=["toolkit=9.1.0"],
    )
    _assert_success(first)
    key_a = _extract_key(first)

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        environment_values=["toolkit=9.2.0"],
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert _extract_key(second) != key_a
    assert counter.read_text() == "2"


def test_operator_text_hash_is_identity_namespace_not_action_key(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    key = _extract_key(first)
    assert len(_find_entries(cache_root, "custom_operator", key)) == 1

    (source / "kernel.cpp").write_text("int source = 2;\n", encoding="utf-8")
    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert _extract_key(second) == key
    assert counter.read_text() == "2"
    assert len(_find_entries(cache_root, "custom_operator", key)) == 2


def test_active_corrupted_artifact_rebuilds_repairs_then_hits(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    key = _extract_key(first)
    entry = _only_entry(cache_root, "custom_operator", key)

    artifact_meta, cached_artifact = _first_file_artifact(entry)
    expected_sha = artifact_meta["sha256"]
    assert _sha256_file(cached_artifact) == expected_sha

    cached_artifact.write_bytes(cached_artifact.read_bytes() + b"\nCORRUPTED\n")
    assert _sha256_file(cached_artifact) != expected_sha

    _fresh_dir(output)

    rebuilt = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(rebuilt)
    assert _extract_key(rebuilt) == key
    assert "[build-cache] MISS" in rebuilt.stdout
    assert "[build-cache] SAVED" in rebuilt.stdout
    assert counter.read_text() == "2"

    repaired_entry = _only_entry(cache_root, "custom_operator", key)
    repaired_meta, repaired_artifact = _first_file_artifact(repaired_entry)
    assert _sha256_file(repaired_artifact) == repaired_meta["sha256"]

    _fresh_dir(output)

    warm = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(warm)
    assert "[build-cache] HIT" in warm.stdout
    assert _extract_key(warm) == key
    assert counter.read_text() == "2"


def test_identical_rebuild_output_is_still_cacheable(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["recipe=a"],
        builder_mode="fixed",
    )
    _assert_success(first)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["recipe=b"],
        builder_mode="fixed",
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert "[build-cache] SAVED" in second.stdout
    assert counter.read_text() == "2"


def test_third_party_whole_unit_restores_regular_product(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "third_party.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["*.a"],
        artifact_name="libtest.a",
    )
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout
    assert counter.read_text() == "1"

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["*.a"],
        artifact_name="libtest.a",
    )
    _assert_success(second)
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"
    assert (output / "libtest.a").read_text(encoding="utf-8") == "artifact-1"


def test_third_party_symlink_restores_link_and_internal_target(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "protobuf.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout

    key = _extract_key(first)
    entry = _only_entry(cache_root, "third_party", key)
    manifest = _manifest(entry)
    artifacts = {artifact["path"]: artifact for artifact in manifest["artifacts"]}

    assert artifacts["protoc"]["kind"] == "symlink"
    assert artifacts["protoc"]["target"] == "ascend_protoc"
    assert artifacts["ascend_protoc"]["kind"] == "file"

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(second)
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"

    protoc = output / "protoc"
    target = output / "ascend_protoc"
    assert protoc.is_symlink()
    assert os.readlink(protoc) == "ascend_protoc"
    assert target.is_file()
    assert os.access(protoc, os.X_OK)


def test_corrupted_cached_symlink_rebuilds(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "protobuf.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(first)

    key = _extract_key(first)
    entry = _only_entry(cache_root, "third_party", key)
    cached_link = entry / "artifacts" / "protoc"
    assert cached_link.is_symlink()
    cached_link.unlink()
    cached_link.symlink_to("wrong_target")

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert "[build-cache] SAVED" in second.stdout
    assert _extract_key(second) == key
    assert counter.read_text() == "2"
