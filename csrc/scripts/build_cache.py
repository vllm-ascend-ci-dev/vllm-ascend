#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Local content-addressed build cache for vLLM-Ascend.
#
# Cache equivalence:
#   prepared_input_hash       - what is compiled?
#   recipe_hash               - how is it compiled?
#   compiler_environment_hash - what compiles it?
#
# The cache engine owns no static operator or third-party unit list.

from __future__ import annotations

import argparse
import contextlib
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Iterable, Sequence

SCHEMA_VERSION = 3
CHUNK_SIZE = 1024 * 1024
DEFAULT_EXCLUDES = (
    ".git",
    ".git/**",
    "__pycache__",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    "*.done",
    "*.log",
    "*.tmp",
)

_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".py",
    ".sh",
    ".cmake",
    ".json",
    ".ini",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(records: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(records):
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def _is_excluded(relative_path: str, excludes: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in excludes)


def _hash_prepared_inputs(
    paths: Sequence[Path],
    excludes: Sequence[str],
) -> tuple[str, list[dict]]:
    records: list[tuple[str, str]] = []
    manifest: list[dict] = []

    for input_index, raw_path in enumerate(paths):
        path = raw_path.resolve()
        label = f"input[{input_index}]"

        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(f"prepared input does not exist: {path}")

        if path.is_symlink():
            target = os.readlink(path)
            records.append((label, f"symlink:{target}"))
            manifest.append({"path": label, "kind": "symlink", "target": target})
            continue

        if path.is_file():
            file_hash = _sha256_file(path)
            logical_path = f"{label}/{path.name}"
            records.append((logical_path, file_hash))
            manifest.append(
                {"path": logical_path, "kind": "file", "sha256": file_hash}
            )
            continue

        for child in sorted(
            path.rglob("*"),
            key=lambda item: item.relative_to(path).as_posix(),
        ):
            relative = child.relative_to(path).as_posix()
            if _is_excluded(relative, excludes):
                continue

            logical_path = f"{label}/{relative}"
            if child.is_symlink():
                target = os.readlink(child)
                records.append((logical_path, f"symlink:{target}"))
                manifest.append(
                    {"path": logical_path, "kind": "symlink", "target": target}
                )
            elif child.is_file():
                file_hash = _sha256_file(child)
                records.append((logical_path, file_hash))
                manifest.append(
                    {"path": logical_path, "kind": "file", "sha256": file_hash}
                )

    return _canonical_hash(records), manifest


def _looks_text(path: Path, data: bytes) -> bool:
    if path.name == "CMakeLists.txt" or path.suffix.lower() in _TEXT_SUFFIXES:
        return b"\x00" not in data
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _git_tracked_files(source_dir: Path, repo_root: Path) -> list[Path] | None:
    try:
        relative = source_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None

    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "-z",
                "--",
                relative.as_posix(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    files: list[Path] = []
    for raw in proc.stdout.split(b"\x00"):
        if not raw:
            continue
        files.append(repo_root / os.fsdecode(raw))
    return files


def _hash_operator_text(
    source_dir: Path,
    repo_root: Path | None,
) -> tuple[str, list[dict]]:
    source_dir = source_dir.resolve()
    candidates = (
        _git_tracked_files(source_dir, repo_root)
        if repo_root is not None
        else None
    )

    if candidates is None:
        candidates = [path for path in source_dir.rglob("*") if path.is_file()]

    records: list[tuple[str, str]] = []
    manifest: list[dict] = []

    for path in sorted(candidates, key=lambda item: str(item)):
        if not path.exists() and not path.is_symlink():
            continue
        try:
            relative = path.resolve().relative_to(source_dir).as_posix()
        except ValueError:
            # A tracked symlink may resolve outside source_dir. Use the tracked
            # path identity rather than the resolved target path.
            try:
                relative = path.relative_to(source_dir).as_posix()
            except ValueError:
                continue

        if path.is_symlink():
            target = os.readlink(path)
            records.append((relative, f"symlink:{target}"))
            manifest.append(
                {"path": relative, "kind": "symlink", "target": target}
            )
            continue

        data = path.read_bytes()
        if not _looks_text(path, data):
            continue

        digest = _sha256_bytes(data)
        records.append((relative, digest))
        manifest.append({"path": relative, "sha256": digest})

    return _canonical_hash(records), manifest


def _normalize_text(text: str, normalize_paths: Sequence[Path]) -> str:
    normalized = text.replace("\\", "/")
    replacements: list[tuple[str, str]] = []

    for index, path in enumerate(normalize_paths):
        try:
            value = str(path.resolve()).replace("\\", "/")
        except FileNotFoundError:
            value = str(path.absolute()).replace("\\", "/")
        replacements.append((value.rstrip("/"), f"<PATH_{index}>"))

    for source, replacement in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source:
            normalized = normalized.replace(source, replacement)

    return normalized


def _hash_recipe(
    recipe_files: Sequence[Path],
    recipe_values: Sequence[str],
    command: Sequence[str],
    normalize_paths: Sequence[Path],
) -> tuple[str, list[dict]]:
    records: list[tuple[str, str]] = []
    manifest: list[dict] = []

    for index, raw_path in enumerate(recipe_files):
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"recipe file does not exist: {path}")

        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            value_hash = _sha256_bytes(data)
            kind = "binary"
        else:
            normalized = _normalize_text(text, normalize_paths)
            value_hash = _sha256_bytes(normalized.encode("utf-8"))
            kind = "text"

        records.append((f"recipe_file[{index}]", value_hash))
        manifest.append(
            {
                "index": index,
                "name": path.name,
                "kind": kind,
                "sha256": value_hash,
            }
        )

    for index, value in enumerate(recipe_values):
        normalized = _normalize_text(value, normalize_paths)
        records.append((f"recipe_value[{index}]", normalized))
        manifest.append(
            {
                "index": index,
                "kind": "value",
                "value": normalized,
            }
        )

    normalized_command = [
        _normalize_text(str(token), normalize_paths) for token in command
    ]
    records.append(
        (
            "original_command",
            json.dumps(
                normalized_command,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    )
    manifest.append({"kind": "command", "argv": normalized_command})

    return _canonical_hash(records), manifest


def _command_version(argv: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = proc.stdout.strip()
    return output or None


def _resolve_tool(tool: str) -> str | None:
    path = Path(tool).expanduser()
    if path.is_absolute() or os.sep in tool:
        return str(path) if path.exists() else None
    return shutil.which(tool)


def _candidate_cann_metadata_files() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))

    opp_path = os.environ.get("ASCEND_OPP_PATH")
    if opp_path:
        roots.append(Path(opp_path).parent)

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "version.info",
                root / "ascend_toolkit_install.info",
                root.parent / "version.info",
            ]
        )
    candidates.append(Path("/etc/ascend_install.info"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _hash_compiler_environment(
    profile: str,
    environment_files: Sequence[Path],
    environment_values: Sequence[str],
    environment_tools: Sequence[str],
) -> tuple[str, list[dict]]:
    records: list[tuple[str, str]] = [
        ("profile", profile),
        ("system", platform.system()),
        ("machine", platform.machine()),
    ]
    manifest: list[dict] = [
        {"name": "profile", "value": profile},
        {"name": "system", "value": platform.system()},
        {"name": "machine", "value": platform.machine()},
    ]

    default_tools: list[str] = []
    if profile == "host-cxx":
        default_tools = [] if environment_tools else ["cc", "c++", "cmake"]
    elif profile == "ascendc":
        default_tools = [] if environment_tools else ["bisheng", "ccec"]
        environment_files = list(environment_files) + _candidate_cann_metadata_files()
    else:
        raise ValueError(f"unknown environment profile: {profile}")

    seen_tools: set[str] = set()
    for index, tool in enumerate(list(environment_tools) + default_tools):
        resolved = _resolve_tool(tool)
        if resolved is None:
            continue
        canonical = str(Path(resolved).resolve())
        if canonical in seen_tools:
            continue
        seen_tools.add(canonical)

        output = _command_version([canonical, "--version"])
        if output is None:
            continue

        # The absolute installation path is intentionally not hashed. The
        # executable's reported identity/version is the semantic signal.
        tool_name = Path(canonical).name
        records.append((f"tool[{index}]:{tool_name}", output))
        manifest.append(
            {
                "name": f"tool[{index}]",
                "tool": tool_name,
                "version": output,
            }
        )

    seen_files: set[str] = set()
    for raw_path in environment_files:
        path = raw_path.expanduser()
        if not path.is_file():
            continue
        digest = _sha256_file(path)
        identity = path.name
        dedupe_key = f"{identity}:{digest}"
        if dedupe_key in seen_files:
            continue
        seen_files.add(dedupe_key)
        records.append((f"metadata:{identity}", digest))
        manifest.append(
            {
                "name": f"metadata:{identity}",
                "sha256": digest,
            }
        )

    for index, value in enumerate(environment_values):
        records.append((f"value[{index}]", value))
        manifest.append({"name": f"value[{index}]", "value": value})

    return _canonical_hash(records), manifest


def _artifact_signature(path: Path) -> tuple:
    # lstat() describes the directory entry itself instead of following a
    # symlink. A symlink may be a required build product (protobuf's `protoc`
    # is one such case), so it must participate in artifact discovery.
    stat = path.lstat()
    if path.is_symlink():
        return (
            "symlink",
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            os.readlink(path),
        )
    return ("file", stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _snapshot(root: Path) -> dict[str, tuple]:
    snapshot: dict[str, tuple] = {}
    if not root.exists():
        return snapshot

    for path in sorted(root.rglob("*")):
        if path.is_symlink() or path.is_file():
            snapshot[path.relative_to(root).as_posix()] = _artifact_signature(path)
    return snapshot


def _internal_symlink_target(root: Path, link: Path) -> Path | None:
    target = Path(os.readlink(link))
    if target.is_absolute():
        candidate = target
    else:
        candidate = link.parent / target

    root_abs = Path(os.path.abspath(root))
    candidate_abs = Path(os.path.abspath(os.path.normpath(candidate)))
    try:
        relative = candidate_abs.relative_to(root_abs)
    except ValueError:
        return None
    return root_abs / relative


def _expand_internal_symlink_targets(
    root: Path,
    selected: set[str],
) -> set[str]:
    # Caching only `protoc -> ascend_protoc` would restore a broken link
    # after deleting the transient build tree. Include the in-tree target
    # closure even when the target does not match ARTIFACT_INCLUDE.
    root = Path(os.path.abspath(root))
    expanded = set(selected)
    pending = list(selected)

    while pending:
        relative = pending.pop()
        path = root / relative
        if not path.is_symlink():
            continue

        target = _internal_symlink_target(root, path)
        if target is None:
            # External links belong to the environment. Preserve the link
            # itself but do not import external files into this cache entry.
            continue
        if not (target.is_symlink() or target.is_file()):
            raise RuntimeError(
                f"symlink artifact target does not exist: {path} -> "
                f"{os.readlink(path)}"
            )

        target_relative = target.relative_to(root).as_posix()
        if target_relative not in expanded:
            expanded.add(target_relative)
            pending.append(target_relative)

    return expanded


def _collect_artifacts(
    root: Path,
    before: dict[str, tuple],
    include_patterns: Sequence[str],
) -> list[str]:
    if not root.exists():
        return []

    if include_patterns:
        selected: set[str] = set()
        for path in root.rglob("*"):
            if not (path.is_symlink() or path.is_file()):
                continue
            relative = path.relative_to(root).as_posix()
            if any(
                fnmatch.fnmatch(relative, pattern)
                for pattern in include_patterns
            ):
                selected.add(relative)
        return sorted(_expand_internal_symlink_targets(root, selected))

    after = _snapshot(root)
    selected = {
        relative
        for relative, signature in after.items()
        if before.get(relative) != signature
    }
    return sorted(_expand_internal_symlink_targets(root, selected))


def _safe_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    if not safe:
        raise ValueError(f"invalid empty cache path component from {value!r}")
    return safe


def _entry_path(
    cache_root: Path,
    domain: str,
    unit: str,
    final_key: str,
    operator_text_hash: str | None,
    soc: str | None,
    operator: str | None,
    action: str | None,
) -> Path:
    if domain == "third_party":
        return cache_root / "third_party" / _safe_component(unit) / final_key

    if not soc or not operator or not action or not operator_text_hash:
        raise ValueError(
            "custom_operator cache requires --soc, --operator, --action, "
            "and --operator-source"
        )

    return (
        cache_root
        / "custom_operator"
        / _safe_component(soc)
        / _safe_component(operator)
        / operator_text_hash
        / _safe_component(action)
        / final_key
    )


def _load_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_entry(entry: Path, final_key: str) -> dict | None:
    manifest_path = entry / "manifest.json"
    artifact_root = entry / "artifacts"
    if not manifest_path.is_file() or not artifact_root.is_dir():
        return None

    manifest = _load_json(manifest_path, {})
    if manifest.get("schema") != SCHEMA_VERSION:
        return None
    if manifest.get("final_key") != final_key:
        return None

    for artifact in manifest.get("artifacts", []):
        path = artifact_root / artifact["path"]
        kind = artifact.get("kind", "file")

        if kind == "symlink":
            if not path.is_symlink():
                return None
            if os.readlink(path) != artifact.get("target"):
                return None
            continue

        if kind != "file":
            return None
        if not path.is_file() or path.is_symlink():
            return None
        if _sha256_file(path) != artifact["sha256"]:
            return None
    return manifest


def _remove_existing_artifact(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise RuntimeError(
            f"cannot restore file artifact over non-file path: {path}"
        )


def _restore_entry(entry: Path, output_dir: Path, manifest: dict) -> None:
    artifact_root = entry / "artifacts"

    # Restore real files first so relative symlinks become valid immediately
    # when they are created in the second pass.
    for artifact in manifest["artifacts"]:
        if artifact.get("kind", "file") != "file":
            continue
        source = artifact_root / artifact["path"]
        destination = output_dir / artifact["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_artifact(destination)
        shutil.copy2(source, destination)

    for artifact in manifest["artifacts"]:
        if artifact.get("kind") != "symlink":
            continue
        destination = output_dir / artifact["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_artifact(destination)
        destination.symlink_to(artifact["target"])


def _save_entry(
    entry: Path,
    output_dir: Path,
    artifact_paths: Sequence[str],
    manifest_base: dict,
) -> None:
    entry.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{entry.name}.tmp-", dir=str(entry.parent))
    )
    try:
        artifact_root = temp_dir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)

        artifacts: list[dict] = []
        for relative in artifact_paths:
            source = output_dir / relative
            destination = artifact_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            if source.is_symlink():
                target = os.readlink(source)
                destination.symlink_to(target)
                artifacts.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "target": target,
                    }
                )
                continue

            if not source.is_file():
                continue

            shutil.copy2(source, destination)
            artifacts.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": _sha256_file(destination),
                }
            )

        if not artifacts:
            raise RuntimeError(
                "build command succeeded but produced no cacheable artifacts "
                f"in {output_dir}"
            )

        manifest = dict(manifest_base)
        manifest["schema"] = SCHEMA_VERSION
        manifest["artifacts"] = artifacts
        manifest["created_at"] = int(time.time())
        _atomic_write_json(temp_dir / "manifest.json", manifest)

        for artifact in artifacts:
            cached = artifact_root / artifact["path"]
            if artifact["kind"] == "symlink":
                if not cached.is_symlink():
                    raise RuntimeError(
                        f"cached symlink disappeared: {artifact['path']}"
                    )
                if os.readlink(cached) != artifact["target"]:
                    raise RuntimeError(
                        f"cached symlink target changed: {artifact['path']}"
                    )
            elif _sha256_file(cached) != artifact["sha256"]:
                raise RuntimeError(
                    f"artifact verification failed: {artifact['path']}"
                )

        if entry.exists():
            shutil.rmtree(entry)
        os.replace(temp_dir, entry)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _logical_status(
    domain: str,
    previous_key: str | None,
    current_key: str,
    previous_operator_text_hash: str | None,
    operator_text_hash: str | None,
) -> str:
    if previous_key is None:
        return "NEW"
    if domain == "custom_operator":
        return (
            "UNCHANGED"
            if previous_operator_text_hash == operator_text_hash
            else "MODIFIED"
        )
    return "UNCHANGED" if previous_key == current_key else "MODIFIED"


def _update_index(
    cache_root: Path,
    domain: str,
    unit: str,
    final_key: str,
    prepared_input_hash: str,
    recipe_hash: str,
    environment_hash: str,
    cache_status: str,
    operator_text_hash: str | None,
    soc: str | None,
    operator: str | None,
    action: str | None,
) -> None:
    domain_root = cache_root / domain
    index_path = domain_root / "cache_index.json"
    lock_path = domain_root / "cache_index.json.lock"

    with _file_lock(lock_path):
        index = _load_json(
            index_path,
            {"schema": SCHEMA_VERSION, "units": {}},
        )
        if index.get("schema") != SCHEMA_VERSION:
            index = {"schema": SCHEMA_VERSION, "units": {}}

        if domain == "third_party":
            unit_key = unit
        else:
            unit_key = f"{soc}/{operator}/{action}"

        state = index["units"].setdefault(unit_key, {})
        previous_key = state.get("current_key")
        previous_operator_text_hash = state.get("operator_text_hash")
        logical_status = _logical_status(
            domain,
            previous_key,
            final_key,
            previous_operator_text_hash,
            operator_text_hash,
        )

        state["previous_key"] = previous_key
        state["current_key"] = final_key
        state["operator_text_hash"] = operator_text_hash
        state["prepared_input_hash"] = prepared_input_hash
        state["recipe_hash"] = recipe_hash
        state["compiler_environment_hash"] = environment_hash
        state["logical_status"] = logical_status
        state["cache_status"] = cache_status
        state["updated_at"] = int(time.time())

        history = state.setdefault("history", {})
        history[final_key] = {
            "operator_text_hash": operator_text_hash,
            "prepared_input_hash": prepared_input_hash,
            "recipe_hash": recipe_hash,
            "compiler_environment_hash": environment_hash,
            "logical_status": logical_status,
            "cache_status": cache_status,
            "updated_at": int(time.time()),
        }

        _atomic_write_json(index_path, index)


def _parse_set_env(values: Sequence[str]) -> dict[str, str]:
    environment = os.environ.copy()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--set-env expects NAME=VALUE, got {value!r}")
        name, content = value.split("=", 1)
        environment[name] = content
    return environment


def run(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root).expanduser().resolve()
    prepared_inputs = [Path(value) for value in args.prepared_input]
    recipe_files = [Path(value) for value in args.recipe_file]
    environment_files = [Path(value) for value in args.environment_file]
    normalize_paths = [Path(value) for value in args.normalize_path]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = list(args.command)
    if not command:
        raise ValueError("missing build command after --")

    excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)

    prepared_input_hash, prepared_manifest = _hash_prepared_inputs(
        prepared_inputs,
        excludes,
    )

    operator_text_hash: str | None = None
    operator_text_manifest: list[dict] = []
    if args.domain == "custom_operator":
        if not args.operator_source:
            raise ValueError(
                "custom_operator cache requires --operator-source"
            )
        repo_root = Path(args.repo_root) if args.repo_root else None
        (
            operator_text_hash,
            operator_text_manifest,
        ) = _hash_operator_text(
            Path(args.operator_source),
            repo_root,
        )

    recipe_hash, recipe_manifest = _hash_recipe(
        recipe_files,
        args.recipe_value,
        command,
        normalize_paths,
    )
    environment_hash, environment_manifest = _hash_compiler_environment(
        args.environment_profile,
        environment_files,
        args.environment_value,
        args.environment_tool,
    )
    final_key = _canonical_hash(
        [
            ("prepared_input_hash", prepared_input_hash),
            ("recipe_hash", recipe_hash),
            ("compiler_environment_hash", environment_hash),
        ]
    )

    entry = _entry_path(
        cache_root=cache_root,
        domain=args.domain,
        unit=args.unit,
        final_key=final_key,
        operator_text_hash=operator_text_hash,
        soc=args.soc,
        operator=args.operator,
        action=args.action,
    )

    lock_name = _sha256_bytes(str(entry).encode("utf-8"))
    entry_lock = cache_root / args.domain / ".locks" / f"{lock_name}.lock"

    manifest_base = {
        "domain": args.domain,
        "unit": args.unit,
        "soc": args.soc,
        "operator": args.operator,
        "action": args.action,
        "final_key": final_key,
        "operator_text_hash": operator_text_hash,
        "prepared_input_hash": prepared_input_hash,
        "recipe_hash": recipe_hash,
        "compiler_environment_hash": environment_hash,
        "operator_text_inputs": operator_text_manifest,
        "prepared_inputs": prepared_manifest,
        "recipe": recipe_manifest,
        "compiler_environment": environment_manifest,
    }

    with _file_lock(entry_lock):
        manifest = _validate_entry(entry, final_key)
        if manifest is not None:
            _restore_entry(entry, output_dir, manifest)
            print(
                f"[build-cache] HIT domain={args.domain} unit={args.unit} "
                f"key={final_key}",
                flush=True,
            )
            _update_index(
                cache_root,
                args.domain,
                args.unit,
                final_key,
                prepared_input_hash,
                recipe_hash,
                environment_hash,
                "HIT",
                operator_text_hash,
                args.soc,
                args.operator,
                args.action,
            )
            return 0

        print(
            f"[build-cache] MISS domain={args.domain} unit={args.unit} "
            f"key={final_key}",
            flush=True,
        )

        before = _snapshot(output_dir)
        environment = _parse_set_env(args.set_env)
        start = time.monotonic()
        process = subprocess.run(
            command,
            cwd=args.working_directory,
            env=environment,
            check=False,
            close_fds=False,
        )
        elapsed = time.monotonic() - start
        if process.returncode != 0:
            return process.returncode

        artifacts = _collect_artifacts(
            output_dir,
            before,
            args.artifact_include,
        )
        _save_entry(
            entry,
            output_dir,
            artifacts,
            {
                **manifest_base,
                "build_seconds": elapsed,
            },
        )
        print(
            f"[build-cache] SAVED domain={args.domain} unit={args.unit} "
            f"key={final_key} artifacts={len(artifacts)} "
            f"build_seconds={elapsed:.3f}",
            flush=True,
        )

        _update_index(
            cache_root,
            args.domain,
            args.unit,
            final_key,
            prepared_input_hash,
            recipe_hash,
            environment_hash,
            "MISS_BUILT",
            operator_text_hash,
            args.soc,
            args.operator,
            args.action,
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vLLM-Ascend local build cache"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--cache-root", required=True)
    run_parser.add_argument(
        "--domain",
        required=True,
        choices=("third_party", "custom_operator"),
    )
    run_parser.add_argument("--unit", required=True)
    run_parser.add_argument("--soc")
    run_parser.add_argument("--operator")
    run_parser.add_argument("--action")
    run_parser.add_argument("--operator-source")
    run_parser.add_argument("--repo-root")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--prepared-input", action="append", default=[])
    run_parser.add_argument("--recipe-file", action="append", default=[])
    run_parser.add_argument("--recipe-value", action="append", default=[])
    run_parser.add_argument(
        "--environment-profile",
        choices=("ascendc", "host-cxx"),
        required=True,
    )
    run_parser.add_argument("--environment-file", action="append", default=[])
    run_parser.add_argument("--environment-value", action="append", default=[])
    run_parser.add_argument("--environment-tool", action="append", default=[])
    run_parser.add_argument("--normalize-path", action="append", default=[])
    run_parser.add_argument("--artifact-include", action="append", default=[])
    run_parser.add_argument("--exclude", action="append", default=[])
    run_parser.add_argument("--set-env", action="append", default=[])
    run_parser.add_argument("--working-directory")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand != "run":
        parser.error(f"unsupported subcommand: {args.subcommand}")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
