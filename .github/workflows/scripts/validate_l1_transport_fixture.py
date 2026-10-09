# TEST-ONLY: deterministic action fixture, not an Ascend compiler acceptance test.
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

mode = sys.argv[1]
workspace = Path.cwd().resolve()
cache = Path(os.environ["VLLM_ASCEND_BUILD_CACHE_DIR"])
assert cache == workspace / "csrc/build_cache", (cache, workspace)
root = workspace / ".transport-fixture"
root.mkdir(exist_ok=True)
source_dir = root / "source"
source_dir.mkdir(exist_ok=True)
for index in range(3):
    (source_dir / f"input-{index}.txt").write_text(f"semantic-input-{index}\n")
tool = root / "fixture-compiler"
tool.write_text("#!/bin/sh\necho transport-fixture-compiler-v1\n")
tool.chmod(0o755)
guard = root / "fail-if-compiled"
if mode == "warm":
    guard.touch()
else:
    guard.unlink(missing_ok=True)
event_log = Path(os.environ["VLLM_ASCEND_BUILD_CACHE_EVENT_LOG"])
event_log.unlink(missing_ok=True)
for index in range(3):
    source = source_dir / f"input-{index}.txt"
    output = root / f"output-{index}"
    command = [
        "python3",
        str(workspace / "csrc/scripts/build_cache.py"),
        "run",
        "--cache-root",
        str(cache),
        "--domain",
        "custom_operator",
        "--unit",
        "transport_fixture",
        "--soc",
        "ascend910b",
        "--operator",
        f"TransportFixture{index}",
        "--action",
        f"TransportFixture{index}-0",
        "--operator-source",
        str(source_dir),
        "--repo-root",
        str(workspace),
        "--output-dir",
        str(output),
        "--publish-dir",
        str(root / f"published-{index}"),
        "--publish-state-dir",
        str(root / f"publish-state-{index}"),
        "--prepared-input",
        str(source),
        "--recipe-file",
        str(tool),
        "--recipe-value",
        "transport-fixture-v1",
        "--environment-profile",
        "host-cxx",
        "--environment-tool",
        str(tool),
        "--normalize-path",
        str(root),
        "--artifact-include",
        "kernel.o",
        "--",
        "bash",
        "-c",
        'if [ -e "$3/fail-if-compiled" ]; then echo "unexpected build on warm restore"; exit 97; fi; '
        'mkdir -p "$2"; cp "$1" "$2/kernel.o"',
        "_",
        str(source),
        str(output),
        str(root),
    ]
    subprocess.run(command, check=True)
    assert (root / f"published-{index}/kernel.o").read_bytes() == source.read_bytes()
events = [json.loads(line) for line in event_log.read_text().splitlines()]
counts = collections.Counter(e["status"] for e in events if e["event"] == "cache_result")
expected = {"HIT": 3} if mode == "warm" else {"MISS": 3, "SAVED": 3}
assert counts == expected, counts
assert not any(e["event"] == "lock_timeout" for e in events)
assert cache.joinpath(".updated").exists() == (mode != "warm")
print(
    json.dumps(
        {"mode": mode, "workspace": str(workspace), "cache": str(cache), "results": dict(counts), "lock_timeout": 0}
    )
)
