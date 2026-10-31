# TEST-ONLY: real C compiler fixture; not Ascend/CANN acceptance.
import argparse
import ctypes
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tests/ut/_tools"))
sys.path.insert(0, str(ROOT / ".github/workflows/scripts"))
build_cache_command = importlib.import_module("build_cache_test_utils").build_cache_command
validate_and_copy = importlib.import_module("nightly_csrc_l0").validate_and_copy
write_manifest = importlib.import_module("nightly_csrc_l0").write_manifest


def build(expected):
    fixture = ROOT / ".nightly-validation"
    fixture.mkdir(exist_ok=True)
    source = fixture / "native.c"
    source.write_text("int answer(void) { return 42; }\n")
    builder = fixture / "builder.py"
    builder.write_text(
        "import subprocess,sys\nfrom pathlib import Path\n"
        "source,output,counter=map(Path,sys.argv[1:])\n"
        "if (source.parent/'forbid-compile').exists(): raise RuntimeError('warm path compiled')\n"
        "output.mkdir(parents=True,exist_ok=True)\n"
        "subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(output/'vllm_ascend_C.so')],check=True)\n"
        "counter.write_text('compiled')\n"
    )
    counter = fixture / "compiler-invoked"
    if expected == "HIT":
        (fixture / "forbid-compile").touch()
    command = build_cache_command(
        cache_root=Path(os.environ["VLLM_ASCEND_BUILD_CACHE_DIR"]),
        domain="third_party",
        unit="nightly-cpu-transport",
        output_dir=ROOT / "vllm_ascend",
        environment_profile="host-cxx",
        prepared_inputs=[source],
        recipe_values=[],
        environment_values=[],
        environment_tools=["cc"],
        build_command=[sys.executable, str(builder), str(source), str(ROOT / "vllm_ascend"), str(counter)],
        normalize_paths=[ROOT],
        artifact_includes=["*.so"],
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout)
    assert f"[build-cache] {expected} " in result.stdout
    assert counter.exists() == (expected == "MISS")
    if expected == "HIT":
        assert not (Path(os.environ["VLLM_ASCEND_BUILD_CACHE_DIR"]) / ".updated").exists()
    write_manifest(ROOT, os.environ["TEST_L0_KEY"])
    print(json.dumps({"inner_result": expected, "compiler_invoked": counter.exists()}))


def consume():
    destination = ROOT / ".nightly-consumer"
    assert validate_and_copy(ROOT, destination, os.environ["TEST_L0_KEY"])
    assert not (ROOT / ".nightly-validation/compiler-invoked").exists()
    library = ctypes.CDLL(str(destination / "vllm_ascend/vllm_ascend_C.so"))
    assert library.answer() == 42
    print("L0 binary restored across jobs; answer=42; no consumer compiler invocation")


def reject():
    library = ROOT / "vllm_ascend/vllm_ascend_C.so"
    library.write_bytes(library.read_bytes() + b"changed")
    assert not validate_and_copy(ROOT, ROOT / ".rejected", os.environ["TEST_L0_KEY"])
    assert not library.exists()
    print("Rejected changed native bytes; fallback may compile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("cold", "warm", "consume", "reject"))
    args = parser.parse_args()
    if args.operation in ("cold", "warm"):
        build("MISS" if args.operation == "cold" else "HIT")
    elif args.operation == "consume":
        consume()
    else:
        reject()
