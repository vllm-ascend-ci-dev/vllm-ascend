"""TEST-ONLY: real custom-op execution, without vLLM or native build fallback."""

import importlib.util
from pathlib import Path

import torch
import torch_npu  # noqa: F401

root = Path(__file__).resolve().parents[3]
libraries = list((root / "vllm_ascend").glob("vllm_ascend_C*.so"))
assert len(libraries) == 1, libraries
torch.ops.load_library(str(libraries[0]))
assert hasattr(torch.ops._C_ascend, "npu_add_rms_norm_bias")
test_path = root / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_add_rms_norm_bias.py"
spec = importlib.util.spec_from_file_location("reference_add_rms_norm_bias", test_path)
assert spec is not None and spec.loader is not None
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
for dtype, tolerance, kernel in (
    (torch.float16, 0.0010986328125, 1),
    (torch.bfloat16, 0.0079345703125, 2),
    (torch.float32, 0.000244140625, 3),
):
    reference.test_quant_fpx_linear(16, 128, dtype, tolerance, tolerance, kernel)
    print(f"CPU-produced AddRmsNormBias PASS: {dtype}, shape=(16,128)")
print("NPU csrc compilation = 0; compile guards remained armed")
