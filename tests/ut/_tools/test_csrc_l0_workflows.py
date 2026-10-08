# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Static contracts for mandatory CPU L0 production and compile-free consumers."""

import json

import pytest
import yaml

from .build_cache_test_utils import REPO_ROOT

WORKFLOWS = REPO_ROOT / ".github/workflows"


def workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def test_cpu_producer_is_not_gated_by_obs_writer_secrets():
    jobs = workflow("_ensure_csrc_cache.yaml")["jobs"]
    prepare = jobs["prepare-cache-checks"]
    producer = jobs["prepare-l0-artifacts"]
    assert "continue-on-error" not in prepare
    assert "has_targets" in producer["if"]
    assert "credential" not in producer["if"]
    assert producer["strategy"]["matrix"]["target"] == "${{ fromJSON(needs.prepare-cache-checks.outputs.targets) }}"
    assert "continue_on_error" not in producer["with"]


def test_same_run_upload_is_mandatory_and_precedes_optional_shared_save():
    job = workflow("_build_csrc_cache.yaml")["jobs"]["build"]
    assert "continue-on-error" not in job
    steps = job["steps"]
    build = next(step for step in steps if step.get("name") == "Install vllm-ascend")
    assert build["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    # A failed restore must not leave the mandatory source build without a log path.
    assert build["env"]["VLLM_ASCEND_BUILD_CACHE_EVENT_LOG"] == "${{ runner.temp }}/csrc-cpu-build.jsonl"
    incremental = next(step for step in steps if step.get("id") == "incremental-cache-restore")
    assert incremental["continue-on-error"] == "true"
    incremental_save = next(step for step in steps if step.get("name") == "Save incremental build cache")
    assert incremental_save["continue-on-error"] == "true"
    upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v7")
    save = next(step for step in steps if step.get("uses") == "runs-on/cache/save@v5")
    assert "if" not in upload
    assert "continue-on-error" not in upload
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["name"] == save["with"]["key"]
    assert steps.index(upload) < steps.index(save)
    assert save["continue-on-error"] == "true"
    assert "OBS_CACHE_WRITE_ACCESS_KEY_ID != ''" in save["if"]
    assert "OBS_CACHE_WRITE_SECRET_ACCESS_KEY != ''" in save["if"]


@pytest.mark.parametrize("name", ["_selected_tests.yaml", "_selected_tests_upstream.yaml"])
def test_selected_consumer_downloads_l0_without_l1_or_native_fallback(name: str):
    jobs = workflow(name)["jobs"]
    job = jobs["selected-tests" if name == "_selected_tests.yaml" else "e2e-upstream_pr"]
    steps = job["steps"]
    download = next(
        step for step in steps if step.get("uses") == "./.csrc-cache-workflow/.github/actions/csrc-l0-download"
    )
    install = next(step for step in steps if step.get("name") == "Install vllm-project/vllm-ascend with device")
    assert install["env"]["COMPILE_CUSTOM_KERNELS"] == "0"
    assert steps.index(download) < steps.index(install)
    assert "continue-on-error" not in download
    assert not any("csrc-l1" in step.get("uses", "") for step in steps)
    assert not any("runs-on/cache" in step.get("uses", "") for step in steps)
    assert "CSRC_CACHE_HIT" not in install["run"]
    assert "--no-build-isolation" not in install["run"]


def test_all_pr_npu_consumers_wait_for_cpu_artifacts_and_share_fixed_base():
    jobs = workflow("pr_test.yaml")["jobs"]
    for name in ("run-selected-tests", "run-selected-tests-upstream", "run-selected-tests-a5"):
        job = jobs[name]
        assert "prepare-csrc-cache" in job["needs"]
        assert "needs.prepare-csrc-cache.result == 'success'" in job["if"]
        assert job["with"]["base_sha"] == jobs["prepare-csrc-cache"]["with"]["base_sha"]


def test_target_images_match_selected_runner_environments():
    targets = json.loads((WORKFLOWS / "scripts/csrc_cache_targets.json").read_text())
    runners = json.loads((WORKFLOWS / "scripts/runner_label.json").read_text())
    for runner in runners.values():
        target_id = runner.get("csrc_cache_target")
        if target_id:
            target = targets[target_id]
            assert target["image"] == f"swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:{runner['image_tag']}"
            assert target["device"] == runner["chip"]
