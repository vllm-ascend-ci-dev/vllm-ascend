# TEST-ONLY: no secret values in source; production backend is unchanged.
from pathlib import Path

for name in ("csrc-l1-restore", "csrc-l1-save"):
    path = Path(".github/actions") / name / "action.yaml"
    text = path.read_text()
    text = text.replace("ascend-ci-cache-hk", "obs-guiiyang1-ascend-test")
    text = text.replace("https://obs.ap-southeast-1.myhuaweicloud.com", "https://obs.cn-southwest-2.myhuaweicloud.com")
    path.write_text(text.replace("ap-southeast-1", "cn-southwest-2"))
