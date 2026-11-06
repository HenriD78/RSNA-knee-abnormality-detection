"""Exercise the notebook recovery cell against the old immutable upload."""
import pytest
import json
import os
import subprocess
import sys

from rsna3.common import ROOT, source_hash


@pytest.mark.local_assets
def test_runtime_patch_reaches_kernel_subprocess_and_relative_assets(tmp_path):
    script = r'''
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(sys.argv[1])
WORK = Path(sys.argv[2])
cell = Path(sys.argv[3]).read_text()
exec(compile(cell, "recovery-cell", "exec"))
from rsna3.common import ROOT, input_tables, source_hash, spec
from rsna3.dicom import RENDER
from rsna3.cache import STORAGE
assert ROOT == RSNA3_RUNTIME
assert RENDER["version"] == "lps-oblique-2.5d-v2"
assert STORAGE["format"] == "npz-adaptive-uint8-delta-x-v1"
table, _ = input_tables(PROJECT)
assert len(table) == 4407 and spec("convnext_small")["model_id"] == "convnext_small"
first_hash = source_hash()
exec(compile(cell, "recovery-cell-rerun", "exec"))
from rsna3.common import source_hash
assert source_hash() == first_hash
code = "from rsna3.common import source_hash, ROOT; print(source_hash()); print(ROOT)"
result = subprocess.check_output([sys.executable, "-c", code], text=True).splitlines()
assert result == [first_hash, str(RSNA3_RUNTIME)]
print(json.dumps({"source_hash": first_hash, "runtime": str(RSNA3_RUNTIME)}))
'''
    project = ROOT / "dist/rsna3-training"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(project))
    result = subprocess.run([sys.executable, "-c", script, str(project), str(tmp_path),
                             str(ROOT / "patches/cell_fix_laterality.py")],
                            env=env, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["source_hash"] == source_hash()
    assert report["runtime"].startswith(str(tmp_path))
