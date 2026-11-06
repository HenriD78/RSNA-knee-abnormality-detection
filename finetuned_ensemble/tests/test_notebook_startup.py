"""The first CPU Run All must stop after the measured probe by default."""
import json
from pathlib import Path

import pytest

from rsna3.common import ROOT


@pytest.mark.parametrize("oversized", [False, True])
def test_first_notebook_only_probes_by_default(tmp_path, oversized):
    notebook = json.loads((ROOT / "00_preparer_cache_cpu.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    commands = []

    def cli(*args):
        commands.append(args[0])
        if args[0] != "probe-cache":
            pytest.fail("The initial Run All started the full cache unexpectedly")
        destination = Path(args[args.index("--output") + 1])
        centres = args[args.index("--centres") + 1]
        report = {"elapsed_seconds": 120, "projected_seconds_with_margin": 120 * 4407 / 75 * 1.2,
                  "projected_bytes_with_margin": (13 if oversized and centres == 4 else 5) * (1 << 30)}
        destination.write_text(json.dumps(report))

    env = {"cli": cli, "WORK": tmp_path, "PROJECT": tmp_path, "COMPETITION_ROOT": tmp_path,
           "dicom_directory": lambda _: tmp_path, "json": json}
    for source in sources:
        if source.startswith("MAX_CPU_HOURS =") or source.startswith("BUILD_FULL_CACHE ="):
            exec(compile(source, "00_preparer_cache_cpu.ipynb", "exec"), env)
    assert commands == (["probe-cache", "probe-cache"] if oversized else ["probe-cache"])
    assert env["BUILD_FULL_CACHE"] is False
    assert env["centres"] == (2 if oversized else 4)
    assert not (tmp_path / "rsna3-cache/manifest.json").exists()
