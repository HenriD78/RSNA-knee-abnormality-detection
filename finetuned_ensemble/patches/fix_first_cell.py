"""Run this as the first notebook cell with IPython %run."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys

BUNDLE_ROOT = Path(os.environ.get('RSNA3_BUNDLE_ROOT', '/kaggle/input/datasets/henridargent/cpu-probes'))
PATCH_ROOT = Path(__file__).resolve().parent
if not (BUNDLE_ROOT / 'bundle_manifest.json').is_file():
    raise ValueError(f'Original training bundle not found at {BUNDLE_ROOT}; set RSNA3_BUNDLE_ROOT to its Kaggle Input path.')
module_spec = importlib.util.spec_from_file_location('rsna3_patched_bootstrap', PATCH_ROOT / 'kaggle_bootstrap.py')
bootstrap = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(bootstrap)
bootstrap.install(BUNDLE_ROOT, wheels_root=PATCH_ROOT)
PROJECT = BUNDLE_ROOT
WORK = Path('/kaggle/working')

def cli(*args):
    subprocess.run([sys.executable, '-m', 'rsna3', *map(str, args)], check=True)

cli('check', '--project', PROJECT)
print(f'First cell ready with Python {sys.version_info.major}.{sys.version_info.minor}. Continue with the next notebook cell.')
