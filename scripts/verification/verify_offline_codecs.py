"""Check the notebook's offline installer and bundled Linux wheel integrity.

This does not execute Linux binaries on macOS. Kaggle still needs a fresh run.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile

root = Path(__file__).resolve().parents[2]
(root / "local/scratch").mkdir(parents=True, exist_ok=True)
notebook = json.loads((root / "notebooks/mae/mae_submission.ipynb").read_text())
source = "".join(notebook["cells"][1]["source"])
bootstrap = source.split('if Path("/kaggle/input").is_dir():')[0]
scope = {}
exec(compile(bootstrap, "offline_codec_setup", "exec"), scope)
install = scope["install_offline_gdcm"]
assets = root / "submission/dicom_codecs"
manifest = json.loads((assets / "codecs_manifest.json").read_text())
assert len(manifest["files"]) == 4
for name, digest in manifest["files"].items():
    wheel = assets / name
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        assert 'Version: 3.0.26' in metadata
        # GDCM is standalone: no NumPy or other Python dependency is suppressed.
        assert 'Requires-Dist:' not in metadata

with tempfile.TemporaryDirectory(prefix="offline-codec-check-") as temporary:
    destination = Path(temporary) / "installed"
    previous_path = list(sys.path)
    try:
        for version in ['cp310', 'cp311', 'cp312', 'cp313']:
            with patch('subprocess.run') as run:
                selected = install(assets, destination, version)
                command = run.call_args.args[0]
                assert f'-{version}-{version}-' in selected
                assert '--no-index' in command and '--no-deps' in command
                assert '--target' in command and str(destination) in command
                assert run.call_args.kwargs['check'] is True
        with patch('subprocess.run') as run:
            try:
                install(assets, destination, 'cp314')
            except RuntimeError as exc:
                assert 'No offline GDCM wheel' in str(exc)
            else:
                raise AssertionError('Unsupported Python must fail clearly')
            run.assert_not_called()
        corrupted = Path(temporary) / 'corrupted'
        corrupted.mkdir()
        broken_manifest = json.loads(json.dumps(manifest))
        for name in broken_manifest['files']:
            (corrupted / name).symlink_to(assets / name)
            broken_manifest['files'][name] = '0' * 64
        (corrupted / 'codecs_manifest.json').write_text(json.dumps(broken_manifest))
        with patch('subprocess.run') as run:
            try:
                install(corrupted, destination, 'cp311')
            except ValueError as exc:
                assert 'checksum mismatch' in str(exc)
            else:
                raise AssertionError('Corrupted wheel must fail before pip')
            run.assert_not_called()
    finally:
        sys.path[:] = previous_path

print('PASS: four wheel hashes/archives/metadata, Python selection, offline-only installation, missing wheel and checksum failures.')
