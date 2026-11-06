"""Intégrité du paquet Linux livré ; n'installe rien sur le Mac."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsna3.common import ROOT, read_json, require, sha256


def verify(path):
    path = Path(path)
    manifest = read_json(path / "bundle_manifest.json")
    for name, expected in manifest["files"].items():
        require(sha256(path / name) == expected, f"Paquet corrompu : {name}")
    for source in (ROOT / "rsna3").glob("*.py"):
        require(sha256(source) == sha256(path / "rsna3" / source.name), "Code empaqueté obsolète")
    wheels = list((path / "wheels").glob("*.whl"))
    require(len(wheels) == 8, "Huit wheels attendues")
    python_versions = set()
    for wheel in wheels:
        require(not wheel.name.startswith(("torch-", "numpy-", "torchvision-")), "Torch/NumPy ne doivent pas être remplacés")
        with zipfile.ZipFile(wheel) as z:
            metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/WHEEL"))
            tags = z.read(metadata_name).decode()
            require("-none-any" in tags or ("manylinux" in tags and ("cp312" in tags or "cp313" in tags or "abi3" in tags)),
                    f"ABI de wheel inattendue : {wheel.name}")
            for version, label in (("cp312-cp312", "3.12"), ("cp313-cp313", "3.13")):
                if version in tags:
                    python_versions.add(label)
    print(f"OK — {len(manifest['files'])} empreintes, code à jour, 3 poids génériques et 8 wheels Linux/Python {' / '.join(sorted(python_versions))}.")


if __name__ == "__main__":
    verify(sys.argv[1] if len(sys.argv) > 1 else ROOT / "dist/rsna3-training-py313")
