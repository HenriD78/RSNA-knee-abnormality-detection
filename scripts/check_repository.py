"""Audit the Git file selection without executing notebooks or reading datasets."""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys
from urllib.parse import unquote

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
PRIVATE_DIRS = {
    ".venv", "venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
    "local", "sampled_rsna_data", "artifacts", "assets", "cache", "dist",
    "wheels", "rsna_resume", "rsna_reports", "bundle",
}
PRIVATE_SUFFIXES = {
    ".dcm", ".dicom", ".nii", ".pt", ".pth", ".ckpt", ".safetensors",
    ".onnx", ".npz", ".npy", ".whl", ".zip", ".tar", ".gz", ".h5", ".hdf5", ".pem", ".key",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{32,}|KGAT_[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(r'''["'](?:api_key|access_token|kaggle_key)["']\s*[:=]\s*["'][A-Za-z0-9_-]{24,}["']''', re.I),
)


def selected_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    return sorted({Path(p) for p in result.stdout.decode().split("\0") if p})


def check_notebook(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if cell.outputs or cell.execution_count is not None:
            raise ValueError(f"cell {i}: execution output must be cleared")
        # Transform IPython syntax in exploration notebooks before parsing.
        if any(line.lstrip().startswith(("%", "!", "?")) for line in cell.source.splitlines()):
            from IPython.core.inputtransformer2 import TransformerManager
            source = TransformerManager().transform_cell(cell.source)
        else:
            source = cell.source
        ast.parse(source, filename=f"{path.name}:cell{i}")


def main():
    files = selected_files()
    selected = {p.as_posix() for p in files}
    errors, total, notebooks = [], 0, 0
    for relative in files:
        path = ROOT / relative
        name = relative.as_posix()
        if path.is_symlink() or not path.is_file():
            errors.append(f"{name}: missing file or symlink")
            continue
        size = path.stat().st_size
        total += size
        private = any(part in PRIVATE_DIRS or part.startswith("wheels-linux-") for part in relative.parts)
        private |= relative.suffix.lower() in PRIVATE_SUFFIXES
        private |= relative.parts[0] in {"data", "outputs"} and relative.name != "README.md"
        private |= relative.parts[0] == "submission" and relative.name not in {"README.md", "verification.json"}
        private |= relative.name == "kaggle.json" or relative.name.startswith(".env")
        private |= relative.suffix.lower() == ".csv" and name != "finetuned_ensemble/suivi_folds.csv" and not name.startswith("docs/results/")
        if private:
            errors.append(f"{name}: private or generated artifact in Git selection")
            continue
        if size > MAX_BYTES:
            errors.append(f"{name}: exceeds the 5 MiB source-file limit")
            continue
        if path.suffix not in {".py", ".md", ".json", ".ipynb", ".txt", ".lock", ".yml", ".yaml", ".ini"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"{name}: possible credential; inspect locally")
        if re.search(r"/(?:Users|home)/[\w.-]+/", text) or re.search(r"[A-Z]:\\Users\\", text):
            errors.append(f"{name}: personal absolute path")
        if re.search(r"\b1\.2\.826\.0\.1\.3680043\.8\.498\.\d{10,}", text):
            errors.append(f"{name}: hard-coded local study or series identifier")
        try:
            if path.suffix == ".ipynb":
                check_notebook(path)
                notebooks += 1
            elif path.suffix == ".py":
                ast.parse(text, filename=name)
            elif path.suffix == ".json":
                json.loads(text)
        except (ValueError, SyntaxError, KeyError, nbformat.ValidationError) as error:
            errors.append(f"{name}: {error}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if target.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                target = unquote(target.split("#", 1)[0].strip("<>"))
                destination = (path.parent / target).resolve()
                if not destination.is_relative_to(ROOT):
                    errors.append(f"{name}: link outside repository: {target}")
                    continue
                linked = destination.relative_to(ROOT).as_posix()
                exists_in_git = linked in selected or any(p.startswith(linked.rstrip("/") + "/") for p in selected)
                if not exists_in_git:
                    errors.append(f"{name}: link absent from Git selection: {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"OK: {len(files)} public files, {notebooks} clean notebooks, {total / 1024**2:.2f} MiB.")
    print("Python/JSON syntax, Markdown targets, artifact exclusions and common secret patterns checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
