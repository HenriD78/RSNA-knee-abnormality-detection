"""Écritures atomiques et empreintes pour des reprises vérifiables."""

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_bytes(path, value):
    """Le fichier précédent reste lisible si une écriture est interrompue."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".partial_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, frame):
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.10g").encode())


def write_npz(path, **arrays):
    import numpy as np
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    atomic_bytes(path, buf.getvalue())


def write_checkpoint(path, payload):
    import torch
    buf = io.BytesIO()
    torch.save(payload, buf)
    atomic_bytes(path, buf.getvalue())
