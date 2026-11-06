"""NPZ lossless par étude, prédicteur réversible optionnel et shards reprenables."""
from __future__ import annotations

import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import UID, atomic_json, digest, read_json, require, sha256, source_hash
from .dicom import RENDER, render_study

STORAGE = {"format": "npz-adaptive-uint8-delta-x-v1", "compression": "deflate",
           "predictor": "horizontal_uint8_difference_modulo_256",
           "selection": "smallest_compressed_payload", "lossless": True}
DELTA_KEY = "images_delta_x_v1"


def pack(arrays, meta):
    """Choose raw or reversible horizontal deltas by their actual compressed size."""
    images = arrays["images"]
    require(images.dtype == np.uint8 and images.ndim >= 1 and images.shape[-1] > 0,
            "Le cache lossless requiert des images uint8 non vides")
    require(DELTA_KEY not in arrays and "metadata" not in arrays, "Clé NPZ réservée")
    metadata = np.frombuffer(json.dumps({**meta, "storage": STORAGE}).encode(), dtype=np.uint8)

    def compressed(values):
        buf = io.BytesIO()
        np.savez_compressed(buf, **values, metadata=metadata)
        return buf.getvalue()

    raw = compressed(arrays)
    residual = images.copy()
    residual[..., 1:] = images[..., 1:] - images[..., :-1]
    predicted = compressed({**{k: v for k, v in arrays.items() if k != "images"}, DELTA_KEY: residual})
    return predicted if len(predicted) < len(raw) else raw


def unpack(payload):
    """Restore exactly the uint8 pixels; legacy raw NPZ remain readable."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        require(("images" in arrays) != (DELTA_KEY in arrays), "Encodage des images NPZ ambigu ou absent")
        if DELTA_KEY in arrays:
            residual = arrays[DELTA_KEY]
            require(residual.dtype == np.uint8 and residual.ndim >= 1 and residual.shape[-1] > 0,
                    "Résidus lossless invalides")
            images = np.cumsum(residual, axis=-1, dtype=np.uint8)
        else:
            images = arrays["images"].copy()
        return {"images": images, **{k: arrays[k].copy() for k in ("slots", "positions")}}


def probe(studies, series_csv, dicom_root, output, n=75, centres=4, max_hours=8):
    """Échantillon déterministe par UID, sans lecture des labels."""
    require(50 <= n <= 100 and len(studies) >= n, "Le probe final requiert 50–100 études")
    series = pd.read_csv(series_csv, dtype=str)
    chosen = sorted(studies, key=lambda uid: digest(uid))[:n]
    start, rows = time.monotonic(), []
    for uid in chosen:
        arrays, metadata = render_study(uid, series[series[UID] == uid], dicom_root, centres)
        rows.append({UID: uid, "bytes": len(pack(arrays, metadata)), "windows": len(arrays["images"]),
                     "errors": metadata["errors"], "warnings": metadata["warnings"], "syntaxes": metadata["syntaxes"]})
        require(time.monotonic() - start < max_hours * 3600, "Probe CPU au-delà de son échéance")
    elapsed = time.monotonic() - start
    # 20% size and time safety margins; do not assume compression is constant.
    projection = sum(r["bytes"] for r in rows) / n * len(studies) * 1.2
    report = {"render": RENDER, "storage": STORAGE, "source_hash": source_hash(), "centres": centres,
              "series_sha256": sha256(series_csv), "studies_hash": digest(sorted(studies)),
              "studies": len(studies), "sample": rows, "elapsed_seconds": elapsed,
              "laterality_conflict_studies": [r[UID] for r in rows if r["warnings"]],
              "projected_bytes_with_margin": projection,
              "projected_seconds_with_margin": elapsed / n * len(studies) * 1.2,
              "free_bytes": shutil.disk_usage(Path(output).resolve().parent).free,
              "limit_bytes": 12 * (1 << 30), "max_cpu_hours": max_hours}
    atomic_json(output, report)
    return report


def prepare(studies, series_csv, dicom_root, output, centres=4, probe_path=None,
            shard_bytes=1 << 30, limit_bytes=12 << 30, max_hours=8, fixture=False, inference=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    studies = list(map(str, studies))
    require(len(set(studies)) == len(studies), "UID dupliqués en entrée du cache")
    require(centres in (2, 4), "Deux ou quatre centres requis")
    series = pd.read_csv(series_csv, dtype=str)
    require(set(studies).issubset(set(series[UID])), "Métadonnées de séries absentes")
    contract = {"render": RENDER, "storage": STORAGE, "centres": centres, "series_sha256": sha256(series_csv),
                "studies_hash": digest(sorted(studies)), "source_hash": source_hash(), "fixture": fixture,
                "purpose": "inference" if inference else "training"}
    if not fixture and not inference:
        require(probe_path is not None, "Probe CPU 50–100 études obligatoire avant cache définitif")
        profile = read_json(probe_path)
        for key in ("render", "storage", "centres", "series_sha256", "studies_hash", "source_hash"):
            require(profile.get(key) == contract[key], f"Probe/cache différents : {key} ; refaire le probe")
        require(profile["projected_bytes_with_margin"] <= limit_bytes,
                "Projection > 12 Gio : deux centres puis stockage/représentation à revoir avant tout GPU")
        require(profile["projected_seconds_with_margin"] <= max_hours * 3600,
                "Projection CPU incompatible avec l'échéance")
        existing_bytes = sum(p.stat().st_size for p in output.glob("shard-*.zip"))
        require(max(0., profile["projected_bytes_with_margin"] - existing_bytes) * 1.1 < shutil.disk_usage(output).free,
                "Disque libre insuffisant, marge incluse")
    start = time.monotonic()
    entries, shards = {}, []
    for path in sorted(output.glob("shard-*.json")):
        info = read_json(path)
        require(info["contract"] == contract, "Reprise de cache incompatible")
        require(sha256(output / info["file"]) == info["sha256"], "Shard corrompu")
        require(not (set(entries) & set(info["entries"])), "UID dupliqués dans les shards")
        entries.update(info["entries"])
        shards.append({k: info[k] for k in ("file", "sha256", "bytes")})
    require(set(entries).issubset(studies), "Reprise contenant des UID inattendus")
    total = sum(s["bytes"] for s in shards)
    archive, pending, pending_bytes = None, {}, 0
    shard_no = len(shards)

    def close_shard():
        nonlocal archive, pending, pending_bytes, total, shard_no
        if archive is None:
            return
        archive.close()
        name = f"shard-{shard_no:04d}.zip"
        os.replace(output / (name + ".tmp"), output / name)
        size = (output / name).stat().st_size
        info = {"contract": contract, "file": name, "sha256": sha256(output / name),
                "bytes": size, "entries": pending}
        atomic_json(output / f"shard-{shard_no:04d}.json", info)
        entries.update(pending)
        shards.append({k: info[k] for k in ("file", "sha256", "bytes")})
        total += size
        shard_no += 1
        archive, pending, pending_bytes = None, {}, 0

    try:
        for i, uid in enumerate(studies):
            if uid in entries:
                continue
            require(time.monotonic() - start < max_hours * 3600, "Échéance CPU atteinte ; relancer pour reprendre")
            arrays, metadata = render_study(uid, series[series[UID] == uid], dicom_root, centres)
            payload = pack(arrays, metadata)
            require(total + pending_bytes + len(payload) + (4 << 20) < limit_bytes, "Plafond cache dépassé")
            require(shutil.disk_usage(output).free > len(payload) + (256 << 20), "Disque presque plein")
            if archive is not None and pending_bytes + len(payload) > shard_bytes:
                close_shard()
            if archive is None:
                archive = zipfile.ZipFile(output / f"shard-{shard_no:04d}.zip.tmp", "w", compression=zipfile.ZIP_STORED, allowZip64=True)
            member = digest(uid) + ".npz"
            archive.writestr(member, payload)
            import hashlib
            pending[uid] = {"shard": f"shard-{shard_no:04d}.zip", "member": member,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "windows": len(arrays["images"]), "slot_counts": np.bincount(arrays["slots"], minlength=6).tolist(),
                            "errors": len(metadata["errors"]), "warnings": len(metadata["warnings"])}
            pending_bytes += len(payload)
            if (i + 1) % 25 == 0:
                print(f"Cache : {i + 1}/{len(studies)}, {(total + pending_bytes)/(1<<30):.2f} Gio", flush=True)
    finally:
        # Finished studies in the open shard remain resumable even after a decode failure.
        close_shard()
    require(set(entries) == set(studies), "Cache incomplet")
    manifest = {"contract": contract, "entries": entries, "shards": shards, "complete": True,
                "bytes": total, "session_seconds": time.monotonic() - start,
                "laterality_conflict_studies": [uid for uid in studies if entries[uid].get("warnings", 0)]}
    persisted = sum(p.stat().st_size for p in output.iterdir() if p.suffix in (".zip", ".json") and p.name != "manifest.json")
    require(persisted + len(json.dumps(manifest, indent=2, sort_keys=True).encode()) + 1 < limit_bytes,
            "Tous les index compris : plafond cache dépassé")
    atomic_json(output / "manifest.json", manifest)
    return manifest


class Cache:
    def __init__(self, path, verify=False, allow_fixture=False):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        require(self.manifest["complete"], "Cache non terminé")
        require(allow_fixture or not self.manifest["contract"]["fixture"], "Cache de fixture interdit en production")
        require(self.manifest["contract"]["render"] == RENDER, "Version de rendu incompatible")
        require(self.manifest["contract"].get("storage") in (None, STORAGE), "Format de stockage incompatible")
        if verify:
            for shard in self.manifest["shards"]:
                require(sha256(self.path / shard["file"]) == shard["sha256"], "Empreinte de shard invalide")
        self.entries = self.manifest["entries"]
        self.handles = {}

    def __getstate__(self):
        return {**self.__dict__, "handles": {}}

    def read(self, uid):
        entry = self.entries[uid]
        name = entry["shard"]
        if name not in self.handles:
            self.handles[name] = zipfile.ZipFile(self.path / name)
        with self.handles[name].open(entry["member"]) as f:
            payload = f.read()
        import hashlib
        require(hashlib.sha256(payload).hexdigest() == entry["sha256"], f"Étude cache corrompue : {uid}")
        return unpack(payload)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
