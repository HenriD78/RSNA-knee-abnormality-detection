"""Paste this entire file into a Kaggle cell after setup; no dataset upload needed.

Lossless cache storage and laterality patch, not a notebook generator. The original Input remains read-only.
"""
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

_RSNA3_BASE_FILES = {'__init__.py': '3a97d1e922d3cbe87fb33b8839dadf43bd53e1f24c11d668bcc3f6dad804bc75', '__main__.py': '77313b5367224a46c31869db93ff2a4904e19ddaa2d324b80a0034af7c14dfc1', 'common.py': 'e8a332fa61125321812574cd53d998d0e344e13af37956616c636cee86a6d081', 'data.py': '87d9e59a1341f8a24c4fdccfbe0e613abe6fb2f4c06881914ede8ecf75a14810', 'evaluate.py': '44c982d1f8a4ebab0dfb4e81ae79300669c09612d34d53ab592d18d9991542c7', 'inference.py': '28e1644346853a0f67cd530b0b9144f6744dac6bd5898a4ffd02e78cd98bcd36', 'model.py': '77480e18a5eff0fc28bf98272d0edb9a5878ecedfd0ba883f9ec0537aa6fa07d', 'pilot.py': '89ec9f26c1c6cda0a959fd053e505772dd9ec698f1c82047a053d0904d37b3c8', 'queue.py': '402397fa30e8ae1745de76820dd59cc0fc9d3bb60416388956321fd86ffb9fdf', 'train.py': '9df9d00472f8c3ce7d4b1ff53a8ab7cf29cf4be5b79a1946b007dee8dceed174'}
_RSNA3_PATCH_FILES = {
'dicom.py': r'''"""Rendu LPS, champ centré de 153.6 mm. Géométrie absente : exclusion tracée.

Axes image (droite, bas), normale positive : sagittal (+P,-S),+L ;
coronal (+L,-S),+P ; axial (+L,+P),+S. Aucun miroir selon le genou.
Les obliquités sont conservées dans le plan réel : projection des axes LPS sur
ce plan, puis rééchantillonnage physique. La latéralité DICOM est enregistrée.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import get_decoder
from scipy.ndimage import map_coordinates

from .common import require

RENDER = {"version": "lps-oblique-2.5d-v2", "resolution": 384, "field_mm": 153.6,
          "percentiles": [1., 99.5], "gap_factor": 1.8, "orientation_tolerance_deg": 3.,
          "geometry_missing": "exclude_and_log", "boundary": "only_complete_triplets",
          "intensity": "rescale_then_monochrome1_inversion_then_series_percentiles",
          "quantile_sampling": "each_slice_stride_to_at_most_4096_pixels",
          "laterality": "within_series_reject_LR;between_series_warn;no_side_inference_or_flip",
          "canonical": "LPS; sagittal P/-S/+L; coronal L/-S/+P; axial L/P/+S"}
PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


class CodecError(RuntimeError):
    pass


@dataclass
class Slice:
    pixels: np.ndarray
    ipp: np.ndarray
    iop: np.ndarray
    spacing: np.ndarray
    sop: str
    frame: int
    side: str
    syntax: str
    side_source: str = "unknown"
    laterality_tags: dict = field(default_factory=dict)
    frame_of_reference: str = ""


def _group(ds, frame, sequence, attribute):
    for container in (getattr(ds, "PerFrameFunctionalGroupsSequence", [])[frame:frame + 1],
                      getattr(ds, "SharedFunctionalGroupsSequence", [])):
        if container and hasattr(container[0], sequence):
            return getattr(getattr(container[0], sequence)[0], attribute, None)
    return getattr(ds, attribute, None)


def _laterality(ds, frame):
    """Keep tag provenance; a missing/blank specific tag falls back to the series."""
    tags = {"FrameLaterality": _group(ds, frame, "FrameAnatomySequence", "FrameLaterality"),
            "ImageLaterality": getattr(ds, "ImageLaterality", None),
            "Laterality": getattr(ds, "Laterality", None)}
    tags = {key: str(value).strip().upper() for key, value in tags.items() if value is not None}
    for source, value in tags.items():
        if value:
            return value, source, tags
    return "unknown", "unknown", tags


def decode(path):
    ds = pydicom.dcmread(path)
    syntax = ds.file_meta.TransferSyntaxUID
    decoder = get_decoder(syntax)
    if not decoder.is_available:
        raise CodecError(f"Codec manquant pour {syntax}: {decoder.missing_dependencies}")
    try:
        array = ds.pixel_array
    except Exception as e:
        if syntax.is_compressed:
            raise CodecError(f"Décodage compressé impossible {path}, {syntax}: {e}") from e
        raise
    require(int(getattr(ds, "SamplesPerPixel", 1)) == 1, "DICOM non monochrome")
    frames = int(getattr(ds, "NumberOfFrames", 1))
    if frames == 1 and array.ndim == 2:
        array = array[None]
    require(array.ndim == 3 and len(array) == frames, "Dimensions multi-frame invalides")
    result = []
    for i, pixels in enumerate(array):
        ipp = _group(ds, i, "PlanePositionSequence", "ImagePositionPatient")
        iop = _group(ds, i, "PlaneOrientationSequence", "ImageOrientationPatient")
        spacing = _group(ds, i, "PixelMeasuresSequence", "PixelSpacing")
        require(ipp is not None and iop is not None and spacing is not None,
                "Géométrie physique absente (pas de substitution JPEG/InstanceNumber)")
        ipp, iop, spacing = (np.asarray(v, float) for v in (ipp, iop, spacing))
        require(ipp.shape == (3,) and iop.shape == (6,) and spacing.shape == (2,), "Géométrie mal formée")
        require(all(np.isfinite(v).all() for v in (ipp, iop, spacing)) and (spacing > 0).all(),
                "Géométrie non finie ou espacement invalide")
        require(abs(np.dot(iop[:3], iop[3:])) < .01 and
                abs(np.linalg.norm(iop[:3]) - 1) < .01 and
                abs(np.linalg.norm(iop[3:]) - 1) < .01, "Axes IOP non orthonormaux")
        slope = _group(ds, i, "PixelValueTransformationSequence", "RescaleSlope")
        intercept = _group(ds, i, "PixelValueTransformationSequence", "RescaleIntercept")
        pixels = pixels.astype(np.float32) * float(1 if slope is None else slope)
        pixels += float(0 if intercept is None else intercept)
        if str(ds.PhotometricInterpretation) == "MONOCHROME1":
            pixels = -pixels
        require(np.isfinite(pixels).all(), "Intensités non finies")
        side, source, tags = _laterality(ds, i)
        result.append(Slice(pixels, ipp, iop, spacing, str(ds.SOPInstanceUID), i,
                            side, str(syntax), source, tags, str(getattr(ds, "FrameOfReferenceUID", ""))))
    return result


def ordered_slices(slices):
    require(len(slices) >= 3, "Moins de trois coupes exploitables")
    sides = {s.side.strip().upper() for s in slices if s.side.strip().upper() in ("L", "R")}
    require(len(sides) <= 1, "Latéralités contradictoires dans une série")
    reference = slices[0].iop
    normal = np.cross(reference[:3], reference[3:])
    normal /= np.linalg.norm(normal)
    axis = int(np.abs(normal).argmax())
    if normal[axis] < 0:
        normal = -normal
    cutoff = np.cos(np.deg2rad(RENDER["orientation_tolerance_deg"]))
    # Reject a mixed-orientation series instead of connecting incompatible frames.
    require(all(np.dot(s.iop[:3], reference[:3]) >= cutoff and
                np.dot(s.iop[3:], reference[3:]) >= cutoff for s in slices),
            "Changement d'orientation dans une série")
    seen, unique, duplicates = set(), [], 0
    for s in slices:
        key = (s.sop, s.frame)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
            unique.append(s)
    unique.sort(key=lambda s: (float(s.ipp @ normal), s.sop, s.frame))
    # Coincident physical locations are duplicate images too, even with another SOP UID.
    cleaned = []
    for s in unique:
        if cleaned and abs(float((s.ipp - cleaned[-1].ipp) @ normal)) < 1e-3:
            duplicates += 1
        else:
            cleaned.append(s)
    require(len(cleaned) >= 3, "Moins de trois positions physiques distinctes")
    coords = np.array([s.ipp @ normal for s in cleaned])
    gaps = np.diff(coords)
    step = float(np.median(gaps))
    good = np.flatnonzero((gaps[:-1] <= RENDER["gap_factor"] * step) &
                         (gaps[1:] <= RENDER["gap_factor"] * step)) + 1
    require(len(good) > 0, "Aucun triplet sans trou d'espacement")
    return cleaned, coords, good, axis, duplicates


def physical_render(s, plane_axis, lo, hi):
    normal = np.cross(s.iop[:3], s.iop[3:])
    normal /= np.linalg.norm(normal)
    right = np.array([0., 1., 0.]) if plane_axis == 0 else np.array([1., 0., 0.])
    right -= normal * (right @ normal)
    right /= np.linalg.norm(right)
    desired_down = np.array([0., 1., 0.]) if plane_axis == 2 else np.array([0., 0., -1.])
    down = desired_down - normal * (desired_down @ normal) - right * (desired_down @ right)
    down /= np.linalg.norm(down)
    rows, cols = s.pixels.shape
    centre = s.ipp + s.iop[:3] * (cols - 1) * s.spacing[1] / 2 + s.iop[3:] * (rows - 1) * s.spacing[0] / 2
    size = RENDER["resolution"]
    offsets = (np.arange(size) - (size - 1) / 2) * RENDER["field_mm"] / size
    points = centre[:, None, None] + right[:, None, None] * offsets[None, None, :] + down[:, None, None] * offsets[None, :, None]
    delta = points - s.ipp[:, None, None]
    rr = np.einsum("i,ihw->hw", s.iop[3:], delta) / s.spacing[0]
    cc = np.einsum("i,ihw->hw", s.iop[:3], delta) / s.spacing[1]
    image = map_coordinates(s.pixels, [rr, cc], order=1, mode="constant", cval=float(lo), prefilter=False)
    return np.rint(np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), centre.tolist()


def render_study(uid, series, dicom_root, centres=4):
    selected, errors, syntaxes = {}, [], set()
    for row in series.to_dict("records"):
        sid = str(row["SeriesInstanceUID"])
        files = sorted((Path(dicom_root) / uid / sid).rglob("*.dcm"))
        slices = []
        for path in files:
            try:
                slices.extend(decode(path))
            except CodecError:
                raise  # A broken codec must stop the cache, not silently empty a slot.
            except Exception as e:
                errors.append({"series": sid, "file": path.name, "reason": str(e)})
        try:
            ordered, coords, good, axis, duplicates = ordered_slices(slices)
            require(row["Anatomical_Plane"] in PLANES and PLANES[row["Anatomical_Plane"]] == axis,
                    "Plan déclaré et normale DICOM incompatibles")
            require(int(row["Fat_Suppression"]) in (0, 1), "Fat-sat inconnu")
            slot = axis + (0 if int(row["Fat_Suppression"]) else 3)
            candidate = (len(ordered), sid, ordered, coords, good, axis, duplicates)
            if slot not in selected or (-candidate[0], sid) < (-selected[slot][0], selected[slot][1]):
                selected[slot] = candidate
        except ValueError as e:
            errors.append({"series": sid, "reason": str(e), "files": len(files)})
    require(bool(selected), f"Étude entièrement inexploitable {uid}: {errors}")
    sides = {s.side.strip().upper() for candidate in selected.values() for s in candidate[2]
             if s.side.strip().upper() in ("L", "R")}
    warnings = []
    if len(sides) > 1:
        # Study-level MIL uses the declared study membership, with no anatomical
        # side inference, majority vote or cross-series coordinate registration.
        # Keep the ambiguity visible; different FrameOfReferenceUIDs do not prove
        # that these series share a physical coordinate system.
        warnings.append({"code": "cross_series_laterality_conflict", "study": uid,
                         "reason": "L/R tags disagree between selected series; anatomical side unresolved",
                         "series": [{"series": candidate[1], "slot": slot,
                                     "laterality": sorted({s.side for s in candidate[2]}),
                                     "sources": sorted({s.side_source for s in candidate[2]}),
                                     "frame_of_reference_uids": sorted({s.frame_of_reference for s in candidate[2]})}
                                    for slot, candidate in sorted(selected.items())]})
    images, slots, positions, windows = [], [], [], []
    for slot, (_, sid, ordered, coords, good, axis, duplicates) in sorted(selected.items()):
        samples = np.concatenate([s.pixels.ravel()[::max(1, int(np.ceil(s.pixels.size / 4096)))] for s in ordered])
        lo, hi = np.percentile(samples, RENDER["percentiles"])
        require(hi > lo, f"Série d'intensité constante : {sid}")
        centres_idx = good[np.unique(np.rint(np.linspace(0, len(good) - 1, min(centres, len(good)))).astype(int))]
        for j in centres_idx:
            triplet, physical = [], []
            for s in ordered[j - 1:j + 2]:
                image, centre = physical_render(s, axis, lo, hi)
                triplet.append(image)
                physical.append(centre)
                syntaxes.add(s.syntax)
            position = float(2 * (coords[j] - coords[0]) / (coords[-1] - coords[0]) - 1)
            images.append(np.stack(triplet))
            slots.append(slot)
            positions.append(position)
            windows.append({"study": uid, "series": sid, "slot": slot, "position": position,
                            "sops_frames": [[s.sop, s.frame] for s in ordered[j - 1:j + 2]],
                            "centres_lps_mm": physical, "iop": ordered[j].iop.tolist(),
                            "pixel_spacing": ordered[j].spacing.tolist(), "laterality": ordered[j].side,
                            "laterality_source": ordered[j].side_source,
                            "laterality_tags": ordered[j].laterality_tags,
                            "frame_of_reference_uid": ordered[j].frame_of_reference,
                            "intensity_bounds": [float(lo), float(hi)], "duplicate_slices_excluded": duplicates,
                            "valid": True})
    return {"images": np.stack(images), "slots": np.array(slots, np.int64),
            "positions": np.array(positions, np.float32)}, {"windows": windows, "errors": errors,
                                                          "warnings": warnings, "syntaxes": sorted(syntaxes)}
''',
'cache.py': r'''"""NPZ lossless par étude, prédicteur réversible optionnel et shards reprenables."""
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
''',
}


def activate_runtime_patch(project, work):
    project, work = Path(project).resolve(), Path(work).resolve()
    expected = dict(_RSNA3_BASE_FILES)
    expected.update({name: hashlib.sha256(source.encode()).hexdigest()
                     for name, source in _RSNA3_PATCH_FILES.items()})
    def hashes(package):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(package.glob("*.py"))}
    actual = hashes(project / "rsna3")
    if any(actual.get(name) != stamp for name, stamp in _RSNA3_BASE_FILES.items()):
        raise ValueError("This patch requires the original RSNA3 bundle; other source files have changed.")
    if set(actual) != set(expected):
        raise ValueError("Unexpected RSNA3 source files in the attached bundle.")
    stamp = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if actual == expected:
        runtime = project
    else:
        runtime = work / ("rsna3-runtime-" + stamp[:12])
        runtime.mkdir(parents=True, exist_ok=True)
        shutil.copytree(project / "rsna3", runtime / "rsna3", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name, source in _RSNA3_PATCH_FILES.items():
            target = runtime / "rsna3" / name
            temporary = target.with_suffix(".py.tmp")
            temporary.write_text(source)
            os.replace(temporary, target)
        # Preserve ROOT-relative specifications, tables and weights without copying them.
        for source in project.iterdir():
            if source.name in ("rsna3", "__pycache__"):
                continue
            target = runtime / source.name
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    target.unlink()
                    target.symlink_to(source, target_is_directory=source.is_dir())
            elif target.exists():
                raise ValueError(f"Unexpected file in patch runtime: {target}")
            else:
                target.symlink_to(source, target_is_directory=source.is_dir())
        if hashes(runtime / "rsna3") != expected:
            raise ValueError("Patched runtime failed SHA-256 verification.")
    location = str(runtime)
    paths = [location, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    sys.path[:] = [location, *[p for p in sys.path if p != location]]
    for name in list(sys.modules):
        if name == "rsna3" or name.startswith("rsna3."):
            del sys.modules[name]
    importlib.invalidate_caches()
    print("RSNA3 active: laterality warnings and lossless cache compression.")
    return runtime


RSNA3_RUNTIME = activate_runtime_patch(PROJECT, WORK)
from rsna3.cache import STORAGE as _RSNA3_ACTIVE_STORAGE
print("Storage verified:", _RSNA3_ACTIVE_STORAGE["format"])
