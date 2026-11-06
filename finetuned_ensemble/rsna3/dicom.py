"""Rendu LPS, champ centré de 153.6 mm. Géométrie absente : exclusion tracée.

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
