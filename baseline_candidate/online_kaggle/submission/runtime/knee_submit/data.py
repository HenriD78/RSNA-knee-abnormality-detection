"""Prépare les DICOM du test COURANT sans lire le train ni les anciens exemples.

Six slots : trois plans × présence/absence de suppression de graisse. Chaque
slot retient la série ayant le plus d'images lisibles, puis huit images choisies
par hash. Les coupes restent un ensemble ; aucune adjacency 2.5D n'est inventée.
"""

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .constants import LABELS, PLANES, SERIES, UID
from .io import fingerprint, write_csv


def read_inputs(root):
    """Relit les UID lors de CHAQUE exécution, y compris celle du scoring caché."""
    root = Path(root)
    test = pd.read_csv(root / "test.csv", dtype={UID: str})
    if UID not in test or test.empty or test[UID].isna().any() or test[UID].duplicated().any():
        raise ValueError("test.csv vide, incomplet ou avec des UID dupliqués.")
    # Le template sert au schéma. Ses UID d'exemple ne bornent jamais le test caché.
    schema = pd.read_csv(root / "sample_submission.csv", nrows=0)
    if list(schema.columns) != [UID] + LABELS:
        raise ValueError("Schéma sample_submission.csv différent des douze findings attendus.")
    series = pd.read_csv(root / "test_series.csv", dtype={UID: str, SERIES: str})
    required = [UID, SERIES, "Anatomical_Plane", "Fat_Suppression"]
    if not set(required) <= set(series) or series[required].isna().any().any():
        raise ValueError("Métadonnées test_series.csv incomplètes.")
    if series[SERIES].duplicated().any() or set(series[UID]) != set(test[UID]):
        raise ValueError("Les séries ne correspondent pas exactement aux études test.")
    return test[UID].tolist(), series


def slot_of(plane, fat_suppression):
    if plane not in PLANES or fat_suppression not in (0, 1):
        raise ValueError(f"Métadonnées de slot invalides : {plane}, {fat_suppression}")
    return PLANES.index(plane) + 3 * (1 - int(fat_suppression))


def pixel_array_to_jpeg_pixels(frame):
    """Reproduit la conversion JPEG empirique auditée dans le projet.

    Min–max PAR IMAGE → uint8 → resize INTER_AREA 224 → JPEG qualité 95 →
    décodage Pillow. Aucun nouveau fenêtrage, retournement MONOCHROME1 ou
    redressement anatomique n'est ajouté par rapport aux JPEG d'entraînement.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2 or not frame.size or not np.isfinite(frame).all():
        raise ValueError("Image grise 2D finie et non vide attendue.")
    lo, hi = float(frame.min()), float(frame.max())
    scaled = (frame - lo) / (hi - lo) if hi > lo else np.zeros_like(frame)
    pixels = (scaled * 255).astype(np.uint8)
    pixels = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Échec de l'encodage JPEG intermédiaire.")
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        return np.array(image, dtype=np.uint8, copy=True)


def check_codecs():
    """Échoue avant le calcul si les formats compressés usuels sont indisponibles."""
    import pydicom
    from pydicom.pixels import get_decoder
    from pydicom.uid import JPEGLossless, JPEGLosslessSV1, JPEG2000Lossless, JPEG2000
    if pydicom.__version__ != "3.0.2":
        raise RuntimeError("pydicom 3.0.2 requis ; relancer l'installation hors ligne.")
    missing = []
    for syntax in [JPEGLossless, JPEGLosslessSV1, JPEG2000Lossless, JPEG2000]:
        decoder = get_decoder(syntax)
        if not decoder.is_available:
            missing.append(f"{syntax.name}: {decoder.missing_dependencies}")
    if missing:
        raise ImportError("Décodeurs DICOM absents : redémarrer la session et réinstaller les wheels. "
                          + "; ".join(missing))


def dicom_frames(path, uid, series_uid):
    """Chaque résultat = (SOP UID, indice de frame, nombre de frames, pixels bruts)."""
    import pydicom
    from pydicom.pixels import get_decoder
    ds = pydicom.dcmread(path)
    for tag, expected in [(UID, uid), (SERIES, series_uid)]:
        if str(getattr(ds, tag, expected)) != expected:
            raise ValueError(f"DICOM rangé dans la mauvaise étude/série : {path}")
    if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
        raise ValueError("DICOM couleur hors du contrat d'entraînement en niveaux de gris.")
    syntax = ds.file_meta.TransferSyntaxUID
    try:
        decoder = get_decoder(syntax)
    except NotImplementedError as exc:
        raise ImportError(f"Syntaxe DICOM non prise en charge : {syntax}") from exc
    if not decoder.is_available:
        raise ImportError(f"Décodeur manquant pour {syntax}: {decoder.missing_dependencies}")
    values = ds.pixel_array
    count = int(getattr(ds, "NumberOfFrames", 1))
    if values.ndim == 2 and count == 1:
        values = values[None]
    if values.ndim != 3 or values.shape[0] != count:
        raise ValueError(f"Forme DICOM non prise en charge : {values.shape}")
    sop = str(getattr(ds, "SOPInstanceUID", Path(path).stem))
    for index, frame in enumerate(values):
        yield sop, index, count, frame


def series_candidates(directory, uid, series_uid, input_format, errors, seed=42, limit=8):
    """Compte tout, mais ne garde en RAM que les huit meilleurs hash de la série."""
    suffix = "*.dcm" if input_format == "dicom" else "*.jpeg"
    files = sorted(directory.glob(suffix))
    picked, seen, readable = [], set(), 0
    for path in files:
        try:
            if input_format == "jpeg":
                with Image.open(path) as image:
                    image.load()
                    if image.mode != "L" or image.size != (224, 224):
                        raise ValueError("JPEG local attendu : L 224x224.")
                    frames = [(path.stem, 0, 1, np.array(image, copy=True))]
            else:
                frames = dicom_frames(path, uid, series_uid)
            for sop, index, count, frame in frames:
                if frame.ndim != 2 or not frame.size or not np.isfinite(frame).all():
                    raise ValueError("Pixels DICOM invalides.")
                # Cas normal : même chemin logique .jpeg que pendant l'entraînement.
                # Multi-frame : suffixe déterministe, sans prétendre connaître le nom
                # qu'aurait utilisé le convertisseur JPEG original indisponible.
                name = sop + (f".frame{index:06d}" if count > 1 else "") + ".jpeg"
                logical = f"test_series/{uid}/{series_uid}/{name}"
                if logical in seen:
                    continue
                seen.add(logical)
                readable += 1
                key = fingerprint([seed, uid, series_uid, logical])
                if len(picked) < limit or key < picked[-1]["key"]:
                    picked.append({"key": key, "logical_path": logical, "source": str(path),
                                   "pixels": np.array(frame, copy=True)})
                    picked.sort(key=lambda x: x["key"])
                    picked = picked[:limit]
        except ImportError:
            raise  # Une dépendance manquante ne doit jamais devenir une image "corrompue".
        except Exception as exc:
            errors.append({UID: uid, SERIES: series_uid, "file": str(path),
                           "error": f"{type(exc).__name__}: {exc}"})
    return readable, picked


def select_study(root, uid, series_rows, input_format, errors):
    slots, diagnostics = {}, []
    for row in series_rows.sort_values(SERIES).itertuples(index=False):
        series_uid = getattr(row, SERIES)
        slot = slot_of(row.Anatomical_Plane, row.Fat_Suppression)
        directory = (Path(root) / "test_series" / uid / series_uid).resolve()
        if not directory.is_relative_to(Path(root).resolve()):
            raise ValueError("Chemin de série hors du dossier compétition.")
        count, picked = series_candidates(directory, uid, series_uid, input_format, errors)
        diagnostics.append({UID: uid, SERIES: series_uid, "slot": slot, "readable_images": count})
        old = slots.get(slot)
        if picked and (old is None or (-count, series_uid) < (-old[0], old[1])):
            slots[slot] = (count, series_uid, picked)
    if not slots:
        raise RuntimeError(f"Aucune image utilisable pour {uid}. Consulter decode_errors.csv et les entrées.")
    images, slot_ids, token_ids, selected = [], [], [], []
    for slot, (_, series_uid, picked) in sorted(slots.items()):
        for token, item in enumerate(picked):
            pixels = (pixel_array_to_jpeg_pixels(item["pixels"]) if input_format == "dicom"
                      else item["pixels"])
            images.append(pixels); slot_ids.append(slot); token_ids.append(token)
            selected.append({UID: uid, SERIES: series_uid, "slot": slot, "token": token,
                             "logical_path": item["logical_path"]})
    return {"pixels": np.stack(images), "slots": np.array(slot_ids, np.int8),
            "tokens": np.array(token_ids, np.int8)}, diagnostics, selected


def prepare_test(root, cache_dir, output_dir, input_format="dicom"):
    """Un cache temporaire de petites images sélectionnées, partagé par les lecteurs."""
    if input_format not in {"dicom", "jpeg"}:
        raise ValueError("input_format attendu : dicom, ou jpeg pour les tests locaux.")
    root, cache_dir, output_dir = Path(root), Path(cache_dir), Path(output_dir)
    ids, series = read_inputs(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths, diagnostics, selected, errors = [], [], [], []
    grouped = {uid: rows for uid, rows in series.groupby(UID, sort=False)}
    try:
        for i, uid in enumerate(ids):
            arrays, counts, rows = select_study(root, uid, grouped[uid], input_format, errors)
            path = cache_dir / (fingerprint(uid) + ".npz")
            # pixels [N<=48,224,224] uint8 ; slots/tokens [N]. Les slots absents
            # seront masqués au moment de créer [6,8,D], sans duplication d'image.
            np.savez_compressed(path, **arrays)
            paths.append(path); diagnostics.extend(counts); selected.extend(rows)
            if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(ids):
                print(f"Préparation test : {i + 1}/{len(ids)} études", flush=True)
    finally:
        write_csv(output_dir / "decode_errors.csv", pd.DataFrame(
            errors, columns=[UID, SERIES, "file", "error"]))
    write_csv(output_dir / "series_diagnostics.csv", pd.DataFrame(diagnostics))
    write_csv(output_dir / "selected_images.csv", pd.DataFrame(selected))
    return ids, paths, {"selected_images": len(selected), "decode_errors": len(errors)}
