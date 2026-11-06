"""Point d'entrée : DICOM du scoring → prédictions → /kaggle/working/submission.csv."""

import os
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import pandas as pd
import torch

from .blend import build_blends, validate_submission
from .constants import READERS, UID
from .data import check_codecs, prepare_test
from .io import read_json, sha256, write_csv, write_json
from .predict import predict_reader


def run_inference(bundle, competition, output="/kaggle/working", device="cuda",
                  input_format="dicom", work_dir="/kaggle/temp/rsna_submission"):
    """Lance uniquement l'inférence. Le mode jpeg/cpu sert aux tests locaux limités."""
    started = time.perf_counter()
    bundle, competition, output, work_dir = map(Path, [bundle, competition, output, work_dir])
    output.mkdir(parents=True, exist_ok=True)
    # Un ancien CSV ne doit pas être publié après l'échec d'une nouvelle exécution.
    (output / "submission.csv").unlink(missing_ok=True)
    (output / "inference_report.json").unlink(missing_ok=True)
    if Path("/kaggle/input").exists() and (device != "cuda" or input_format != "dicom"):
        raise ValueError("Sur Kaggle : GPU CUDA et DICOM officiels obligatoires pour ce notebook.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Activer le GPU T4 dans les paramètres Kaggle.")
    if input_format == "dicom":
        check_codecs()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read_json(bundle / "submission_bundle.json")
    if manifest["primary_blend"] != "candidate05":
        raise ValueError("Le paquet de référence doit conserver le candidat principal prédéfini.")
    work_dir.mkdir(parents=True, exist_ok=True)
    # Cache NOUVEAU à chaque run : il ne peut pas conserver les trois exemples
    # lorsque Kaggle remplace test.csv et test_series/ par les données cachées.
    with tempfile.TemporaryDirectory(prefix="pixels_", dir=work_dir) as temporary:
        ids, caches, preparation = prepare_test(competition, temporary, output, input_format)
        print(f"Inférence sur les {len(ids)} études du test courant.", flush=True)
        predictions, timings = {}, {}
        for reader in READERS:
            frame, timings[reader] = predict_reader(bundle, manifest, reader, ids, caches, device)
            validate_submission(frame, ids)
            predictions[reader] = frame
            write_csv(output / "reader_predictions" / f"{reader}.csv", frame)
        blends = build_blends(predictions)
    for name, frame in blends.items():
        validate_submission(frame, ids)
        write_csv(output / "blend_predictions" / f"submission_{name}.csv", frame)
    primary = blends[manifest["primary_blend"]]
    validate_submission(primary, ids)
    report = {"format": "rsna-submission-report-v1", "bundle_id": manifest["bundle_id"],
              "blend": manifest["primary_blend"], "studies": len(ids),
              "test_csv_sha256": sha256(competition / "test.csv"),
              "test_series_csv_sha256": sha256(competition / "test_series.csv"),
              "input_format": input_format, "device": device, "head_count": 20,
              **preparation, "reader_timings": timings,
              "total_seconds": round(time.perf_counter() - started, 3),
              "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "timm": version("timm"), "pydicom": version("pydicom"),
                           "opencv": cv2.__version__},
              "note": "Inférence sur le test courant ; aucun calcul de score sans labels test."}
    # Publication seulement après validation des quatre lecteurs et du mélange.
    write_json(output / "inference_report.json", report)
    write_csv(output / "submission.csv", primary)
    print(f"PRÊT : {output / 'submission.csv'} — {len(primary)} lignes, douze findings.", flush=True)
    return primary, report
