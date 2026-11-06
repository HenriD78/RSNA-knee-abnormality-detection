from __future__ import annotations

import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from .cache import Cache, prepare
from .common import (LABELS, ROOT, SPECS, UID, atomic_csv, atomic_json, digest,
                     input_tables, read_json, require, sha256, source_hash, validate_predictions, versions)
from .evaluate import blend, collect_model, ranks
from .model import StudyModel
from .train import predict_model


def export(project, output, destination):
    """Inference-only package: complete encoder+head, no optimizer, no train cache."""
    table, _ = input_tables(project)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    require(not (destination / "manifest.json").exists(), "Export déjà présent ; employer un nouveau dossier")
    files, configs = {}, {}
    cache_contract = None
    for model_id in SPECS:
        collect_model(output, model_id, table)  # Reject partial OOF/folds before exporting any model.
    for model_id in SPECS:
        for fold in range(5):
            folder = Path(output) / model_id / f"fold{fold}"
            meta = read_json(folder / "completed.json")
            cfg = meta["config"]
            require(meta["contract"]["source_hash"] == source_hash() and meta["contract"]["versions"] == versions(),
                    "Export avec un autre code/environnement que l'entraînement")
            if cache_contract is None:
                cache_contract = cfg["cache_manifest_sha256"]
            require(cache_contract == cfg["cache_manifest_sha256"], "Caches différents entre lecteurs")
            configs[model_id] = cfg
            relative = f"models/{model_id}/final_fold{fold}.safetensors"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / f"final_fold{fold}.safetensors", target)
            files[relative] = sha256(target)
    shutil.copytree(ROOT / "rsna3", destination / "rsna3", ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    source_cache = next(iter(configs.values()))["profile"]["cache_manifest_sha256"]
    manifest = {"format": "rsna3-full-models-v1", "labels": LABELS, "configs": configs,
                "files": files, "versions": versions(), "source_hash": source_hash(),
                "training_cache_manifest_sha256": source_cache,
                "render": next(iter(configs.values()))["profile"]["config"]["input"]["cache_resolution"],
                "centres": None}
    # Cache cardinality is stored by the pilot (not inferred from model weights).
    centres = [c["profile"]["cache_centres"] for c in configs.values()]
    require(len(set(centres)) == 1 and centres[0] in (2, 4), "Nombre de centres de cache incompatible")
    manifest["centres"] = centres[0]
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def load_bundle(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    require(manifest["format"] == "rsna3-full-models-v1" and manifest["labels"] == LABELS, "Paquet d'inférence incompatible")
    require(manifest["source_hash"] == source_hash(), "Code d'inférence différent du code exporté")
    require(manifest["versions"] == versions(), "Versions différentes de l'entraînement ; utiliser l'environnement exporté")
    expected = {f"models/{m}/final_fold{k}.safetensors" for m in SPECS for k in range(5)}
    require(set(manifest["files"]) == expected and set(manifest["configs"]) == set(SPECS), "Quinze poids complets requis")
    for name, checksum in manifest["files"].items():
        require(sha256(bundle / name) == checksum, f"Poids corrompu : {name}")
    return manifest


def benchmark(bundle, studies_csv, series_csv, dicom_root, work, n=50, quota_factor=1., max_hours=2):
    """Full decode+render+15-model timing on an unlabelled representative sample."""
    require(50 <= n <= 100 and quota_factor > 0, "50–100 études et facteur de quota mesuré requis")
    manifest = load_bundle(bundle)
    table = pd.read_csv(studies_csv, dtype={UID: str})
    require(table[UID].is_unique and len(table) >= n, "Échantillon insuffisant")
    ids = sorted(table[UID], key=lambda uid: digest(uid))[:n]
    sample = pd.DataFrame({UID: ids})
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "benchmark-cache"
    start = time.monotonic()
    deadline = start + max_hours * 3600
    prepare(ids, series_csv, dicom_root, cache, centres=manifest["centres"], inference=True, max_hours=max_hours)
    predict_bundle(bundle, sample, cache, work, deadline)
    elapsed = time.monotonic() - start
    startup = sum(read_json(work / f"profile_{model}_fold{fold}.json")["load_seconds"] for model in SPECS for fold in range(5))
    admission = {"studies": n, "uids_hash": digest(ids), "startup_seconds": startup,
                 "seconds_per_study": max(0., elapsed - startup) / n, "quota_factor": quota_factor,
                 "elapsed_seconds": elapsed, "scope": "decode_render_all_15_models_global_ranks",
                 "series_sha256": sha256(series_csv)}
    manifest["admission_profile"] = admission
    atomic_json(Path(bundle) / "manifest.json", manifest)
    return admission


def run_public_recipe(recipe_path, uids, work, seconds, test_csv=None, series_csv=None, dicom_root=None):
    """Optional exact public worker, supplied as argv with its own offline env.

    The runner never invents Raptor/R6 assets or forces our timm onto that worker.
    The manifest must pin code/assets hashes and provide an isolated interpreter.
    """
    recipe = read_json(recipe_path)
    base = Path(recipe_path).resolve().parent
    require(recipe.get("kind") == "verified_external_submission_recipe", "Recette de repli non vérifiée")
    require(recipe.get("files") and isinstance(recipe.get("argv"), list) and recipe["argv"], "Manifest de repli incomplet")
    for name, checksum in recipe["files"].items():
        require(sha256(base / name) == checksum, f"Asset public corrompu : {name}")
    csv = Path(work) / "public_base.csv"
    csv.unlink(missing_ok=True)
    substitutions = {"recipe_root": base, "output_csv": csv, "test_csv": test_csv,
                     "series_csv": series_csv, "dicom_root": dicom_root}
    argv = list(map(str, recipe["argv"]))
    for key, value in substitutions.items():
        argv = [v.replace("{" + key + "}", str(value)) for v in argv]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(recipe.get("env", {}))
    with (Path(work) / "public_worker.log").open("a") as log:
        process = subprocess.Popen(argv, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=max(1., seconds))
            require(code == 0, f"Worker public en échec : {code}")
        finally:
            # Include children of an isolated worker, even if its parent already exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    prediction = validate_predictions(pd.read_csv(csv, dtype={UID: str}), uids)
    return prediction, {"recipe_sha256": sha256(recipe_path), "output_sha256": sha256(csv), "returncode": code}


def predict_bundle(bundle, table, cache, work, deadline, workers=1, device="cuda"):
    manifest = load_bundle(bundle)
    device = torch.device(device)
    require(device.type == "cuda" and torch.cuda.is_available(), "CUDA requis pour le profil d'inférence de compétition")
    torch.set_num_threads(1)
    reader = Cache(cache, verify=True)
    require(set(reader.entries) == set(table[UID]), "Cache de test incomplet")
    require(reader.manifest["contract"]["centres"] == manifest["centres"], "Centres du cache test différents du cache train")
    reader.close()
    values = {}
    sample_n = min(16, len(table))
    for model_id, cfg in manifest["configs"].items():
        total = np.zeros((len(table), len(LABELS)), np.float64)
        for fold in range(5):
            require(time.monotonic() < deadline, "Échéance interne d'inférence atteinte")
            start = time.monotonic()
            model = StudyModel(cfg).to(device)
            model.load_state_dict(load_file(str(Path(bundle) / f"models/{model_id}/final_fold{fold}.safetensors")), strict=True)
            load_seconds = time.monotonic() - start
            # Time the real cache + all windows on a deterministic cohort.
            # Reuse these predictions rather than running the sample a second time.
            sample = table.iloc[:sample_n]
            torch.cuda.synchronize(device)
            start = time.monotonic()
            first = predict_model(model, sample, cache, cfg, fold, device, workers, deadline=deadline)
            torch.cuda.synchronize(device)
            per_study = (time.monotonic() - start) / sample_n
            remaining = len(table) - sample_n
            # Family-specific startup costs are recorded separately from throughput.
            estimated = remaining * per_study * 1.2
            require(time.monotonic() + estimated < deadline,
                    f"Temps projeté trop long pour {model_id}/fold{fold}: {estimated/3600:.2f} h restantes")
            if remaining:
                rest = predict_model(model, table.iloc[sample_n:], cache, cfg, fold, device, workers, deadline=deadline)
                predicted = pd.concat([first, rest], ignore_index=True)
            else:
                predicted = first
            aligned = validate_predictions(predicted, table[UID].tolist())
            total += aligned[LABELS].to_numpy(float) / 5
            atomic_json(Path(work) / f"profile_{model_id}_fold{fold}.json",
                        {"load_seconds": load_seconds, "seconds_per_study": per_study, "studies": len(table)})
            del model
            torch.cuda.empty_cache()
        values[model_id] = total  # Mean probabilities over folds, before global ranks.
        atomic_csv(Path(work) / f"test_{model_id}.csv", pd.DataFrame({UID: table[UID], **dict(zip(LABELS, total.T))}))
    return blend(values)


def submit(bundle, test_csv, series_csv, dicom_root, work, output_csv,
           max_hours=8, public_recipe=None, candidate_3pct=False, workers=1, optional_quota_hours=None):
    work, output_csv = Path(work), Path(output_csv)
    work.mkdir(parents=True, exist_ok=True)
    # A failed run must never expose a previous version's submission.csv.
    output_csv.unlink(missing_ok=True)
    require(not candidate_3pct or public_recipe is not None, "Le candidat 97/3 exige un socle public complet")
    require(0 < max_hours <= 8.25, "Garder au moins 45 minutes sur les neuf heures officielles")
    start = time.monotonic()
    deadline = start + max_hours * 3600
    branch_deadline = deadline - 120
    table = pd.read_csv(test_csv, dtype={UID: str})
    require(UID in table and table[UID].is_unique and len(table) > 0, "test.csv invalide")
    uids = table[UID].tolist()
    base, provenance = None, {}
    if public_recipe:
        base, provenance["public_base"] = run_public_recipe(public_recipe, uids, work, branch_deadline - time.monotonic(),
                                                           test_csv, series_csv, dicom_root)
        base = validate_predictions(base, uids)
    try:
        manifest = load_bundle(bundle)
        admission = manifest.get("admission_profile")
        require(admission is not None, "Effectuer benchmark-inference sur 50–100 études avant la soumission")
        estimate = admission["startup_seconds"] + len(table) * admission["seconds_per_study"] * 1.2
        require(time.monotonic() + estimate < branch_deadline, "Pipeline complet hors délai projeté")
        if base is not None:
            # Admit an optional branch only against an actual measured full-pipeline profile.
            require(optional_quota_hours is not None and estimate / 3600 * admission["quota_factor"] <= optional_quota_hours,
                    "Branche optionnelle hors quota mesuré")
        cache = work / ("test-cache-" + digest({"uids": uids, "series": sha256(series_csv), "source": source_hash(),
                                              "centres": manifest["centres"]})[:12])
        if not (cache / "manifest.json").exists():
            prepare(uids, series_csv, dicom_root, cache, centres=manifest["centres"], inference=True,
                    max_hours=max(.001, (branch_deadline - time.monotonic()) / 3600))
        predicted = predict_bundle(bundle, table, cache, work, branch_deadline, workers)
        if base is not None and candidate_3pct:
            predicted = ranks(.97 * ranks(base[LABELS].to_numpy(float)) + .03 * ranks(predicted))
            provenance["recipe"] = "rank_97pct_public_3pct_clean_ensemble_experimental"
        elif base is not None:
            # Public base remains a separate submission unless the preregistered candidate is requested.
            predicted = base[LABELS].to_numpy(float)
            provenance["recipe"] = "public_base_separate_clean_predictions_saved"
        else:
            provenance["recipe"] = "clean_equal_rank_three_families_five_fold_probability_means"
    except Exception as e:
        if base is None:
            raise
        predicted = base[LABELS].to_numpy(float)
        provenance.update({"recipe": "public_base_global_fallback", "optional_branch_error": f"{type(e).__name__}: {e}"})
    final = pd.DataFrame({UID: uids, **dict(zip(LABELS, predicted.T))})
    final = validate_predictions(final, uids)[[UID] + LABELS]
    require(time.monotonic() < deadline, "Échéance interne atteinte avant écriture")
    atomic_csv(output_csv, final)
    provenance.update({"studies": len(uids), "test_csv_sha256": sha256(test_csv),
                       "submission_sha256": sha256(output_csv), "elapsed_seconds": time.monotonic() - start})
    atomic_json(work / "submission_receipt.json", provenance)
    return provenance
