"""Parallel DICOM preparation around the unchanged, pinned RSNA3 renderer.

This module is outside rsna3: training code, rendering and model hashes stay
unchanged. Its own checksum and worker count are pinned in the new benchmark.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

PIPELINE = "parallel-study-cache-v1"
SCOPE = "decode_render_parallel_cache_all_15_models_global_ranks"


def execution_contract(workers, batch_size):
    from rsna3.common import require, sha256
    require(isinstance(workers, int) and 1 <= workers <= 8, "Use one to eight CPU processes")
    require(isinstance(batch_size, int) and 1 <= batch_size <= 8, "Use batches of one to eight studies")
    return {"pipeline": PIPELINE, "workers": workers, "batch_size": batch_size,
            "helper_sha256": sha256(Path(__file__))}


def _prepare_batch(task):
    """Each spawned process calls the original renderer and original cache writer."""
    from rsna3.cache import prepare
    from rsna3.common import require
    ids, series_csv, dicom_root, destination, centres, deadline = task
    require(time.monotonic() < deadline, "Parallel preprocessing deadline reached")
    manifest = prepare(ids, series_csv, dicom_root, destination, centres=centres,
                       inference=True, max_hours=max(.001, (deadline - time.monotonic()) / 3600))
    require(manifest["complete"], "Incomplete worker cache")
    return str(Path(destination) / "manifest.json")


def _merge_worker_caches(ids, manifests, output, series_csv, centres, started,
                         execution, limit_bytes):
    from rsna3.cache import Cache, STORAGE
    from rsna3.common import atomic_json, digest, read_json, require, sha256, source_hash
    from rsna3.dicom import RENDER
    contract = {"render": RENDER, "storage": STORAGE, "centres": centres,
                "series_sha256": sha256(series_csv), "studies_hash": digest(sorted(ids)),
                "source_hash": source_hash(), "fixture": False, "purpose": "inference"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries, shards, copied_bytes = {}, [], 0
    for marker in sorted(map(Path, manifests)):
        worker = read_json(marker)
        worker_contract = worker["contract"]
        require(worker.get("complete"), "Worker cache is incomplete")
        for key, expected in contract.items():
            if key != "studies_hash":
                require(worker_contract.get(key) == expected, f"Worker rendering contract differs: {key}")
        require(set(worker["entries"]).issubset(ids), "Unexpected study in a worker cache")
        require(worker_contract["studies_hash"] == digest(sorted(worker["entries"])), "Worker study contract differs")
        require(not set(entries).intersection(worker["entries"]), "Duplicate worker studies")
        renamed = {}
        for original in worker["shards"]:
            old_name = original["file"]
            source = marker.parent / old_name
            require(sha256(source) == original["sha256"], "Worker shard is corrupt")
            require(source.stat().st_size == original["bytes"], "Worker shard size differs")
            new_name = f"shard-{len(shards):04d}.zip"
            copied_bytes += original["bytes"]
            require(copied_bytes + (4 << 20) < limit_bytes, "Combined cache exceeds its storage limit")
            require(shutil.disk_usage(output).free > original["bytes"] + (256 << 20), "Insufficient merge disk space")
            shutil.copy2(source, output / new_name)
            renamed[old_name] = new_name
            shards.append({**original, "file": new_name})
        for uid, entry in worker["entries"].items():
            entries[uid] = {**entry, "shard": renamed[entry["shard"]]}
    require(set(entries) == set(ids), "Merged cache has incomplete study coverage")
    for index, shard in enumerate(shards):
        atomic_json(output / f"shard-{index:04d}.json", {
            **shard, "contract": contract,
            "entries": {uid: e for uid, e in entries.items() if e["shard"] == shard["file"]}})
    manifest = {"contract": contract, "entries": entries, "shards": shards, "complete": True,
                "bytes": copied_bytes, "session_seconds": time.monotonic() - started,
                "execution": execution,
                "laterality_conflict_studies": [uid for uid in ids if entries[uid].get("warnings", 0)]}
    index_bytes = sum(p.stat().st_size for p in output.glob("shard-*.json"))
    require(copied_bytes + index_bytes + (4 << 20) < limit_bytes, "Combined cache indexes exceed the storage limit")
    atomic_json(output / "manifest.json", manifest)
    reader = Cache(output, verify=True)
    reader.close()
    return manifest


def prepare_parallel(ids, series_csv, dicom_root, output, centres, workers=4,
                     batch_size=4, max_hours=2., limit_bytes=12 << 30):
    from rsna3.cache import Cache, STORAGE
    from rsna3.common import digest, read_json, require, sha256, source_hash
    from rsna3.dicom import RENDER
    ids = list(map(str, ids))
    require(ids and len(ids) == len(set(ids)), "Study IDs must be nonempty and unique")
    require(centres in (2, 4) and max_hours > 0, "Invalid rendering settings")
    execution = execution_contract(workers, batch_size)
    output, started = Path(output), time.monotonic()
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        expected = {"render": RENDER, "storage": STORAGE, "centres": centres,
                    "series_sha256": sha256(series_csv), "studies_hash": digest(sorted(ids)),
                    "source_hash": source_hash(), "fixture": False, "purpose": "inference"}
        require(manifest["contract"] == expected and manifest.get("execution") == execution,
                "Existing parallel cache belongs to different inputs or execution settings")
        require(set(manifest["entries"]) == set(ids), "Existing parallel cache has incorrect coverage")
        reader = Cache(output, verify=True)
        reader.close()
        return manifest
    deadline = started + max_hours * 3600
    parts = output.with_name(output.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    tasks = [(batch, str(series_csv), str(dicom_root), str(parts / digest(batch)[:24]), centres, deadline)
             for batch in batches]
    # Bound native BLAS/OpenMP thread pools in the spawned processes.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    pool = mp.get_context("spawn").Pool(processes=min(workers, len(tasks)))
    manifests, done = [], 0
    try:
        iterator = pool.imap_unordered(_prepare_batch, tasks, chunksize=1)
        for index in range(len(tasks)):
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Parallel preprocessing deadline reached")
            try:
                marker = iterator.next(timeout=remaining)
            except mp.TimeoutError as error:
                raise ValueError("Parallel preprocessing deadline reached") from error
            manifests.append(marker)
            done += len(read_json(marker)["entries"])
            print(f"Parallel cache: {done}/{len(ids)} studies ({index + 1}/{len(tasks)} batches)", flush=True)
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    require(time.monotonic() < deadline, "Parallel preprocessing deadline reached before merge")
    manifest = _merge_worker_caches(ids, manifests, output, series_csv, centres, started, execution, limit_bytes)
    shutil.rmtree(parts)
    return manifest


def benchmark_parallel(bundle, studies_csv, series_csv, dicom_root, work,
                       workers=4, batch_size=4, quota_factor=1., n=50, max_hours=2.):
    import pandas as pd
    from rsna3.common import UID, atomic_json, digest, read_json, require, sha256
    from rsna3.inference import load_bundle, predict_bundle
    require(50 <= n <= 100 and quota_factor > 0, "Use fifty to one hundred studies and the measured quota factor")
    manifest = load_bundle(bundle)
    execution = execution_contract(workers, batch_size)
    table = pd.read_csv(studies_csv, dtype={UID: str})
    require(table[UID].is_unique and len(table) >= n, "Insufficient unique sample studies")
    ids = sorted(table[UID], key=digest)[:n]
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cache, started = work / "parallel-cache", time.monotonic()
    require(not cache.exists() and not cache.with_name(cache.name + ".parts").exists(),
            "A timing benchmark requires a fresh cache directory; use a new saved run")
    deadline = started + max_hours * 3600
    prepare_parallel(ids, series_csv, dicom_root, cache, manifest["centres"], workers, batch_size, max_hours)
    cache_seconds = time.monotonic() - started
    predict_bundle(bundle, pd.DataFrame({UID: ids}), cache, work, deadline)
    elapsed = time.monotonic() - started
    startup = sum(read_json(work / f"profile_{model}_fold{fold}.json")["load_seconds"]
                  for model in manifest["configs"] for fold in range(5))
    profile = {"studies": n, "uids_hash": digest(ids), "startup_seconds": startup,
               "seconds_per_study": max(0., elapsed - startup) / n,
               "quota_factor": quota_factor, "elapsed_seconds": elapsed,
               "preprocessing_seconds": cache_seconds, "scope": SCOPE,
               "execution": execution, "series_sha256": sha256(series_csv),
               "cache_reused": False,
               "coldness_note": "Mounted data and OS caches are not forcibly evicted; compare on the same study sample."}
    # Retain the original serial measurement for auditing. Never admit the
    # parallel estimate to the serial submit function.
    if manifest.get("admission_profile", {}).get("scope") != SCOPE:
        manifest["serial_admission_profile"] = manifest.get("admission_profile")
    manifest["required_execution_pipeline"] = PIPELINE
    manifest["admission_profile"] = profile
    atomic_json(Path(bundle) / "manifest.json", manifest)
    return profile


def submit_parallel(bundle, test_csv, series_csv, dicom_root, work, output_csv,
                    max_hours=8., workers=4, batch_size=4, loader_workers=1):
    import pandas as pd
    from rsna3.common import LABELS, UID, atomic_csv, atomic_json, digest, require, sha256, source_hash, validate_predictions
    from rsna3.inference import load_bundle, predict_bundle
    output_csv, work = Path(output_csv), Path(work)
    output_csv.unlink(missing_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    require(0 < max_hours <= 8.25, "Keep time to write the CSV before the official runtime limit")
    started = time.monotonic()
    deadline = started + max_hours * 3600
    branch_deadline = deadline - 120
    table = pd.read_csv(test_csv, dtype={UID: str})
    require(UID in table and len(table) and table[UID].is_unique, "Invalid test.csv")
    manifest = load_bundle(bundle)
    profile = manifest.get("admission_profile", {})
    execution = execution_contract(workers, batch_size)
    require(manifest.get("required_execution_pipeline") == PIPELINE and profile.get("scope") == SCOPE,
            "Measure this parallel pipeline on fifty studies before submitting")
    require(profile.get("execution") == execution, "Worker count, batch size or parallel helper differs from benchmark")
    estimate = profile["startup_seconds"] + len(table) * profile["seconds_per_study"] * 1.2
    require(time.monotonic() + estimate < branch_deadline,
            f"Parallel pipeline exceeds its measured runtime budget: {estimate / 3600:.2f} hours")
    ids = table[UID].tolist()
    cache = work / ("parallel-test-cache-" + digest({"uids": ids, "series": sha256(series_csv),
                    "source": source_hash(), "centres": manifest["centres"], "execution": execution})[:12])
    prepare_parallel(ids, series_csv, dicom_root, cache, manifest["centres"], workers, batch_size,
                     max_hours=max(.001, (branch_deadline - time.monotonic()) / 3600))
    predicted = predict_bundle(bundle, table, cache, work, branch_deadline, loader_workers)
    final = validate_predictions(pd.DataFrame({UID: ids, **dict(zip(LABELS, predicted.T))}), ids)[[UID] + LABELS]
    require(time.monotonic() < deadline, "Inference deadline reached before CSV creation")
    atomic_csv(output_csv, final)
    receipt = {"recipe": "clean_equal_rank_three_families_five_fold_probability_means",
               "execution": execution, "studies": len(ids), "test_csv_sha256": sha256(test_csv),
               "submission_sha256": sha256(output_csv), "elapsed_seconds": time.monotonic() - started,
               "projected_seconds_with_margin": estimate}
    atomic_json(work / "submission_receipt.json", receipt)
    return receipt
