"""Construit les archives Kaggle sans lancer d'encodeur ni d'entraînement.

Ce fichier est un outil de conditionnement, pas un générateur de notebook.
Les entrées locales restent en lecture seule. Étapes : data, puis finish.
"""

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
PROJECT = LOCAL.parent
DIST = HERE / "dist"
READERS = {
    "coatnet": {"model": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "size": 384},
    "small": {"model": "convnext_small.fb_in22k_ft_in1k", "size": 224},
    "tiny": {"model": "convnext_tiny.fb_in22k_ft_in1k", "size": 224},
    "dino": {"model": "vit_small_patch14_dinov2.lvd142m", "size": 224},
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(payload), 0, 0o644
    archive.addfile(info, io.BytesIO(payload))


def build_data():
    DIST.mkdir(parents=True, exist_ok=True)
    prepared = LOCAL / "artifacts/full/prepared"
    origin = read(prepared / "manifest.json")
    if fp({k: v for k, v in origin.items() if k != "signature"}) != origin["signature"]:
        raise ValueError("Manifeste local incohérent.")
    source = (LOCAL / origin["config"]["data_dir"]).resolve()
    for name, expected in origin["source_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"CSV source modifié : {name}")
    for name, expected in origin["table_hashes"].items():
        if digest(prepared / name) != expected:
            raise ValueError(f"Table locale modifiée : {name}")
    selected = pd.read_csv(prepared / "selected_images.csv")
    original_columns = list(selected.columns)
    hashes, paths, seen = [], [], set()
    temporary = DIST / "data.tar.partial"
    with tarfile.open(temporary, "w") as archive:
        for i, row in enumerate(selected.itertuples(index=False)):
            path = (source / row.path).resolve()
            if not path.is_relative_to(source):
                raise ValueError("Chemin d'image hors du data_dir.")
            stat = path.stat()
            if stat.st_size != row.size_bytes or stat.st_mtime_ns != row.mtime_ns:
                raise ValueError(f"Image modifiée : {row.path}")
            payload = path.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            # Noms courts : décompression plus rapide et aucune géométrie inventée.
            name = "images/" + fp(row.path) + ".jpeg"
            if name not in seen:
                add_bytes(archive, name, payload)
                seen.add(name)
            hashes.append(sha); paths.append(name)
            if i % 10000 == 0:
                print(f"Images conditionnées : {i:,}/{len(selected):,}", flush=True)
        selected["source_path"] = selected["path"]
        selected["path"] = paths
        selected["sha256"] = hashes
        # La date locale n'est pas un identifiant de contenu portable.
        selected = selected.drop(columns="mtime_ns")
        tables = {}
        for name in origin["table_hashes"]:
            payload = (selected.to_csv(index=False).encode() if name == "selected_images.csv"
                       else (prepared / name).read_bytes())
            add_bytes(archive, "prepared/" + name, payload)
            tables[name] = hashlib.sha256(payload).hexdigest()
        manifest = {
            "format": "rsna-portable-v1", "origin_signature": origin["signature"],
            "table_hashes": tables, "active_studies": origin["active_studies"],
            "gold_studies": origin["gold_studies"], "selected_images": len(selected),
            "split_contract": {k: origin["config"][k] for k in ["seed", "n_folds", "samples_per_slot"]},
            "source_hashes": origin["source_hashes"],
        }
        manifest["signature"] = fp(manifest)
        add_bytes(archive, "prepared/manifest.json", json.dumps(manifest, indent=2).encode())
    temporary.replace(DIST / "data.tar")
    save(DIST / "prepared_manifest.json", manifest)
    print("Archive des données prête.", flush=True)

    with tarfile.open(DIST / "assets.tar.partial", "w") as archive:
        asset_records = {}
        for reader in READERS:
            directory = LOCAL / "assets" / reader
            meta = read(directory / "manifest.json")
            if digest(directory / meta["weight_file"]) != meta["weight_sha256"]:
                raise ValueError(f"Poids local modifié : {reader}")
            for name in ["manifest.json", meta["weight_file"]]:
                archive.add(directory / name, arcname=reader + "/" + name, recursive=False)
            asset_records[reader] = {"weight_sha256": meta["weight_sha256"], "revision": meta["revision"]}
    (DIST / "assets.tar.partial").replace(DIST / "assets.tar")

    seeds = {}
    with tarfile.open(DIST / "seed_features.tar.partial", "w") as archive:
        for reader in READERS:
            directory = LOCAL / "artifacts/full/features" / reader
            if not (directory / "train/manifest.json").exists():
                continue
            encoder, cache = read(directory / "encoder.json"), read(directory / "train/manifest.json")
            expected = fp({"prepared": origin["signature"], "reader": reader,
                           "model": READERS[reader], "details": encoder})
            if cache["identity"] != expected:
                raise ValueError(f"Cache local incompatible : {reader}")
            if encoder["asset"]["weight_sha256"] != asset_records[reader]["weight_sha256"]:
                raise ValueError("Poids/cache différents.")
            records = {}
            allowed = set(selected.loc[selected.split == "train", "StudyInstanceUID"])
            for uid, record in cache["studies"].items():
                path = directory / "train" / record["file"]
                if uid not in allowed or digest(path) != record["sha256"]:
                    raise ValueError(f"Cache local corrompu : {reader}/{uid}")
                archive.add(path, arcname=reader + "/train/" + record["file"], recursive=False)
                records[uid] = record
            seeds[reader] = {"encoder": encoder, "studies": records,
                             "origin_cache_identity": cache["identity"]}
        add_bytes(archive, "manifest.json", json.dumps({"prepared": manifest["signature"],
                                                        "readers": seeds}, indent=2).encode())
    (DIST / "seed_features.tar.partial").replace(DIST / "seed_features.tar")
    save(DIST / "data_build.json", {"prepared": manifest, "assets": asset_records,
                                  "seed_studies": {r: len(s["studies"]) for r, s in seeds.items()},
                                  "original_selected_columns": original_columns})
    print("Poids et caches locaux sauvegardés :", {r: len(s["studies"]) for r, s in seeds.items()}, flush=True)


def finish():
    required = [DIST / name for name in ["data.tar", "assets.tar", "seed_features.tar", "data_build.json"]]
    if not all(p.exists() for p in required):
        raise FileNotFoundError("Lancer d'abord --stage data.")
    with zipfile.ZipFile(DIST / "runtime_code.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((HERE / "runtime").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(HERE / "runtime").as_posix())
    # Extensions .bin : Kaggle ne doit pas décompresser récursivement les archives.
    files = {"data_payload.bin": DIST / "data.tar", "assets_payload.bin": DIST / "assets.tar",
             "seed_payload.bin": DIST / "seed_features.tar", "code_payload.bin": DIST / "runtime_code.zip"}
    wheels = sorted((HERE / "wheels").glob("*.whl"))
    if len(wheels) < 2:
        raise FileNotFoundError("Les wheels timm et iterative-stratification sont requises.")
    files.update({"wheels/" + p.name: p for p in wheels})
    checks = {name: digest(path) for name, path in files.items()}
    build = read(DIST / "data_build.json")
    marker = {"format": "rsna-kaggle-upload-v1", "prepared": build["prepared"]["signature"],
              "files": checks, "seed_studies": build["seed_studies"], "created": "2026-10-16"}
    marker["bundle_id"] = fp(marker)
    save(DIST / "bundle_manifest.json", marker)
    target = DIST / "UPLOAD_TO_KAGGLE.zip"
    with zipfile.ZipFile(target.with_suffix(".partial"), "w", allowZip64=True) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.write(DIST / "bundle_manifest.json", "bundle_manifest.json")
        archive.write(HERE / "GUIDE_KAGGLE_FR.md", "GUIDE_KAGGLE_FR.md")
        for name in ["KAGGLE_GPU.ipynb", "COLAB_GPU.ipynb", "OFFRES_ETUDIANT_FR.md",
                     "VERIFICATION_EN_LIGNE_FR.md", "REFERENCES_MODELES_FR.md"]:
            archive.write(HERE / name, name)
    target.with_suffix(".partial").replace(target)
    save(DIST / "UPLOAD_SHA256.json", {"file": target.name, "sha256": digest(target),
                                      "bytes": target.stat().st_size, "bundle_id": marker["bundle_id"]})
    print(f"Prêt à téléverser : {target} ({target.stat().st_size / 1e9:.2f} Go)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["data", "finish"], required=True)
    args = parser.parse_args()
    build_data() if args.stage == "data" else finish()
