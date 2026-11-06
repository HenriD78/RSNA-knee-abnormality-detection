"""Vérifie les archives livrées en lecture seule, sans décompression sur disque.

Contrôle les JPEG sélectionnés, les poids et les caches importés ; aucun appel
de modèle ou entraînement. Le reçu JSON permet de retrouver les contrôles réels.
"""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from knee_baseline.constants import LABELS, UID
from knee_baseline.io import fingerprint, read_json, sha256, write_json


def member_digest(archive, member):
    with archive.extractfile(member) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    images, tables = {}, {}
    with tarfile.open(ROOT / "dist/data.tar", "r:") as archive:
        for member in archive:
            assert member.isfile(), member.name
            if member.name.startswith("images/"):
                assert member.name not in images
                images[member.name] = (member.size, member_digest(archive, member))
            else:
                assert member.name.startswith("prepared/") and member.name not in tables
                tables[member.name] = archive.extractfile(member).read()
    meta = json.loads(tables["prepared/manifest.json"])
    assert fingerprint({k: v for k, v in meta.items() if k != "signature"}) == meta["signature"]
    for name, expected in meta["table_hashes"].items():
        payload = tables["prepared/" + name]
        assert hashlib.sha256(payload).hexdigest() == expected
        if name != "selected_images.csv":
            # Les folds/labels sont conservés exactement, octet pour octet.
            assert payload == (ROOT.parent / "artifacts/full/prepared" / name).read_bytes()
    selected = pd.read_csv(io.BytesIO(tables["prepared/selected_images.csv"]))
    assert set(selected.path) == set(images)
    for row in selected.itertuples(index=False):
        assert images[row.path] == (row.size_bytes, row.sha256)
    master = pd.read_csv(io.BytesIO(tables["prepared/train_folds.csv"]), dtype={UID: str})
    targets = pd.read_csv(io.BytesIO(tables["prepared/targets.csv"]), dtype={UID: str})
    assert len(master) == 4407 and master[UID].nunique() == 4407
    assert int(master.is_gold.sum()) == 58 and master.active.all()
    assert set(master.fold) == set(range(5))
    assert master.groupby("group_id").fold.nunique().max() == 1
    gold = targets[targets.is_gold]
    assert (gold[[c + "__conf" for c in LABELS]].to_numpy() == 1).all()
    assert set(selected[selected.split == "train"][UID]) == set(master[UID])
    assert selected[selected.split == "test"][UID].nunique() == 3
    assert (selected.groupby(["split", UID, "slot"]).size() <= 8).all()
    assert not selected.duplicated(["split", UID, "slot", "path"]).any()
    build = read_json(ROOT / "dist/data_build.json")
    with tarfile.open(ROOT / "dist/assets.tar", "r:") as archive:
        for reader, expected in build["assets"].items():
            record = json.load(archive.extractfile(reader + "/manifest.json"))
            assert record["revision"] == expected["revision"]
            weight = archive.getmember(reader + "/" + record["weight_file"])
            assert member_digest(archive, weight) == expected["weight_sha256"] == record["weight_sha256"]
    with tarfile.open(ROOT / "dist/seed_features.tar", "r:") as archive:
        seed = json.load(archive.extractfile("manifest.json"))
        assert seed["prepared"] == meta["signature"]
        for reader, entry in seed["readers"].items():
            for uid, record in entry["studies"].items():
                assert uid in set(master[UID])
                payload = archive.extractfile(reader + "/train/" + record["file"]).read()
                assert hashlib.sha256(payload).hexdigest() == record["sha256"]
                with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                    x, mask = arrays["features"], arrays["mask"]
                    assert x.shape == (6, 8, entry["encoder"]["feature_dim"])
                    assert x.dtype == np.float16 and np.isfinite(x).all()
                    assert mask.shape == (6, 8) and mask.any()
    receipt = {"selected_images": len(selected), "image_bytes": sum(s[0] for s in images.values()),
               "by_split": selected.groupby("split").size().to_dict(),
               "studies": len(master), "gold": int(master.is_gold.sum()),
               "report_groups": master.group_id.nunique(),
               "fold_counts": master.groupby("fold").size().to_dict(),
               "gold_fold_counts": master[master.is_gold].groupby("fold").size().to_dict(),
               "seed_studies": build["seed_studies"], "prepared_signature": meta["signature"],
               "table_bytes_match_original": True, "all_image_weight_seed_hashes_valid": True}
    write_json(ROOT / "dist/payload_check.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
