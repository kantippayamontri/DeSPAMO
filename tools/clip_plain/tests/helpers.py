import json
from pathlib import Path

import numpy as np

from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json


def config_at(tmp_path):
    from tools.clip_plain import policy

    c = json.loads(Path(policy.__file__).with_name("config.json").read_text())
    for key in [
        "parent_root",
        "feature_base",
        "motion_root",
        "annotations",
        "study_root",
        "archive_root",
        "historical_root",
        "cache",
    ]:
        c[key] = str(tmp_path / key)
    return c


def parent_fixture(tmp_path):
    c = config_at(tmp_path)
    root = Path(c["parent_root"])
    encoder = dict(
        model=c["model"],
        revision=c["revision"],
        scales=[224, 448],
        tokens=c["tokens"],
        dtype="float32",
        tile_order="row_major_tile_then_frame",
        processor_sha256="processor",
        model_files={},
    )
    identity = dict(
        policy="official-frozen-clip-s2-spatial-v1",
        kind="spatial",
        width=2048,
        encoder=encoder,
        inputs={"source": {}},
        precision="float32",
    )
    inventory, records = {}, []
    annotations = Path(c["annotations"])
    annotations.mkdir(parents=True)
    for split, count in [("train", 8), ("dev", 3), ("test", 2)]:
        inventory[split] = {}
        rows = {}
        for n in range(count):
            clip = f"{split}{n}"
            frames = tmp_path / "frames" / split / clip
            frames.mkdir(parents=True)
            paths = []
            for k in range(20):
                p = frames / f"{k:03}.dat"
                p.write_bytes(bytes([k]))
                paths.append(str(p))
            source_rows = [
                dict(index=k, path=p, sha256=sha256_file(Path(p))) for k, p in enumerate(paths)
            ]
            values = np.full((20, 2048), n + 1, dtype=np.float32)
            values[:, 1024:] += 50
            p = root / split / f"{clip}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, values)
            receipt = dict(
                clip_id=clip,
                split=split,
                identity_hash=digest(identity),
                frame_count=20,
                feature_rows=20,
                width=2048,
                paths=paths,
                source_rows=source_rows,
                source_hash=digest([(r["path"], r["sha256"]) for r in source_rows]),
                source_indices=list(range(20)),
                feature_hash=sha256_file(p),
                extra={},
            )
            atomic_json(root / "receipts" / split / f"{clip}.json", receipt)
            inventory[split][clip] = receipt
            records.append(
                dict(
                    clip_id=clip,
                    split=split,
                    path=f"{split}/{clip}.npy",
                    length=20,
                    width=2048,
                    dtype="float32",
                )
            )
            rows[n] = dict(
                fileid=clip,
                signer="Signer01",
                text=f"{split} german {n}",
                gloss="G",
                en_text="en",
                fr_text="fr",
                es_text="es",
            )
        np.save(annotations / f"{split}_info_ml.npy", rows)
    atomic_json(root / "complete/identity.json", identity)
    atomic_json(
        root / "complete/manifest.json",
        dict(schema_version=1, encoder=digest(identity), expected_dim=2048, records=records),
    )
    atomic_json(
        root / "complete/frame_rows.json",
        dict(
            encoder_key=digest(identity),
            clips={k: v for clips in inventory.values() for k, v in clips.items()},
        ),
    )
    c["parent_manifest_sha256"] = sha256_file(root / "complete/manifest.json")
    c["parent_identity_sha256"] = sha256_file(root / "complete/identity.json")
    return c
