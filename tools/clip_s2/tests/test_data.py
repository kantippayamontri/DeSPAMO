import numpy as np
import pytest

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.phoenix14t import Phoenix14T
from tools.clip_s2.data import batch_for
from tools.dinov3.identity import sha256_file


def test_batch_uses_admitted_roots_and_rejects_changed_array(tmp_path):
    roots = {k: tmp_path / k for k in ("spatial", "motion")}
    hashes, manifests = {}, {}
    for kind, width, value in (("spatial", 2048, 7), ("motion", 1024, 9)):
        folder = roots[kind] / "train"
        folder.mkdir(parents=True)
        p = folder / "clip.npy"
        np.save(p, np.full((5 if kind == "spatial" else 1, width), value, dtype=np.float32))
        hashes[kind] = {("train", "clip"): sha256_file(p)}
        record = FeatureRecord(
            "clip", "train", "train/clip.npy", 5 if kind == "spatial" else 1, width, "float32"
        )
        manifests[kind] = FeatureManifest(1, "fixture", width, (record,))
    annotation = tmp_path / "annotation.npy"
    np.save(
        annotation,
        {
            0: dict(
                fileid="clip",
                signer="Signer01",
                gloss="G",
                text="german",
                en_text="en",
                es_text="es",
                fr_text="fr",
            )
        },
    )
    ds = Phoenix14T(
        annotation,
        "train",
        roots["spatial"],
        roots["motion"],
        manifests["spatial"],
        manifests["motion"],
    )
    admission = dict(
        roots=roots,
        datasets={"train": ds},
        ids={"train": ("clip",)},
        lookup={"train": {"clip": 0}},
        feature_hashes=hashes,
    )
    batch = batch_for(admission, "train", ("clip",))
    assert batch.spatial[0, 0, 0] == 7 and batch.motion[0, 0, 0] == 9
    np.save(roots["spatial"] / "train/clip.npy", np.zeros((5, 2048), dtype=np.float32))
    with pytest.raises(ValueError, match="changed"):
        batch_for(admission, "train", ("clip",))
