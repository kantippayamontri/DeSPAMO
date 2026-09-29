"""Signer-disjoint logical views of the immutable physical train corpus."""

from torch.utils.data import Dataset

from despamo.appearance.provenance import digest

DEV_SIGNER = "Signer03"
TEST_SIGNER = "Signer07"
POLICY = "signer-pilot-seed0-v1"


def build_split(source: dict, journals: list[dict]) -> dict:
    if source.get("split") != "train" or not isinstance(source.get("clips"), list):
        raise ValueError("signer split requires physical train source")
    clips = source["clips"]
    by_id = {clip["clip_id"]: clip for clip in clips}
    records = {journal["clip_id"]: journal for journal in journals}
    if (
        not clips
        or len(by_id) != len(clips)
        or len(records) != len(journals)
        or set(by_id) != set(records)
        or any(row["status"] not in {"valid", "failed"} for row in journals)
    ):
        raise ValueError("duplicate, missing or nonterminal signer-pilot source/journal")
    assigned = {"train": [], "dev": [], "test": []}
    for clip_id, clip in by_id.items():
        signer = clip["signer"]
        if not isinstance(signer, str) or not signer.strip():
            raise ValueError("missing signer")
        group = "dev" if signer == DEV_SIGNER else "test" if signer == TEST_SIGNER else "train"
        assigned[group].append(clip_id)
    if any(not ids for ids in assigned.values()):
        raise ValueError("empty signer-disjoint group")
    groups = {}
    for group, ids in assigned.items():
        groups[group] = {
            "clip_ids": sorted(ids),
            "signers": sorted({by_id[clip_id]["signer"] for clip_id in ids}),
            "valid": sum(records[clip_id]["status"] == "valid" for clip_id in ids),
            "failed": sum(records[clip_id]["status"] == "failed" for clip_id in ids),
        }
    result = {
        "schema_version": 1,
        "policy": POLICY,
        "physical_split": "train",
        "dev_signer": DEV_SIGNER,
        "test_signer": TEST_SIGNER,
        "source_hash": digest(source),
        "journal_inventory_hash": digest(
            sorted([item["clip_id"], item["status"], digest(item)] for item in journals)
        ),
        "groups": groups,
    }
    return {**result, "split_hash": digest(result)}


def verify_split(split: dict, source: dict, journals: list[dict]) -> None:
    if split != build_split(source, journals):
        raise ValueError("signer-pilot split/source/journal identity changed")


class PhoenixTrainView(Dataset):
    """A logical split whose feature lookup stays at physical `train/ID.npy`."""

    def __init__(self, parent, clip_ids: tuple[str, ...]) -> None:
        if parent.split != "train":
            raise ValueError("signer view parent must be physical train")
        lookup = {item["fileid"]: i for i, item in enumerate(parent.records)}
        if len(lookup) != len(parent.records):
            raise ValueError("duplicate physical train clip ID")
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("duplicate logical split clip ID")
        if not clip_ids or any(clip_id not in lookup for clip_id in clip_ids):
            raise ValueError("empty or unknown logical split clip ID")
        self.parent = parent
        self.indices = tuple(lookup[clip_id] for clip_id in clip_ids)
        self.records = [parent.records[index] for index in self.indices]
        self.split = "train"
        self.spatial_root = parent.spatial_root
        self.spatial_manifest = parent.spatial_manifest
        self.motion_root = getattr(parent, "motion_root", None)
        self.motion_manifest = getattr(parent, "motion_manifest", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self.indices):
            raise IndexError(index)
        return self.parent[self.indices[index]]
