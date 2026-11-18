"""Recorded nested support sets with a fresh PCG64 generator per seed."""
from __future__ import annotations

import numpy as np

from .config import CLASSES
from .manifest import validate_manifest

SUPPORT_FIELDS = ["protocol_id", "support_seed", "k", "class_id", "video_id", "rank_in_class"]


def feasible_shots(rows: list[dict], config: dict) -> list[int]:
    minimum = min(sum(r["class_name"] == name and r["experimental_role"] == "support_pool" for r in rows) for name in CLASSES)
    shots = [k for k in config["sampling"]["shots"] if k <= minimum]
    if not shots:
        raise ValueError("no common feasible shot count")
    return shots


def sample_support_sets(rows: list[dict], config: dict, protocol_id: str) -> list[dict]:
    validate_manifest(rows, config)
    shots = feasible_shots(rows, config)
    records = []
    for seed in config["sampling"]["support_seeds"]:
        rng = np.random.Generator(np.random.PCG64(seed))
        for class_id, name in enumerate(CLASSES):
            ids = sorted(r["video_id"] for r in rows if r["class_name"] == name and r["experimental_role"] == "support_pool")
            permuted = [ids[int(i)] for i in rng.permutation(len(ids))]
            for k in shots:
                records.extend(dict(protocol_id=protocol_id, support_seed=seed, k=k, class_id=class_id,
                                    video_id=vid, rank_in_class=rank) for rank, vid in enumerate(permuted[:k], 1))
    return records


def validate_support_sets(records: list[dict], rows: list[dict], config: dict, protocol_id: str) -> None:
    expected = sample_support_sets(rows, config, protocol_id)
    key = lambda r: (r["support_seed"], r["k"], r["class_id"], r["rank_in_class"], r["video_id"])
    if sorted(records, key=key) != sorted(expected, key=key):
        raise ValueError("support sets differ from frozen deterministic protocol")
