"""Freeze the training-only holdout after eligibility audit."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import numpy as np

from .config import CLASSES, safe_video_path, validate_config

MANIFEST_FIELDS = ["video_id", "relative_path", "class_name", "class_id", "official_split", "official_tag",
                   "experimental_role", "bytes", "sha256", "frame_count", "duration_seconds", "decode_status", "exclusion_reason",
                   "path_alias_verified_sha256", "alias_reference_sha256"]


def build_manifest(audit: dict, config: dict) -> list[dict]:
    validate_config(config)
    if audit.get("status") != "success" or audit.get("errors"):
        raise ValueError("official audit is blocked")
    rows = [dict(r) for r in sorted(audit["records"], key=lambda r: r["video_id"])]
    rng = np.random.Generator(np.random.PCG64(config["development"]["seed"]))
    count = config["development"]["videos_per_class"]
    for row in rows:
        if row["official_tag"] == 1 and row["decode_status"] == "not_checked":
            raise ValueError("training decode audit must precede manifest freezing")
        if row["decode_status"] in ("failed", "too_short"):
            row["experimental_role"] = "excluded"
        else:
            row["experimental_role"] = {0: "unused", 1: "support_pool", 2: "final_test"}[row["official_tag"]]
    for name in CLASSES:
        pool = [r for r in rows if r["class_name"] == name and r["experimental_role"] == "support_pool"]
        if len(pool) <= count:
            raise ValueError(f"insufficient eligible training videos for {name}: {len(pool)}")
        for index in rng.permutation(len(pool))[:count]:
            pool[int(index)]["experimental_role"] = "development"
    validate_manifest(rows, config)
    return rows


def validate_manifest(rows: list[dict], config: dict) -> None:
    if not rows:
        raise ValueError("empty manifest")
    ids, hashes = set(), set()
    allowed = {0: {"unused", "excluded"}, 1: {"support_pool", "development", "excluded"}, 2: {"final_test", "excluded"}}
    for row in rows:
        vid = row["video_id"]
        safe_video_path(Path("."), vid)
        if vid in ids or row["sha256"] in hashes:
            raise ValueError("duplicate video or content overlap in manifest")
        ids.add(vid)
        hashes.add(row["sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
            raise ValueError("missing verified content hash")
        class_id = row["class_id"]
        if not isinstance(class_id, int) or not 0 <= class_id < 5 or row["class_name"] != CLASSES[class_id]:
            raise ValueError("class mapping mismatch")
        safe_video_path(Path("."), row["relative_path"])
        if vid != row["relative_path"]:
            if row.get("path_alias_verified_sha256") != row["sha256"] or re.fullmatch(r"[0-9a-f]{64}", row.get("alias_reference_sha256", "")) is None:
                raise ValueError("relative path differs without verified alias proof")
        if row["relative_path"].split("/")[0] != row["class_name"] or vid.split("/")[0] != row["class_name"] or row["official_split"] != 1:
            raise ValueError("video identity/split mismatch")
        if row["experimental_role"] not in allowed.get(row["official_tag"], set()):
            raise ValueError("official tag/role mismatch")
        if row["experimental_role"] == "excluded":
            if not row["exclusion_reason"] or row["decode_status"] not in ("failed", "too_short"):
                raise ValueError("invalid exclusion")
        elif row["official_tag"] == 1:
            if row["decode_status"] != "ok" or row["frame_count"] is None or row["frame_count"] < 16:
                raise ValueError("training decode eligibility not established")
        elif row["decode_status"] != "not_checked":
            raise ValueError("sealed test/unused content must not be decoded")
    for name in CLASSES:
        counts = Counter(r["experimental_role"] for r in rows if r["class_name"] == name)
        if counts["development"] != config["development"]["videos_per_class"] or not counts["support_pool"] or not counts["final_test"]:
            raise ValueError(f"invalid fixed role counts: {name}")
