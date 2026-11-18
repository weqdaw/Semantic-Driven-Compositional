"""Audit official lists, local identities and training decode eligibility."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from .config import CLASSES, safe_video_path
from .provenance import digest, now, sha256_file
from .video_io import audit_video, decoder_identity


def parse_annotations(root: Path, split: int = 1) -> tuple[list[dict], dict]:
    records, hashes = [], {}
    for class_id, name in enumerate(CLASSES):
        path = Path(root) / f"{name}_test_split{split}.txt"
        if not path.is_file():
            raise ValueError(f"missing official annotation: {path}")
        hashes[path.name] = sha256_file(path)
        seen = set()
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            match = re.fullmatch(r"(\S+\.avi)\s+([012])\s*", line)
            if match is None:
                raise ValueError(f"malformed annotation {path.name}:{number}")
            filename, tag = match.groups()
            video_id = f"{name}/{filename}"
            safe_video_path(Path(root), video_id)
            if video_id in seen:
                raise ValueError(f"duplicate/conflicting official annotation: {video_id}")
            seen.add(video_id)
            records.append({"video_id": video_id, "relative_path": video_id, "class_name": name,
                            "class_id": class_id, "official_split": split, "official_tag": int(tag)})
        if not seen:
            raise ValueError(f"empty official annotation: {path}")
        if not {1, 2}.issubset({r["official_tag"] for r in records if r["class_name"] == name}):
            raise ValueError(f"annotation must contain train and test: {path}")
    return sorted(records, key=lambda r: r["video_id"]), hashes


def audit_dataset(dataset_root: Path, annotations_root: Path, *, workers: int = 4, aliases: dict | None = None) -> dict:
    result = {"status": "blocked", "created_at": now(), "dataset_root": str(dataset_root),
              "errors": [], "missing_videos": [], "extra_videos": [], "duplicate_content": [],
              "records": [], "annotation_hashes": {}, "decoder": decoder_identity()}
    try:
        records, result["annotation_hashes"] = parse_annotations(annotations_root)
    except (ValueError, OSError) as exc:
        result["errors"].append(str(exc))
        return result
    listed = {r["video_id"] for r in records}
    alias_map = (aliases or {}).get("aliases", {})
    result["path_aliases"] = aliases
    try:
        if alias_map and not (aliases or {}).get("reference"):
            raise ValueError("path aliases require official-content verification provenance")
        targets = set()
        for vid, entry in alias_map.items():
            safe_video_path(dataset_root, vid)
            safe_video_path(dataset_root, entry["relative_path"])
            if vid not in listed or vid.split("/")[0] != entry["relative_path"].split("/")[0] or entry["relative_path"] in targets:
                raise ValueError("invalid/duplicate alias identity")
            if entry["relative_path"] in listed - {vid}:
                raise ValueError("alias shadows another official video")
            if safe_video_path(dataset_root, vid).exists() and vid != entry["relative_path"]:
                raise ValueError("alias must not shadow an existing canonical file")
            target = safe_video_path(dataset_root, entry["relative_path"])
            if not target.is_file() or sha256_file(target) != entry["sha256"]:
                raise ValueError(f"path alias verified content hash mismatch: {vid}")
            targets.add(entry["relative_path"])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
        return result
    actual = set()
    for name in CLASSES:
        directory = Path(dataset_root) / name
        if not directory.is_dir():
            result["errors"].append(f"missing class directory: {directory}")
            continue
        actual.update(f"{name}/{p.name}" for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".avi")
    physical = {vid: alias_map.get(vid, {}).get("relative_path", vid) for vid in listed}
    result["extra_videos"] = sorted(actual - set(physical.values()))
    result["missing_videos"] = sorted(vid for vid, relative in physical.items() if relative not in actual)
    if result["missing_videos"]:
        result["errors"].append(f"missing {len(result['missing_videos'])} required official videos")

    def inspect(record):
        row = {**record, "bytes": 0, "sha256": "", "frame_count": None, "duration_seconds": None,
               "decode_status": "not_checked", "exclusion_reason": ""}
        row["relative_path"] = physical[row["video_id"]]
        path = safe_video_path(dataset_root, row["relative_path"])
        if row["relative_path"] not in actual:
            return None
        before = path.stat()
        row["bytes"], row["sha256"] = before.st_size, sha256_file(path)
        if row["video_id"] in alias_map:
            row["path_alias_verified_sha256"] = alias_map[row["video_id"]]["sha256"]
            row["alias_reference_sha256"] = digest(aliases)
            if row["sha256"] != row["path_alias_verified_sha256"]:
                raise ValueError(f"alias content changed during audit: {row['video_id']}")
        # A broken correspondence audit must not trigger any content decoding.
        if not result["errors"] and row["official_tag"] == 1:
            row.update(audit_video(path))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"video changed during audit: {row['video_id']}")
        return row

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            result["records"] = [row for row in pool.map(inspect, records) if row is not None]
    except (OSError, ValueError) as exc:
        result["errors"].append(str(exc))
    content = defaultdict(list)
    for row in result["records"]:
        content[row["sha256"]].append(row["video_id"])
    result["duplicate_content"] = [ids for ids in content.values() if len(ids) > 1]
    if result["duplicate_content"]:
        result["errors"].append("duplicate content identities require investigation")
    result["status"] = "blocked" if result["errors"] else "success"
    return result
