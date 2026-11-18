import copy
from pathlib import Path

import numpy as np
import pytest

from hmdb51_e0.config import CLASSES, load_config, safe_video_path, validate_config
from hmdb51_e0.inventory import parse_annotations, audit_dataset
from hmdb51_e0.manifest import build_manifest, validate_manifest
from hmdb51_e0.sampling import sample_support_sets, validate_support_sets
from hmdb51_e0.video_io import midpoint_indices


def test_missing_and_conflicting_official_annotations(tmp_path):
    with pytest.raises(ValueError, match="annotation"):
        parse_annotations(tmp_path)
    for name in CLASSES:
        (tmp_path / f"{name}_test_split1.txt").write_text("a.avi 1\na.avi 2\n")
    with pytest.raises(ValueError, match="duplicate|conflict"):
        parse_annotations(tmp_path)


@pytest.mark.parametrize("vid", ["../run/a.avi", "run/../../a.avi", "C:/a.avi", "run\\a.avi", "run/a/b.avi", "other/a.avi"])
def test_unsafe_video_identity_is_rejected(tmp_path, vid):
    with pytest.raises(ValueError):
        safe_video_path(tmp_path, vid)


def test_protocol_edits_require_new_implementation(config):
    validate_config(config)
    config["classifier"]["lambda"] = 0.1
    with pytest.raises(ValueError, match="classifier"):
        validate_config(config)


def test_midpoints_and_short_videos():
    assert midpoint_indices(16).tolist() == list(range(16))
    assert midpoint_indices(32).tolist() == list(range(1, 32, 2))
    assert midpoint_indices(17).tolist() == list(range(8)) + list(range(9, 17))
    with pytest.raises(ValueError, match="16"):
        midpoint_indices(15)


def test_split_excludes_failures_before_development(config, eligible_rows):
    eligible_rows[0].update(decode_status="too_short", frame_count=8, exclusion_reason="fewer_than_16_frames")
    audit = dict(status="success", records=eligible_rows, errors=[])
    rows = build_manifest(audit, config)
    validate_manifest(rows, config)
    assert rows == build_manifest(audit, config)
    assert rows[0]["experimental_role"] == "excluded"
    for c in CLASSES:
        selected = [r for r in rows if r["class_name"] == c]
        assert sum(r["experimental_role"] == "development" for r in selected) == 10
        assert next(r for r in selected if r["official_tag"] == 2)["experimental_role"] == "final_test"
        assert next(r for r in selected if r["official_tag"] == 0)["experimental_role"] == "unused"


def test_manifest_refuses_content_overlap_or_unchecked_train(config, eligible_rows):
    eligible_rows[1]["sha256"] = eligible_rows[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate|overlap"):
        build_manifest(dict(status="success", records=eligible_rows), config)
    eligible_rows[1]["sha256"] = "f" * 64
    eligible_rows[0]["decode_status"] = "not_checked"
    with pytest.raises(ValueError, match="decode|audit"):
        build_manifest(dict(status="success", records=eligible_rows), config)


def test_support_is_nested_reproducible_and_disjoint(config, eligible_rows):
    rows = build_manifest(dict(status="success", records=eligible_rows), config)
    support = sample_support_sets(rows, config, "protocol")
    assert support == sample_support_sets(rows, config, "protocol")
    assert len(support) == 10 * 5 * (1 + 3 + 5)
    for seed in range(10):
        ids = [{r["video_id"] for r in support if r["support_seed"] == seed and r["k"] == k} for k in [1, 3, 5]]
        assert ids[0] < ids[1] < ids[2]
    assert not {r["video_id"] for r in support} & {r["video_id"] for r in rows if r["experimental_role"] != "support_pool"}
    broken = copy.deepcopy(support)
    broken[0]["video_id"] = "clap/not_in_manifest.avi"
    with pytest.raises(ValueError):
        validate_support_sets(broken, rows, config, "protocol")


def test_insufficient_support_is_not_silently_replaced(config, eligible_rows):
    eligible_rows[:] = [r for r in eligible_rows if r["video_id"] != "clap/000.avi" and r["video_id"] != "clap/001.avi"]
    rows = build_manifest(dict(status="success", records=eligible_rows), config)
    # The prescribed guide permits only the common feasible subset of K.
    support = sample_support_sets(rows, config, "p")
    assert {r["k"] for r in support} == {1, 3}


def test_audit_missing_video_blocks_and_never_decodes_test(tmp_path):
    root = tmp_path / "videos"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for name in CLASSES:
        (root / name).mkdir(parents=True)
        (root / name / "test.avi").write_bytes(f"fake-test-{name}".encode())
        (annotations / f"{name}_test_split1.txt").write_text("missing.avi 1\ntest.avi 2\n")
    result = audit_dataset(root, annotations)
    assert result["status"] == "blocked"
    assert len(result["missing_videos"]) == 5
    assert all(r["decode_status"] == "not_checked" for r in result["records"])


def test_explicit_content_verified_alias_preserves_official_identity(tmp_path):
    from hmdb51_e0.provenance import sha256_file
    root, annotations = tmp_path / "videos", tmp_path / "annotations"
    annotations.mkdir()
    for name in CLASSES:
        (root / name).mkdir(parents=True)
        (root / name / "train.avi").write_bytes(f"bad-train-{name}".encode())
        (root / name / "test.avi").write_bytes(f"test-{name}".encode())
        filename = "#test.avi" if name == "clap" else "test.avi"
        (annotations / f"{name}_test_split1.txt").write_text(f"train.avi 1\n{filename} 2\n")
    aliases = {"reference": {"source_url": "https://example.org/original.rar"},
               "aliases": {"clap/#test.avi": {"relative_path": "clap/test.avi", "sha256": sha256_file(root / "clap/test.avi")}}}
    result = audit_dataset(root, annotations, aliases=aliases)
    assert result["status"] == "success"
    row = next(r for r in result["records"] if r["video_id"] == "clap/#test.avi")
    assert row["relative_path"] == "clap/test.avi" and row["official_tag"] == 2
    assert row["path_alias_verified_sha256"] == row["sha256"]
    aliases["aliases"]["clap/#test.avi"]["sha256"] = "f" * 64
    assert audit_dataset(root, annotations, aliases=aliases)["status"] == "blocked"
