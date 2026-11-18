from pathlib import Path
import csv
import numpy as np
import pytest

from hmdb51_e0.pipeline import freeze_manifest, load_bundle, write_supports, run_development, extract_features
from hmdb51_e0.feature_cache import FeatureCache
from hmdb51_e0.clip_features import pool_features
from hmdb51_e0.provenance import read_json, write_json


@pytest.fixture
def prepared(tmp_path, config, eligible_rows):
    audit = dict(status="success", records=eligible_rows, errors=[], annotation_hashes={"a": "b"}, decoder={"backend": "fixture"})
    bundle_path = freeze_manifest(audit, config, tmp_path / "manifest")
    bundle = load_bundle(bundle_path, config)
    supports = write_supports(bundle, config, tmp_path / "supports")
    identity = {"protocol_id": bundle["protocol_id"], "encoder": {"fixture": True}}
    cache = FeatureCache(tmp_path / "features", identity)
    entries = []
    rng = np.random.default_rng(87)
    for row in bundle["rows"]:
        if row["experimental_role"] not in ("support_pool", "development"):
            continue
        entry = {"video_id": row["video_id"], "video_sha256": row["sha256"], "frame_indices": list(range(1, 32, 2))}
        z, v = pool_features(rng.normal(size=(16, 7)))
        cache.save(entry, z, v)
        entries.append(entry)
    index_path = cache.root / "index-development.json"
    write_json(index_path, dict(status="success", evaluation_stage="development", identity=identity, entries=entries,
                               frozen_verification={"unchanged": True}))
    paths = {"feature_cache_root": tmp_path / "features", "run_root": tmp_path / "runs"}
    return bundle_path, supports, index_path, paths


def test_full_development_batch_persists_predictions_and_is_reproducible(prepared, config):
    bundle, supports, index, paths = prepared
    first = run_development(bundle, supports, index, config, paths)
    batch = read_json(first / "batch.json")
    assert batch["status"] == "success"
    assert len(batch["runs"]) == 30
    for run in batch["runs"]:
        directory = first / run["directory"]
        with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        assert len(predictions) == 50
        assert len({r["video_id"] for r in predictions}) == 50
        provenance = read_json(directory / "provenance.json")
        assert provenance["status"] == "success"
        assert provenance["n_support"] == 5 * run["k"]
    second = run_development(bundle, supports, index, config, paths)
    reruns = read_json(second / "batch.json")["runs"]
    for a, b in zip(batch["runs"], reruns):
        assert read_json(first / a["directory"] / "metrics.json") == read_json(second / b["directory"] / "metrics.json")


def test_frozen_manifest_tampering_and_sealed_test_fail(prepared, config):
    bundle_path, supports, index, paths = prepared
    with pytest.raises(ValueError, match="final_test|development|sealed"):
        run_development(bundle_path, supports, index, config, paths, stage="final_test")
    data = read_json(bundle_path)
    data["rows"][0]["sha256"] = "f" * 64
    write_json(bundle_path, data)
    with pytest.raises(ValueError, match="identity|hash"):
        load_bundle(bundle_path, config)


def test_incomplete_cache_causes_explicit_failed_batch(prepared, config):
    bundle, supports, index, paths = prepared
    data = read_json(index)
    cache = FeatureCache(paths["feature_cache_root"], data["identity"])
    cache.paths(data["entries"][0])[0].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        run_development(bundle, supports, index, config, paths)
    batches = list(paths["run_root"].glob("*/batch.json"))
    assert len(batches) == 1
    assert read_json(batches[0])["status"] == "failed"


def test_changed_video_blocks_extraction_instead_of_replacement(tmp_path, config, eligible_rows):
    audit = dict(status="success", records=eligible_rows, errors=[], annotation_hashes={}, decoder={})
    bundle_path = freeze_manifest(audit, config, tmp_path / "manifest")
    class Encoder:
        identity = {"synthetic": True}
        def encode(self, frames):
            pytest.fail("must check video identity before encoding")
        def verify_frozen(self):
            return {"unchanged": True}
    root = tmp_path / "videos"
    (root / "clap").mkdir(parents=True)
    (root / "clap/000.avi").write_bytes(b"changed")
    with pytest.raises(ValueError, match="decoder|identity|changed"):
        extract_features(bundle_path, config, {"dataset_root": root, "feature_cache_root": tmp_path / "cache"}, Encoder())
