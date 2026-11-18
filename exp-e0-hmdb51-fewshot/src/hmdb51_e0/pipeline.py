"""Stage orchestration; identities are checked before labels meet features."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import traceback
import numpy as np

from .clip_features import CHECKPOINT_SHA256, pool_features
from .config import CLASSES, ROOT, safe_video_path, validate_config
from .feature_cache import FeatureCache
from .manifest import MANIFEST_FIELDS, build_manifest, validate_manifest
from .metrics import classification_metrics
from .provenance import digest, environment, new_directory, now, read_json, sha256_file, source_identity, write_csv, write_json
from .ridge import fit_ridge
from .sampling import SUPPORT_FIELDS, feasible_shots, sample_support_sets, validate_support_sets
from .video_io import decoder_identity, midpoint_indices, sampled_frames

PREDICTION_FIELDS = ["run_id", "evaluation_stage", "support_seed", "k", "video_id", "true_class_id", "predicted_class_id",
                     *[f"score_{name}" for name in CLASSES]]


def protocol_config(config: dict) -> dict:
    return {k: v for k, v in config.items() if k != "status"}


def official_clip_condition(identity: dict) -> bool:
    encoder = identity.get("encoder", {})
    return (encoder.get("implementation") == "openai/CLIP" and encoder.get("architecture") == "ViT-B/32"
            and encoder.get("checkpoint_sha256") == CHECKPOINT_SHA256)


def freeze_manifest(audit: dict, config: dict, output: Path) -> Path:
    rows = build_manifest(audit, config)
    payload = {"config": protocol_config(config), "rows": rows,
               "annotation_hashes": audit["annotation_hashes"], "decoder": audit["decoder"]}
    bundle = {**payload, "protocol_id": digest(payload), "created_at": now(),
              "audit_sha256": digest(audit), "status": "success"}
    directory = new_directory(output, "manifest")
    write_json(directory / "bundle.json", bundle)
    write_json(directory / "audit.json", audit)
    write_csv(directory / "video_manifest.csv", rows, MANIFEST_FIELDS)
    write_json(directory / "class_mapping.json", [{"class_id": i, "class_name": name} for i, name in enumerate(CLASSES)])
    for role in ("support_pool", "development", "final_test", "unused", "excluded"):
        write_json(directory / f"{role}.json", [r["video_id"] for r in rows if r["experimental_role"] == role])
    return directory / "bundle.json"


def load_bundle(path: Path, config: dict) -> dict:
    validate_config(config)
    bundle = read_json(path)
    payload = {key: bundle[key] for key in ("config", "rows", "annotation_hashes", "decoder")}
    if bundle.get("status") != "success" or digest(payload) != bundle["protocol_id"] or bundle["config"] != protocol_config(config):
        raise ValueError("manifest identity/hash/config mismatch")
    validate_manifest(bundle["rows"], config)
    return bundle


def write_supports(bundle: dict, config: dict, output: Path) -> Path:
    records = sample_support_sets(bundle["rows"], config, bundle["protocol_id"])
    directory = new_directory(output, "supports")
    payload = {"protocol_id": bundle["protocol_id"], "records": records, "shots": feasible_shots(bundle["rows"], config)}
    write_json(directory / "support_sets.json", {**payload, "sha256": digest(payload)})
    write_csv(directory / "support_sets.csv", records, SUPPORT_FIELDS)
    return directory / "support_sets.json"


def load_supports(path: Path, bundle: dict, config: dict) -> list[dict]:
    data = read_json(path)
    payload = {k: data[k] for k in ("protocol_id", "records", "shots")}
    if data["sha256"] != digest(payload) or data["protocol_id"] != bundle["protocol_id"]:
        raise ValueError("support artifact identity mismatch")
    validate_support_sets(data["records"], bundle["rows"], config, bundle["protocol_id"])
    return data["records"]


def smoke_ids(rows: list[dict]) -> tuple[list[str], list[str]]:
    support, query = [], []
    for name in CLASSES:
        ids = sorted(r["video_id"] for r in rows if r["class_name"] == name and r["experimental_role"] == "support_pool")
        if len(ids) < 2:
            raise ValueError("smoke check needs two support-pool videos per class")
        support.append(ids[0])
        query.append(ids[1])
    return support, query


def extract_features(bundle_path: Path, config: dict, paths: dict, encoder, *, stage: str = "development") -> Path:
    if stage not in ("development", "smoke"):
        raise ValueError("final_test is sealed; only development/smoke extraction is implemented")
    bundle = load_bundle(bundle_path, config)
    decoder = decoder_identity()
    if bundle["decoder"] != decoder:
        raise ValueError("decoder identity changed after eligibility audit")
    identity = {"protocol_id": bundle["protocol_id"], "encoder": encoder.identity,
                "decoder": decoder, "pooling": config["pooling"], "video": config["video"],
                "feature_code": {name: sha256_file(ROOT / "src/hmdb51_e0" / name) for name in
                                 ("clip_features.py", "video_io.py", "feature_cache.py")}}
    cache = FeatureCache(paths["feature_cache_root"], identity)
    index_path = cache.root / f"index-{stage}.json"
    rows = [r for r in bundle["rows"] if r["experimental_role"] in ("support_pool", "development")]
    if stage == "smoke":
        a, b = smoke_ids(rows)
        rows = [r for r in rows if r["video_id"] in a + b]
    index = {"status": "running", "started_at": now(), "evaluation_stage": stage, "identity": identity,
             "entries": [], "errors": [], "frozen_verification": None}
    write_json(index_path, index)
    try:
        for number, row in enumerate(rows, 1):
            path = safe_video_path(paths["dataset_root"], row["relative_path"])
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ValueError(f"frozen video identity changed: {row['video_id']}")
            expected = {"video_id": row["video_id"], "video_sha256": row["sha256"],
                        "frame_indices": midpoint_indices(row["frame_count"]).tolist()}
            if cache.exists(expected):
                cache.load(expected)  # Existence is never sufficient evidence.
            else:
                indices, frames = sampled_frames(path, row["frame_count"])
                if indices.tolist() != expected["frame_indices"] or sha256_file(path) != row["sha256"]:
                    raise ValueError(f"video changed during decoding: {row['video_id']}")
                z, vector = pool_features(encoder.encode(frames), config["pooling"]["epsilon"])
                cache.save(expected, z, vector)
            index["entries"].append(expected)
            if number % 10 == 0 or number == len(rows):
                print(f"features {number}/{len(rows)} [{stage}]", flush=True)
                write_json(index_path, index)
        index["frozen_verification"] = encoder.verify_frozen()
        if not index["frozen_verification"].get("unchanged"):
            raise ValueError("encoder freeze verification failed")
        index["status"] = "success"
    except Exception as exc:
        index["status"] = "failed"
        index["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        index["finished_at"] = now()
        write_json(index_path, index)
    return index_path


def load_feature_vectors(index_path: Path, bundle: dict, paths: dict, required: set[str], stage: str) -> tuple[dict, dict]:
    index = read_json(index_path)
    if index["status"] != "success" or index["evaluation_stage"] != stage or not index["frozen_verification"]["unchanged"]:
        raise ValueError("feature extraction is incomplete or stage/freeze check failed")
    identity = index["identity"]
    if identity["protocol_id"] != bundle["protocol_id"]:
        raise ValueError("feature protocol identity mismatch")
    if "feature_code" in identity:
        for name, expected in identity["feature_code"].items():
            if sha256_file(ROOT / "src/hmdb51_e0" / name) != expected:
                raise ValueError("feature implementation changed; extract a new cache")
    cache = FeatureCache(paths["feature_cache_root"], identity)
    by_id = {r["video_id"]: r for r in bundle["rows"]}
    entries = index["entries"]
    if len(entries) != len({e["video_id"] for e in entries}) or not required.issubset({e["video_id"] for e in entries}):
        raise ValueError("duplicate or missing feature identities")
    vectors = {}
    dimension = None
    for entry in entries:
        row = by_id.get(entry["video_id"])
        if row is None or row["experimental_role"] not in ("support_pool", "development"):
            raise ValueError("sealed or unknown video in feature index")
        expected = {"video_id": row["video_id"], "video_sha256": row["sha256"],
                    "frame_indices": midpoint_indices(row["frame_count"]).tolist()}
        if entry != expected:
            raise ValueError("feature/video identity mismatch")
        if entry["video_id"] in required:
            _, vector = cache.load(expected)
            dimension = dimension or len(vector)
            if len(vector) != dimension:
                raise ValueError("mixed feature dimensions")
            encoded_dimension = identity.get("encoder", {}).get("dimension")
            if encoded_dimension is not None and len(vector) != 2 * encoded_dimension:
                raise ValueError("feature dimension differs from actual encoder identity")
            vectors[entry["video_id"]] = vector
    return vectors, identity


def run_development(bundle_path: Path, supports_path: Path, index_path: Path, config: dict, paths: dict,
                    *, stage: str = "development") -> Path:
    if stage not in ("development", "smoke"):
        raise ValueError("final_test remains sealed; use development or smoke")
    bundle = load_bundle(bundle_path, config)
    directory = new_directory(paths["run_root"], f"e0-v0-{stage}")
    batch = {"status": "running", "started_at": now(), "evaluation_stage": stage,
             "protocol_id": bundle["protocol_id"], "config": protocol_config(config), "runs": [],
             "expected_pairs": [], "errors": [], "manifest_file_sha256": sha256_file(bundle_path),
             "support_file_sha256": sha256_file(supports_path), "feature_index_sha256": sha256_file(index_path)}
    write_json(directory / "batch.json", batch)
    try:
        records = load_supports(supports_path, bundle, config)
        by_id = {r["video_id"]: r for r in bundle["rows"]}
        if stage == "smoke":
            smoke_support, evaluation = smoke_ids(bundle["rows"])
            records = [dict(protocol_id=bundle["protocol_id"], support_seed=0, k=1, class_id=i,
                            video_id=vid, rank_in_class=1) for i, vid in enumerate(smoke_support)]
            pairs = [(1, 0)]
        else:
            evaluation = sorted(r["video_id"] for r in bundle["rows"] if r["experimental_role"] == "development")
            pairs = [(k, seed) for k in feasible_shots(bundle["rows"], config) for seed in config["sampling"]["support_seeds"]]
        batch["expected_pairs"] = [list(pair) for pair in pairs]
        batch["evaluation_ids"] = evaluation
        required = set(evaluation) | {r["video_id"] for r in records}
        vectors, identity = load_feature_vectors(index_path, bundle, paths, required, stage)
        batch["feature_identity"] = digest(identity)
        batch["feature_condition"] = "official_openai_clip_vit_b32" if official_clip_condition(identity) else "validation_only_unverified_features"
        batch["source"] = source_identity(ROOT)
        write_json(directory / "environment.json", environment())
        write_json(directory / "manifest_snapshot.json", bundle)
        write_json(directory / "feature_index_snapshot.json", read_json(index_path))
        write_json(directory / "support_sets_snapshot.json", records)
        x_eval = np.stack([vectors[vid] for vid in evaluation])
        # Only these query labels are used for metrics, after prediction.
        truth = np.array([by_id[vid]["class_id"] for vid in evaluation])
        if set(truth.tolist()) != set(range(5)):
            raise ValueError("evaluation manifest must contain all five classes")
        for k, seed in pairs:
            support = sorted((r for r in records if r["k"] == k and r["support_seed"] == seed),
                             key=lambda r: (r["class_id"], r["rank_in_class"]))
            ids = [r["video_id"] for r in support]
            if len(ids) != 5 * k or len(set(ids)) != len(ids) or set(ids) & set(evaluation):
                raise ValueError("support count or video overlap invalid")
            run_dir = new_directory(directory, f"{config['experiment_id']}-p{bundle['protocol_id'][:10]}-{stage}-k{k}-s{seed}")
            run_id = run_dir.name
            run = {"directory": run_id, "k": k, "support_seed": seed}
            batch["runs"].append(run)
            provenance = {"run_id": run_id, "status": "running", "started_at": now(), "evaluation_stage": stage,
                          "k": k, "support_seed": seed, "n_support": len(ids), "support_ids": ids,
                          "support_sha256": digest(sorted(ids)), "evaluation_ids": evaluation,
                          "protocol_id": bundle["protocol_id"], "feature_identity": digest(identity),
                          "manifest_file_sha256": batch["manifest_file_sha256"],
                          "annotation_hashes": bundle["annotation_hashes"], "class_mapping_sha256": digest(list(CLASSES)),
                          "source": batch["source"], "exclusions": [r for r in bundle["rows"] if r["experimental_role"] == "excluded"]}
            write_json(run_dir / "resolved_config.json", protocol_config(config))
            write_json(run_dir / "provenance.json", provenance)
            write_csv(run_dir / "support_set.csv", support, SUPPORT_FIELDS)
            try:
                # No pool/development label array is passed into the fitter.
                weights, intercept = fit_ridge(np.stack([vectors[vid] for vid in ids]),
                                              np.array([r["class_id"] for r in support]), config["classifier"]["lambda"])
                scores = x_eval.astype(np.float64) @ weights + intercept
                metrics = {**classification_metrics(truth, scores), "evaluation_stage": stage, "n_support": len(ids)}
                prediction = scores.argmax(axis=1)
                predictions = [dict(run_id=run_id, evaluation_stage=stage, support_seed=seed, k=k,
                                    video_id=vid, true_class_id=int(truth[i]), predicted_class_id=int(prediction[i]),
                                    **{f"score_{name}": float(scores[i, j]) for j, name in enumerate(CLASSES)}) for i, vid in enumerate(evaluation)]
                write_csv(run_dir / "predictions.csv", predictions, PREDICTION_FIELDS)
                write_json(run_dir / "metrics.json", metrics)
                np.savez_compressed(run_dir / "ridge.npz", W=weights, b=intercept)
                provenance.update(status="success", predictions_sha256=sha256_file(run_dir / "predictions.csv"),
                                  metrics_sha256=sha256_file(run_dir / "metrics.json"),
                                  support_csv_sha256=sha256_file(run_dir / "support_set.csv"))
                (run_dir / "run.log").write_text(f"success; {len(ids)} support videos; {len(evaluation)} {stage} videos\n", encoding="utf-8")
            except Exception:
                provenance["status"] = "failed"
                (run_dir / "run.log").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            finally:
                provenance["finished_at"] = now()
                write_json(run_dir / "provenance.json", provenance)
            write_json(directory / "batch.json", batch)
        batch["status"] = "success"
    except Exception as exc:
        batch["status"] = "failed"
        batch["errors"].append(f"{type(exc).__name__}: {exc}")
        (directory / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        batch["finished_at"] = now()
        write_json(directory / "batch.json", batch)
    return directory
