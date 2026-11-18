# E0 runtime implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement in this session. User has explicitly requested implementation of the existing experiment. Do not re-open the scaffold-only restriction from the earlier task.

**Goal:** Run the prescribed frozen CLIP five-class development baseline and retain reproducible inputs, predictions and statistics.

**Architecture:** Keep the existing src package and six stage scripts. Audit official annotations and training-video eligibility before freezing manifests; extract label-free, identity-bound features; fit only on recorded support labels; validate every run before aggregation.

**Tech stack:** Python in a project venv, NumPy, OpenCV FFmpeg, OpenAI CLIP/PyTorch, YAML, pytest and matplotlib.

**Spec:** `docs/EXPERIMENT_GUIDE.md`, `docs/DATA_CONTRACTS.md`, existing implementation plan.

## Constraints and implementation choices

- Preserve fixed class order, Split 1, development seed 20261116, 10 development videos/class, K=1/3/5, support seeds 0–9, PCG64, 16 midpoint frames, population std and lambda=1.
- External videos remain read-only; no final-test feature extraction or evaluation in this implementation.
- NumPy NPZ plus JSON metadata is the cache format. Hash video, checkpoint, decoder, native preprocessing, pooling and code identities. Reject malformed/stale caches.
- Use OpenAI CLIP's native processor and verified official ViT-B/32 checkpoint. Disable gradients and verify parameter identity before/after extraction.
- Registered Python 3.12 fails to launch on this machine. Use working Python 3.11.9 in `.venv`; record this environment deviation and freeze actual dependencies.
- No Git repository exists. Work in the user-named directory, do not initialize Git or invent commits; save source snapshot hashes.
- Download official annotations and checkpoint into ignored local directories, recording URL/time/hash. Missing or conflicting required data blocks the real experiment, while implementation and synthetic verification continue.

## Review focus

Missing official rows and path traversal must stop; short/decode failures must precede split freezing; role/content overlap must stop; stale cache identity must stop; incomplete/mixed-seed result batches must never silently aggregate.

## Task 1: audit and manifests

Files: config.py, provenance.py, inventory.py, video_io.py, manifest.py, sampling.py; stage scripts 01/02/04; tests/test_data.py.

Interfaces: `load_config(path) -> dict`, `load_paths(path, root) -> dict[str, Path]`, `audit_dataset(...) -> dict`, `build_manifest(audit, config) -> list[dict]`, `sample_support_sets(rows, config, protocol_id) -> list[dict]`.

- [x] Write and run failing tests for annotation absence/conflicts, unsafe IDs, training-only decode eligibility, role disjointness, deterministic nested support sets, insufficient candidates and identity mismatch.
- [x] Implement validation, read-only audit and immutable manifest bundles with checksums.
- [x] Run tests and real official/local correspondence audit; retain blocking differences if any.

## Task 2: feature representation and ridge

Files: clip_features.py, feature_cache.py, ridge.py, metrics.py; stage 03; tests/test_numerics.py, tests/test_cache.py.

Interfaces: `midpoint_indices(n, count=16)`, `pool_features(raw, epsilon=1e-6) -> (Z,V0)`, `FrozenCLIP`, `FeatureCache`, `fit_ridge(X,y,lambda_=1) -> (W,b)`, `classification_metrics(y,scores) -> dict`.

- [x] Write and run failing tests for exact frame boundaries, zero variance, invalid feature values, cache corruption, regularization scale and unpenalized intercept, class column order and ties.
- [x] Implement fixed native encoding and strictly checked cache; implement centered dual ridge with n*lambda and five-class metrics.
- [x] Run numerical tests and actual small training-only extraction; verify frozen weights.

## Task 3: runs, reports and real execution

Files: pipeline.py, reporting.py, cli.py; all six entry scripts; tests/test_pipeline.py, tests/test_reporting.py; README/environment/acceptance documentation.

- [x] Write and run failing end-to-end tests using generated videos and a clearly synthetic encoder, including final-test refusal, frozen-input changes, missing/duplicate seeds, mixed feature identities and failed runs.
- [x] Save unique runs with config/environment/source/input identities, support IDs, predictions, metrics and failure state. Aggregate only a complete consistent batch; report sample std and unique support counts.
- [x] Execute small real smoke check, then all 30 development runs if official audit passes; verify deterministic rerun and generate learning curve/confusions/report.
- [x] Request one independent final code/protocol review, fix material findings, run the full test suite and document exact completed/blocked milestones. No final-test opening.
