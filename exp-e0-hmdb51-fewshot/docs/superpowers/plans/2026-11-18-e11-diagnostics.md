# E1.1 implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation, with one independent review at the end.

**Goal:** Prepare geometry controls, equal-weight fusion, and frame-level concept audit for E1.1 without running the real experiment.

**Architecture:** Add independent E1.1 modules. Validate and snapshot a complete E1 batch, reuse its cached Z/E/C1 and its V1/C1 reference predictions, then derive G1, VC1 and VG1. Historical E0/E1 Python files remain byte-identical to preserve cache provenance.

**Tech Stack:** Existing NumPy, PyYAML, OpenCV, Pillow, torchvision and pytest; no new model dependencies.

**Spec:** `docs/E11_GUIDE.md`; approved preceding experiment recommendation.

## Global constraints

- Development only; final test sealed, frozen CLIP, fixed concept catalogue, original source split and support identities.
- G1 uses E @ Q, one orthogonal Q per seed 0..4, common to all concepts/videos; preserve E Eᵀ within numerical tolerance.
- VC1 and VG1 concatenate separately normalized blocks with equal energy, then L2 normalize. No fusion-weight sweep.
- K=1/3/5, support seeds 0..9 and ridge lambda=1 inherited unchanged. 330 new runs, 60 reused V1/C1 runs.
- Frame audit: 10 label-independent selected development videos, cached positions 2/7/13, concepts pose01/02/03/10/11/12. 180 blank human judgments; original and native crop images.
- No automatic real fitting/preparation, upload, Vault update or human annotation.

## Review focus

1. Parent/artifact mutation, unsafe output overlap and mixed feature identities must be rejected.
2. Saved scores must match actual features and ridge forward evaluation, not only stored hashes.
3. Rotation seeds must be averaged within supports before reporting variation across supports.
4. Audit must use exact cached frame indices/native crop, verify source hashes and never decode sealed test.
5. CLI must refuse synthetic production features and final-test stages.

## Tasks and ledger

- [x] Add failing tests for Gram preservation, fixed fusion, configuration and scheduling; implement `e11_features.py`.
- [x] Add failing workflow tests; implement parent snapshots, cache validation and explicit fitting in `e11_workflow.py`, independent report validation in `e11_reporting.py`.
- [x] Add failing frame/audit/CLI tests; implement `e11_audit.py`, `e11_cli.py`, script 10 and fixed configuration.
- [x] Add detailed startup/data contract documentation and output ignore rules; run full synthetic suite and read-only real preflight.
- [x] Obtain independent review, fix material findings, record verification; leave formal E1.1 fitting unexecuted.

Ruling: Edit the user-named checkout in place and leave changes uncommitted — the request targets this existing working copy, with no publish request.

Final ledger: 146 tests passed (94.07s); real E1 read-only check verified350 cached/50dev/16concepts, zero fits. Independent review found no Critical/Important; Minor synthetic summarize access fixed with red/green regression. Eighty pre-existing source/config/test/result file hashes unchanged. No real E1.1 outputs, Vault writes, commits or pushes. Evidence: docs/E11_VERIFICATION.md.
