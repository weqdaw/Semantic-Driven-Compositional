# E0 implementation ledger — 2026-11-16

Plan: `docs/superpowers/plans/2026-11-16-e0-runtime-plan.md`.

User request authorizes implementing and running the documented experiment. Existing specification fixes the design; proceed without a new design-approval round.

Ruling: keep work in this directory — no Git repository exists, so Git worktree/commit steps do not apply. Use code snapshot hashes.

Ruling: use project-local Python 3.11.9 — registered 3.12 cannot launch; original specification recommends rather than requires 3.12. Record and lock actual environment.

Pre-flight: audit feeds manifests with full training decode status; manifests feed support sampling and label-free extraction; support IDs and development IDs join features only during fitting/evaluation; report consumes the entire expected seed/K batch. All stages share class order and protocol hash.

- Task 1: complete. Official Serrelab RAR matched Torchvision's MD5; 1165 five-class entries audited. Six leading-# differences were verified against official reference video bytes and handled using an explicit read-only alias artifact. All 350 training videos passed eligibility before the 50-development/300-support split was frozen.
- Task 2: complete. Official CLIP checkpoint verified; native transforms and 512-D frame / 1024-D video arrays cached. Frozen parameter identity verified before/after; numerical and real smoke tests passed.
- Task 3: complete. All 30 development runs plus a complete repeat retained; per-video scores and metrics identical. Reports validated and plots visually inspected. Independent review completed; no unresolved Critical/Important findings.

Ruling: allow only explicit, official-content-hash-verified physical path aliases — preserves canonical official video identity and sample membership without changing external filenames. DATA_CONTRACTS and EXPERIMENT_GUIDE updated before implementation. Unverified/fuzzy matching remains rejected.

Debugging: constant float32 frames produced roundoff std that block normalization amplified. Float64 accumulation plus a constant-frame regression removed the artifact. Native preprocessing repr contained a callable address and changed cache identity across processes; structured transform serialization and cross-process tests fixed this. Superseded extraction artifacts remain separate and are not included in the reported batch.

Review finding resolved: nonofficial/synthetic vectors could receive a hardcoded CLIP report title. Production CLI now rejects that condition; API-only validation artifacts are labeled validation only. Feature dimension is checked against the actual encoder identity.

Environment verification: 37 pytest cases, pip check, generated-AVI integration, actual CLIP smoke, complete 30-run repeat, one actual new-process feature re-encode (bitwise equal), unchanged parameter hash. Original-source grouping across different files remains unverified and is disclosed; final test remains sealed.
