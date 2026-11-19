# E1.4 text-anchored score correction implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, then one independent code review.

**Goal:** Implement the approved E1.4 protocol without running the real experiment.

**Architecture:** Independent numerical/workflow/reporting/CLI modules. Validate and snapshot completed E1.3, reuse q/T/Q and original support sets, keep the class-text branch fixed and add a controlled visual score residual. Preserve all historical source/config/result bytes.

**Tech Stack:** Existing NumPy/PyYAML/pytest; no new dependency, model, text encoding or optimizer.

**Spec:** Approved preceding E1.4 proposal and docs/E14_GUIDE.md.

## Global constraints

- Same CLIP space, fixed class prompts, source split,15 supports/50dev and10support seeds; final test sealed.
- A1: qT +1.0*qp; A2(primary): qT +0.1*qp; A3: A2 with visual rows[1,2,3,4,0]; A4: A2 with p@Q, common Q seeds0..4. Never rotate text or normalize combined weights/scores.
- New80 evaluations (10+10+10+50); reuse T0 once, P0/P1 each10; total101 saved/reference evaluations, no gradient training.
- Save text/auxiliary/final scores and all model branches; reconstruct from original support IDs independently before reporting. Query truth only computes metrics.
- T0 no artificial support SD; random controls average within support then10support SD; primary A2−T0/A3/A4.
- Continuation gate: A2−T0 >=.02 Macro-F1,>=8positive supports, better mean than both wrong controls; exploratory, no significance claim.
- No real prepare/run, Git push, Vault mutation, source overwrite or audit-label modification.

## Review focus

1. Text branch must equal original T0 in every condition; A4 perturbs only visual p, A3 mapping must be fixed.
2. No row/final normalization or accidental softmax/temperature; lambda is fixed and stored.
3. Saved weights, branch scores and final predictions must match independent support-only reconstruction even after rehashing tampered artifacts.
4. All support/source identities must pair with21references; T0 comparisons remain a single fixed prediction set.
5. Complete grid, hash/code/env parent snapshots, synthetic production guard and safe output boundary.

## Tasks / ledger

- [x] RED/GREEN numerical tests and e14_features.py plus fixed config.
- [x] RED/GREEN complete80-evaluation workflow and independent validation/reporting.
- [x] RED/GREEN CLI guards and script13.
- [x] Guide/contracts/ignore rules; full regression and original E1.3 read-only preflight.
- [x] Independent review/fixes and verification record; no formal E1.4 execution.

Ruling: Modify the user-named working copy, preserving previous uncommitted E1.2/E1.3 work; leave uncommitted and do not touch unrelated untracked files.

Final ledger: 160 tests passed in388.63s; complete new synthetic workflow independently passed238.67s. Independent static review found no Critical/Important. Real E1.3 read-only check verified350cached/d512/50dev,80planned+21references,zero encoding/gradient training. Historical99file hashes unchanged; guide4PowerShell blocks parse; links/ignore rules verified; no real E1.4 outputs, Vault writes, commit or push. Evidence: docs/E14_VERIFICATION.md.
