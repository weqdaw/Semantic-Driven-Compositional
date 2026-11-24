# E1.5 instance-cache implementation plan

> **For agentic workers:** Implement the approved design with test-first changes; independent numerical and documentation tasks may run in parallel, followed by whole-change review.

**Goal:** Add a controlled instance-cache diagnostic without running the formal experiment.

**Architecture:** Preserve E1.4 modules and artifacts; add numerical, workflow, reporting and CLI modules that validate and snapshot completed E1.4 and reuse its q/T bank and supports.

**Tech Stack:** Existing Python3.12, NumPy, PyYAML and pytest; no new dependencies.

**Spec:** docs/superpowers/specs/2026-11-20-e15-instance-cache-design.md

## Global constraints

- Frozen original features, class prompts and15support/50dev protocol; final test sealed.
- IC1/IC2/IC3 each10runs, alpha=.1,beta=5.5, class mean votes; reuse T0 once and A2 ten times.
- Raw text+alpha*cache scores, no calibration, search, fitting or encoding.
- No formal prepare/run, Vault edits, historical source overwrite, commit or push.

## Review focus

1. Nonlinear per-instance voting must differ from collapsing keys; slot count and denominator remain matched.
2. Wrong-label keys stay unchanged; only labels/values shift, with true labels preserved.
3. Rehashed model, branch, evidence or support mutations must fail independent reconstruction.
4. T0 remains one evaluation, and reused A2 uses identical support SHA; no fake independent repeats.
5. Sealed test, production-feature guard, source identity and safe output behavior remain enforced.

## Tasks / ledger

- [x] Task1: Numerical tests RED (15 missing-module failures), implement e15_features/config, targeted GREEN (15 passed). Owns tests/test_e15_features.py,src/hmdb51_e0/e15_features.py,configs/e15.instance-cache.yaml.
- [x] Task2: Workflow/CLI/report tests RED (2 missing-module failures); implement e15_workflow/reporting/cli and script14. Complete synthetic chain,41records, branch/evidence reconstruction and tamper rejection passed (17 numerical+workflow tests,366.58s). Added continuation boundary RED/GREEN (8 missing-function failures then23numerical+decision passed).
- [x] Task3: Add E15_GUIDE and shared documentation/ignore references, consistent launch commands and interpretation boundaries. Four CLI help/argument checks and four PowerShell parser blocks passed; artifact ignore rules and links checked.
- [x] Task4: Independent review and follow-up found no remaining Critical/Important issues; full regression185passed in639.23s. Real parent read-only preflight passed;103baseline historical file SHA unchanged. Verification:docs/E15_VERIFICATION.md. No formal E1.5prepare/run, Vault mutation, commit or push.

Ruling: The user's explicit E1.5 approval covers the preceding design; settle numerical defaults above as implementation choices without another approval round. Work directly in the named clean checkout, leave changes uncommitted. Shared docs are intentionally updated; all prior experimental implementations/config/results remain unchanged.

Read-only production preflight passed:350cached videos,d512,50development videos,30planned new evaluations+11references; zero encoding or gradient training. Formal E1.5 prepare/run not executed.
