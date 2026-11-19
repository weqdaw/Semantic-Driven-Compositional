# E1.2 support-scale and concept-audit implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one fresh-context code review at the end.

**Goal:** Implement the approved E1.2 proposal, without executing real preparation or fitting.

**Architecture:** Add independent modules. Snapshot and validate the completed E1.1 batch, reuse its V1/C1/G1 and its E1 parent's R1, and fit per-support scalar normalization. Preserve all historical Python, configs and output bytes.

**Tech Stack:** Existing NumPy/PyYAML/pytest; no additional model or dependency.

**Spec:** User-approved preceding E1.2 design; details in docs/E12_GUIDE.md.

## Global constraints

- Primary K=3 only, support seeds0..9, direction seeds0..4; 240 new fits, original50 development IDs; final test sealed.
- Eight conditions: V1_lambda2, SV1, SC1, SG1, SR1, SVC1, SVG1, SVR1. All lambda1 except V1_lambda2.
- Each block's mean and single RMS centered-norm scale use only the current15 support videos. Apply to development; no per-row L2 after scaling/fusion. Fusion divides concatenation by sqrt2.
- Scale<=1e-6 is a recorded batch failure; never silently drop a condition. Save means/scales and before/after variation.
- Audit existing180 rows only: present/absent/uncertain, reviewer/time/evidence. Do not fill labels or use them in fitting/concept selection.
- Keep old runs/configs untouched, local outputs ignored, no Git push or Vault writes.

## Review focus

1. Developer/query data must never enter scaler fitting; raw inputs remain immutable.
2. Original R1 must be from the same E1 ancestor, Q/direction seeds must not be mixed.
3. Rehashed wrong scalers or predictions must fail deterministic recomputation and saved model forward checks.
4. Reference baselines and random averages must match support IDs; complete grid, no silent omission.
5. Audit must bind to original packet/E1 evidence, reject changed rows and keep incomplete reviews explicit.

## Tasks and ledger

- [x] RED/GREEN numerical tests; add e12_features.py and fixed YAML.
- [x] RED/GREEN workflow test; add e12_workflow.py (snapshots/per-support fitting), e12_reporting.py (independent reconstruction,420 total runs including180 references).
- [x] RED/GREEN audit/CLI tests; add e12_audit.py and e12_cli.py, script11. Optional audit summaries cannot affect fitting.
- [x] Add guide/contracts/ignore rules; full suite and real E1.1 read-only preflight; verify historical bytes.
- [x] Independent review, fix material findings and record verification. No real experiment execution.

Ruling: Work in the user-named existing checkout; leave changes uncommitted. Unrelated existing untracked files are outside this task and remain untouched.

Final ledger: 151 tests passed in273.24s. Two Important audit-evidence findings fixed with observed RED/GREEN regressions and independently re-reviewed. Real E1.1 check verified350 cached/50dev,240 planned fits,180 references,zero fits; existing real audit packet verified180 blank rows. Five PowerShell guide blocks parse without errors. Eighty-two historical file hashes unchanged. No real E1.2 preparation/fitting/report output, Vault mutation, commit or push. Details: docs/E12_VERIFICATION.md.
