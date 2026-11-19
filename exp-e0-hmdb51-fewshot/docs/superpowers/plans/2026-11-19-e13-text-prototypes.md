# E1.3 text-reference and prototype implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline and request one independent final review.

**Goal:** Implement the approved E1.3 proposal, without real text preparation or evaluation.

**Architecture:** Add independent numerical, workflow, reporting and CLI modules. Validate/snapshot a complete E1.2 batch, reuse its E1 ancestor's frozen frame features, encode five fixed class prompts only during prepare, derive support prototypes only during run. Historical code/configs/results remain byte-identical.

**Tech Stack:** Existing NumPy/PyYAML/native OpenAI CLIP/pytest; no additional dependencies.

**Spec:** Approved preceding E1.3 design; exact protocol in docs/E13_GUIDE.md.

## Global constraints

- Classes/order clap/jump/run/walk/wave; prompts clapping/jumping/running/walking/waving with fixed template `a photo of a person {verb}.`.
- q=unit(mean16 Z). T=unit(native text E). p_c=unit(mean of current3 support q for class c). All use the same d-dimensional CLIP space (official d512).
- T0=T, evaluated once with k0/support_seed null; P0=p; P1=unit(p+T); P2=unit(p+T[[1,2,3,4,0]]); P3=unit(p+T@Q) for common orthogonal seeds0..4.
- P0/P1/P2 each10 support sets, P3 50; total81 evaluations, no gradient fitting. Fixed K3, frozen encoder, same15 support IDs and50 development videos, final test sealed.
- Mean/prototypes only use support IDs. Query labels only compute metrics. Degenerate normalization fails, no dropping conditions. Zero-shot has no support rows or repeated support statistics.
- Preserve cache code/env/encoder/tokenizer/freeze and parent hash identities; independently reconstruct prototypes/scores/metrics before reporting. No old Python changes, no real runs, no push/Vault writes.

## Review focus

1. T0 must not be duplicated as ten independent repeats; P comparisons must share support hashes.
2. Fixed P2 mapping must be explicit and never use the query's true class to select text.
3. Native text must match the exact joint checkpoint/tokenizer and remain frozen; synthetic CLI rejected.
4. Rehashed corrupt directions, q arrays, model prototypes or scores must fail reconstruction.
5. Output must never overwrite inputs; incomplete/random-seed grids and final-test requests fail.

## Tasks / ledger

- [x] RED/GREEN: numerical geometry, label permutation, support-only prototype tests; e13_features.py and fixed config.
- [x] RED/GREEN: preparation / full81-evaluation workflow / independent report validation; e13_workflow.py and e13_reporting.py.
- [x] RED/GREEN: CLI stage/synthetic/final-test guards; e13_cli.py and script12.
- [x] Guide/contracts/ignore rules, full regression and real E1.2 read-only check.
- [x] Independent review and fixes; verification record; no real prepare/run.

Ruling: Preserve the user-named checkout and all earlier uncommitted E1.2 work; leave E1.3 uncommitted. Do not read or modify unrelated untracked files. No experiments from an interrupted previous turn were found running.

Final ledger: 156 tests passed in258.28s; native text tokenizer/BPE ancestor-binding Important fixed with RED/GREEN unit regression and independently confirmed. Real E1.2 read-only check verified350cached/50dev/d512/81planned evaluations, zero encoding or gradient fits. Native tokenizer hashes match original E1; fixed5prompts tokenize without truncation, no text embedding executed. Historical104file SHA unchanged; no real E1.3 outputs. Four PowerShell guide blocks parse and documentation links/ignore rules validate. Evidence in docs/E13_VERIFICATION.md.
