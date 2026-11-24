# E1.5 implementation verification

Date: 2026-11-20 (Asia/Shanghai). Scope: implementation and verification only; formal E1.5 has not been prepared or run. User clarified that E1.5 is added while completed E1.4 is preserved.

## Implemented protocol

- Original frozen q/T, class order, 3-shot/10supports and source-isolated development IDs.
- IC1 original instance keys; IC2 unit-mean collapsed keys repeated in matching15slots; IC3 original keys with labels shifted+1mod5 and true identities retained.
- Fixed alpha=.1,beta=5.5, mean-of3kernel votes; no selection, encoding or gradient fitting.
- Thirty new evaluations, eleven references (T0once,A2ten),41records; production50dev gives2050repeated predictions.
- Model/evidence/branch reconstruction, exact support SHA pairing, fixed T0 anchor, paired statistics and error transitions.

## Test-first evidence

- Numerical RED:15missing-module failures before implementation; GREEN:15passed.
- Workflow/CLI RED:2missing-module failures before implementation.
- Complete synthetic chain and numerical suite:17passed in366.58s. Includes all41records, parent preservation, missing-schedule rejection, rehashed model/evidence/support-ID/score corruption rejection, sealed final test and production synthetic guard.
- Decision RED:8missing-function failures before extracting the continuation rule; numerical+decision GREEN:23passed in0.15s. Literal cases exercise inclusive gain=.02 and8positive boundary, insufficient gain/consistency, and tied/negative controls.
- Full repository regression: `.\.venv\Scripts\python.exe -m pytest -q` → **185passed in639.23s (10m39s)**, exit0. This includes the original E0–E1.4 tests and all25new E1.5 tests; no failures or warnings were reported.

All synthetic artifacts were produced in pytest temporary directories. They are program verification, not E1.5 research results.

## Real parent read-only preflight

Command (no formal prepare/run):

```powershell
.\.venv\Scripts\python.exe scripts/14_e15_instance_cache.py check --e14-batch 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot\runs\e14-text-anchor-development-20261119T170615813485Z-ec172b40'
```

Result: verified,development,350cached videos,d512,50development videos,30planned new evaluations,11reused references,zero text/image encoding and zero gradient fits. Final test disabled. Protocol identity: `fb75e9e07d306a6296c869a34b59b8c1ff6a95867acbe76dba002578b371d5e5`.

## Independent review and static checks

Independent reviewer found no Critical/Important bugs. Minor normalized-mean interpretation wording was clarified; the continuation gate received eight explicit boundary tests and follow-up review found no material issues. Source and evidence identities, fixed reference counts, labels, per-key voting, statistics and production guards were examined.

Four E15_GUIDE PowerShell blocks parsed with zero errors; all four CLI help commands/guide argument shapes accepted; guide links, fixed YAML defaults and NPZ/prediction field contracts checked. New cache/report/run output ignore rules checked. `git diff --check` passed (only existing Git LF/CRLF notices).

SHA comparison:103baseline files (previous tracked implementation/test/config/script files and selected original E1.4 report evidence) checked unchanged. Shared README,DATA_CONTRACTS and.gitignore are intentionally updated. No formal artifacts/e15_instance_cache directory, E1.5 run or E1.5 report exists. No raw data, Vault, Zotero, Git commit or remote push was changed.

## Interpretation boundaries

Fixed coefficients do not establish an optimal cache. IC1−IC2 compares original instance keys against unit-normalized collapsed means; key geometry also changes. Development has already been reused; support SD and repeated predictions do not establish independent generalization, concept grounding, unseen composition, temporal inference or drowning reliability. See [E15_GUIDE](E15_GUIDE.md) for formal launch commands and the exploratory decision rule.
