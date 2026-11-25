# E2.0-R implementation plan and ledger

User authorized adding E2.0-R code and execution instructions. Implement and verify only; no formal plan/packet creation, video decode, CLIP encoding, annotation, Vault write, commit or push.

## Fixed design

- Six primary concepts:pose01,pose02,pose03 anchors;pose04,pose06,pose14 visibility-oriented additions from the original16-concept catalog. Reuse exact original English prompts/native E embeddings; clarify annotation rules without changing text vectors. Preserve E2.0 foot-concept findings as historical evidence,not a comparable new AUC leaderboard.
- Representative cohort:20new support_pool videos,4per class,source cap2,at least10source groups. Exclude audited development sources and original audit videos; deterministic PCG64(seed20261124) source-balanced round robin,not score selection. Final-test rows lack audited source IDs,remain sealed,and must not be described as source-isolated. Three original cached frame positions2,7,13 =>60frames.
- Three pixel views:original(context labels only),native224 CLIP crop,letterbox224 preserving aspect ratio with BICUBIC resize and RGBpad(123,117,104). Fixed checkpoint/native normalization for both scored inputs. No classifier training or new text encoding.
-360judgments per view,1080primary total;review6distinct videos/one frame each =>36per view/108secondary total,plus actual disagreements. Sequential native→letterbox→original→review-native→review-letterbox→review-original→adjudication. Opaque public IDs,private identity/scores,immutable templates,explicit reviewer_origin human/ai. AI labels never satisfy independent-human continuation gate.
- Optional coverage_supplement cohort via explicit CSV video_id,target_concept_id,target_label,selection_reason,selector,selection_origin.1..12unique videos,all eligible support_pool,source cap2 and source-disjoint from representative cohort. Selection targets are metadata only,never copied into judgment labels. Separate reports; supplemental/enriched samples cannot establish representative prevalence or advance the representative gate.
- Per-input per-concept coverage,AUROC/tie-safe AP,wrong-concept cyclic mapping,source bootstrap. Primary input effect uses paired determinability and AUC on rows with both views determinable and agreeing labels; contradictory labels counted separately. Never infer original-image model scores from original judgment.
- Per-input gate:>=3concepts with determinability>=.70,5positive/5negative each3videos,AUROC>=.70 and correct-minus-wrong>=.05;representative cohort and complete independent-human review/adjudication required. All stages report incomplete progress honestly.

## Modules and commands

- e20r_protocol.py:fixedconfig,source-balanced selection,supplement validation,templates/annotation origins.
- e20r_media.py:letterbox/native pixel views and frozen CLIP batch encoding using original checkpoint;verify identity/parameters,selected E embeddings and raw score reconstruction.
- e20r_workflow.py:check,plan,prepare,load_plan/load_packet,encode,freeze,status. Plans pin originalE1.1/audit/source/catalog/code identities;packets snapshot compact plans,images and source code. No historical source modifications.
- e20r_reporting.py:progress,coverage,retention,per-input alignment,paired-input comparisons,coverage gaps and separate-cohort report.
- e20r_cli.py/scripts/16_e20r_coverage_repair.py:check/plan (--e11-batch,--audit-directory;plan optional --base-plan/--supplement-selection);prepare --plan;encode/status/freeze/summarize --packet. All production paths reject synthetic sources and final test.
- configs/e20r.coverage-repair.yaml;test numerical/media/protocol plus complete synthetic workflow and guards;E20R_GUIDE/VERIFICATION,README,DATA_CONTRACTS,.gitignore.

## Tasks

- [x] Protocol/media/statistical tests RED→GREEN.
- [x] Plan/prepare/encode/freeze/report/CLI tests RED→GREEN,including supplement isolation,AI gate,source/pixel/score tamper,blank/partial reviews.
- [x] Guide/PowerShell/parser/help checks;read-only real preflight. Actual two-synthetic-image GPU smoke passed;no real dataset images encoded.
- [x] Full regression215passed(630.38s),final affected suite15passed(31.78s);current216unique cases covered.120protected original file SHA unchanged;verification recorded. Historical E20 snapshot status verified withpython-B;generated pycache removed without modifying original evidence.

Compatibility ruling:historical E2.0 packets pin all source files at their preparation time. Adding new modules intentionally changes the current global source identity;do not rewrite old packet hashes or relax old validators. The E20R guide must document using the old packet's saved code_snapshot entrypoint with explicit current paths/output to inspect or continue a historical packet. New E20R source validation reads the E1.1/audit evidence directly,not old E2.0's current-code gate.

Execution ruling:reuse the user-named dirty checkout,preserve all preexisting E2.0 work. Only additive E20R modules/config/tests plus explicitly updated shared docs/ignore rules. No new dependencies.
