# E1.5 instance-cache experiment design

The user approved the preceding E1.5 proposal and clarified that this request adds E1.5 while preserving completed E1.4. Implement and verify only; do not run the formal experiment, prepare formal snapshots, change the Vault, commit, or push.

## Question and alternatives

Can retaining the individual support videos improve the fixed class-text anchor compared with collapsing each class into one mean prototype? The selected approach is a frozen, nonparametric instance cache inspired by Tip-Adapter, adapted to the existing video features. Continuing the E1.4 coefficient search would not isolate instance retention. A richer semantic/concept model is a separate subsequent study if this diagnostic fails.

## Fixed resources and scoring

Use the validated completed E1.4 batch as parent. Reuse its original E1.3 q/T bank, 5 classes in clap/jump/run/walk/wave order, original 3-shot support sets (15 videos), 10 support seeds and 50 development videos. No decoding, image/text encoding, gradients, additional labels, or final-test access.

Implementation choices made here: alpha=0.1, beta=5.5, per-class mean voting, all fixed before E1.5 evaluation. Beta follows the existing Tip-Adapter literature baseline; alpha is a fixed weak-correction convention, not an optimal value or a claim of matching E1.4 branch scales. No hyperparameter grid, support fitting, development-based selection, softmax or temperature calibration.

For unit query q and cache key k_i, affinity_i=exp(-beta*(1-q dot k_i)); vote_i=affinity_i/3. Cache score for class c sums vote_i whose cache label is c. Final score_c=q dot T_c + alpha*cache_score_c. Raw text scores must match the single original T0 prediction set.

- IC1 primary: all 15 original support vectors and correct labels.
- IC2 collapsed control: replace each support key with its class unit mean prototype, keeping 15 slots, the same labels, kernel, alpha, beta and denominator.
- IC3 wrong-label control: retain IC1 keys and shift every support label by +1 modulo 5. Class balance and all numerical hyperparameters remain unchanged; original support labels/identities remain recorded separately.
- Reuse T0 once and A2 for each of 10 original supports. Total: 30 new evaluations plus 11 references = 41 records; 2050 repeated video predictions, not independent videos.

## Interfaces and artifacts

Create e15_features/workflow/reporting/cli modules, script 14_e15_instance_cache.py and configs/e15.instance-cache.yaml. Numerical API: cache_model(bank,representation,support_indices,labels,config=DEFAULT_CONFIG) returns text_prototypes,visual_prototypes,support_keys,cache_keys,true_labels,cache_labels,cache_values. score_branches(queries,model,config=DEFAULT_CONFIG) returns final,text,cache,cosine_similarities,affinities,votes. schedule(config) returns [representation,k,support_seed] records, ordered IC1/IC2/IC3 then seeds0..9.

CLI: check/prepare/run take --e14-batch; prepare writes an independent input snapshot; run requires --feature-index and accepts only development; summarize takes --batch. Production CLI refuses synthetic/unverified features before output creation. Outputs use fresh timestamped directories under artifacts/e15_instance_cache, runs/e15-instance-cache-development-* and reports/e15-instance-cache-report-*.

Save each cache model, exact original support rows, resolved config, separate text/cache/final per-class prediction scores, metrics, provenance, and cache_evidence.npz (query IDs, support IDs, cosine similarities, affinities and individual per-key votes). Evidence matrix columns follow support_set.csv order, rows follow evaluation IDs. Reconstruct all arrays and scores independently from the validated parent's bank and original support identities before accepting a report, even if corrupted files have been rehashed. Bind parent tree, config, code/environment snapshots, feature identity and complete run schedule.

## Analysis and decision

Primary pairs: IC1-T0, IC1-IC2, IC1-IC3. Secondary: IC1-A2, IC2-A2, IC2-T0, IC3-T0. Report Macro-F1, accuracy, per-class recall, support SD, paired deltas/positive counts, and changed/fixed/harmed/both-wrong/net-correct transitions relative to T0. Verify support SHA pairing. T0 stays one fixed reference without support SD; no rotations or ensembles.

Exploratory continuation requires IC1-T0 mean Macro-F1 gain >=0.02, at least8/10positive supports, and positive mean differences from IC2 and IC3. This is not statistical significance, independent generalization, concept grounding, unseen composition or drowning detection validation. Reused references may have unavailable cache fields, which must remain blank. Failed/null findings must remain reportable.

## Verification

Use hand-derived numerical tests for nonlinear instance effects, alpha=0退化 behavior via a numerical override if explicitly supported (production config remains fixed), correct/collapsed/wrong-label controls and query-label independence. Exercise a full synthetic parent chain and all41 records, missing-run rejection, rehashed model/branch/vote corruption, sealed final test, synthetic production guard and output boundary protection. Run the full regression suite and a real E1.4 read-only check only. Preserve previous source/config/result bytes except explicitly updated shared documentation/ignore rules.
