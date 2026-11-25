"""Per-concept ranking, source-cluster uncertainty and audit decisions."""
import numpy as np

from .e20_protocol import DEFAULT_CONFIG, validate_config, CONCEPTS


def ranking_metrics(labels, scores):
    y, score = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != score.shape or not np.isin(y, (0, 1)).all() or not np.isfinite(score).all():
        raise ValueError('ranking requires finite scores and binary labels')
    prevalence = float(np.mean(y)) if len(y) else None
    if not len(y) or len(np.unique(y)) != 2:
        return dict(auroc=None, average_precision=None, prevalence=prevalence)
    positive, negative = score[y == 1], score[y == 0]
    differences = positive[:, None]-negative[None, :]
    auc = float(np.mean((differences > 0)+.5*(differences == 0)))
    ap, previous_recall = 0., 0.
    for threshold in np.unique(score)[::-1]:
        selected = score >= threshold
        tp = float(np.sum(y[selected]))
        recall = tp/len(positive)
        ap += (recall-previous_recall)*tp/np.sum(selected)
        previous_recall = recall
    return dict(auroc=auc, average_precision=float(ap), prevalence=prevalence)


def paired_cluster_interval(labels, correct, wrong, sources, config=DEFAULT_CONFIG):
    validate_config(config)
    y, a, b, group = map(np.asarray, (labels, correct, wrong, sources))
    if not (y.shape == a.shape == b.shape == group.shape):
        raise ValueError('cluster arrays have mismatched identities')
    ranking_metrics(y, a)
    ranking_metrics(y, b)
    result = dict(n_sources=len(np.unique(group)), n_valid=0, correct_ci_low=None, correct_ci_high=None,
                  delta_ci_low=None, delta_ci_high=None)
    unique = sorted(set(group.tolist()))
    if len(unique) < config['bootstrap']['min_sources']:
        return result
    rng = np.random.Generator(np.random.PCG64(config['seed']))
    outcomes = []
    for _ in range(config['bootstrap']['repetitions']):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(group == value) for value in sample])
        aa = ranking_metrics(y[indices], a[indices])['auroc']
        bb = ranking_metrics(y[indices], b[indices])['auroc']
        if aa is not None and bb is not None:
            outcomes.append((aa, aa-bb))
    result['n_valid'] = len(outcomes)
    if len(outcomes) >= config['bootstrap']['min_valid']:
        limits = np.quantile(np.asarray(outcomes), [.025, .975], axis=0)
        result.update(correct_ci_low=float(limits[0, 0]), correct_ci_high=float(limits[1, 0]),
                      delta_ci_low=float(limits[0, 1]), delta_ci_high=float(limits[1, 1]))
    return result


def continuation_decision(alignment, review_complete, config=DEFAULT_CONFIG):
    validate_config(config)
    rule = config['continuation']
    usable = [bool(r['determinable_fraction'] >= rule['min_determinable']
                   and r['n_present'] >= rule['min_present'] and r['n_absent'] >= rule['min_absent']
                   and r['n_present_videos'] >= rule['min_label_videos'] and r['n_absent_videos'] >= rule['min_label_videos']
                   and r['correct_auroc'] is not None and r['correct_auroc'] >= rule['min_auroc']
                   and r['auroc_delta'] is not None
                   and (r['auroc_delta'] >= rule['min_auroc_delta']
                        or np.isclose(r['auroc_delta'], rule['min_auroc_delta'], atol=1e-12, rtol=0))) for r in alignment]
    return dict(n_usable_concepts=sum(usable), usable_concepts=usable,
                exploratory_continuation_gate=bool(len(usable) == 6 and sum(usable) >= rule['min_usable_concepts'] and review_complete))


def analyze_rows(mapping, scores, crop, original, secondary, adjudication, review_complete, config=DEFAULT_CONFIG):
    validate_config(config)
    identity = {r['opaque_id']: r for r in mapping}
    source_scores = {(r['opaque_id'], r['concept_id']): float(r['cosine_score']) for r in scores}
    primary = {(r['opaque_id'], r['concept_id'], r['view']): r for r in [*crop, *original]}
    effective = dict(primary)
    for row in adjudication:
        effective[row['opaque_id'], row['concept_id'], row['view']] = row
    coverage, retention, alignment, matched = [], [], [], []
    for cid in CONCEPTS:
        for view in ('crop', 'original'):
            part = [r for (frame, concept, v), r in effective.items() if concept == cid and v == view]
            counts = {label: sum(r['judgment'].strip() == label for r in part) for label in ('present', 'absent', 'uncertain', '')}
            n_judged = len(part)-counts['']
            coverage.append(dict(concept_id=cid, view=view, n_rows=len(part), n_present=counts['present'], n_absent=counts['absent'],
                n_uncertain=counts['uncertain'], n_blank=counts[''], n_judged=n_judged,
                determinable_fraction=(counts['present']+counts['absent'])/n_judged if n_judged else None,
                **{'uncertain_'+reason: sum(r['judgment'].strip() == 'uncertain' and r['reason_code'].strip() == reason for r in part)
                   for reason in ('crop_missing', 'occlusion', 'blur', 'ambiguous_person', 'ground_unknown', 'other')}))
        pairs = [(effective[frame, cid, 'original'], effective[frame, cid, 'crop']) for frame in identity]
        n_original_present = sum(a['judgment'].strip() == 'present' for a, b in pairs)
        retained = sum(a['judgment'].strip() == b['judgment'].strip() == 'present' for a, b in pairs)
        retention.append(dict(concept_id=cid, n_original_present=n_original_present, n_retained_present=retained,
            retention_rate=retained/n_original_present if n_original_present else None,
            original_present_crop_uncertain=sum(a['judgment'].strip() == 'present' and b['judgment'].strip() == 'uncertain' for a, b in pairs),
            original_present_crop_missing=sum(a['judgment'].strip() == 'present' and b['judgment'].strip() == 'uncertain' and b['reason_code'].strip() == 'crop_missing' for a, b in pairs),
            original_determinable_crop_uncertain=sum(a['judgment'].strip() in ('present', 'absent') and b['judgment'].strip() == 'uncertain' for a, b in pairs),
            **{f'{a}_to_{b}': sum(x['judgment'].strip() == a and y['judgment'].strip() == b for x, y in pairs)
               for a in ('present', 'absent', 'uncertain') for b in ('present', 'absent', 'uncertain')}))
        rows = [effective[frame, cid, 'crop'] for frame in identity]
        known = [r for r in rows if r['judgment'].strip() in ('present', 'absent')]
        wrong_cid = config['wrong_concept_ids'][list(CONCEPTS).index(cid)]
        for row in rows:
            frame = row['opaque_id']
            matched.append(dict(opaque_id=frame, concept_id=cid, judgment=row['judgment'].strip(),
                video_id=identity[frame]['video_id'], source_group_id=identity[frame]['source_group_id'],
                correct_score=source_scores[frame, cid], wrong_concept_id=wrong_cid, wrong_score=source_scores[frame, wrong_cid]))
        y = [int(r['judgment'].strip() == 'present') for r in known]
        a = [source_scores[r['opaque_id'], cid] for r in known]
        b = [source_scores[r['opaque_id'], wrong_cid] for r in known]
        correct, wrong = ranking_metrics(y, a), ranking_metrics(y, b)
        positive = [r for r in known if r['judgment'].strip() == 'present']
        negative = [r for r in known if r['judgment'].strip() == 'absent']
        entry = dict(concept_id=cid, wrong_concept_id=wrong_cid, n_present=len(positive), n_absent=len(negative),
            n_uncertain=len(rows)-len(known), determinable_fraction=len(known)/len(rows),
            n_present_videos=len({identity[r['opaque_id']]['video_id'] for r in positive}),
            n_absent_videos=len({identity[r['opaque_id']]['video_id'] for r in negative}),
            correct_auroc=correct['auroc'], wrong_auroc=wrong['auroc'],
            auroc_delta=correct['auroc']-wrong['auroc'] if correct['auroc'] is not None else None,
            correct_average_precision=correct['average_precision'], wrong_average_precision=wrong['average_precision'],
            prevalence=correct['prevalence'], present_score_median=float(np.median([source_scores[r['opaque_id'], cid] for r in positive])) if positive else None,
            absent_score_median=float(np.median([source_scores[r['opaque_id'], cid] for r in negative])) if negative else None)
        rule = config['continuation']
        entry['coverage_sufficient'] = bool(min(len(positive), len(negative)) >= 5 and
            min(entry['n_present_videos'], entry['n_absent_videos']) >= rule['min_label_videos'])
        entry.update(paired_cluster_interval(y, a, b, [identity[r['opaque_id']]['source_group_id'] for r in known], config))
        alignment.append(entry)
    disagreement = []
    for second in secondary:
        key = second['opaque_id'], second['concept_id'], second['view']
        first = primary[key]
        disagreement.append(dict(opaque_id=key[0], concept_id=key[1], view=key[2],
                                 first_judgment=first['judgment'].strip(), second_judgment=second['judgment'].strip(),
                                 agreed=first['judgment'].strip() == second['judgment'].strip()))
    decision = continuation_decision(alignment, review_complete, config)
    for row, usable in zip(alignment, decision['usable_concepts']):
        row['initially_usable'] = usable
    return dict(concept_coverage=coverage, crop_retention=retention, concept_alignment=alignment,
                matched_scores=matched, agreement_rows=disagreement,
                agreement_fraction=float(np.mean([r['agreed'] for r in disagreement])) if disagreement else None,
                **decision)
