import numpy as np
import pytest


def test_ranking_uses_half_credit_for_ties_and_threshold_grouped_average_precision():
    from hmdb51_e0.e20_metrics import ranking_metrics
    # Two positive/negative tie pairs produce AUC .5; tied groups must not be
    # resolved by arbitrary CSV ordering. AP=(.5*.5)+(.5*.5)=.5.
    result = ranking_metrics([1, 0, 1, 0], [.9, .9, .1, .1])
    assert result == dict(auroc=.5, average_precision=.5, prevalence=.5)
    assert ranking_metrics([0, 1, 0, 1], [.9, .9, .1, .1]) == result
    # A positive ranked last: AP=1/3; one-class inputs stay unassessable.
    result = ranking_metrics([0, 0, 1], [.9, .8, .1])
    assert result['auroc'] == 0 and result['average_precision'] == pytest.approx(1/3)
    assert ranking_metrics([1, 1], [1., 0.])['auroc'] is None
    assert ranking_metrics([], [])['average_precision'] is None
    with pytest.raises(ValueError):
        ranking_metrics([1, 2], [1., 0.])


def test_source_cluster_bootstrap_is_paired_and_suppresses_sparse_intervals():
    from hmdb51_e0.e20_metrics import paired_cluster_interval
    from hmdb51_e0.e20_protocol import DEFAULT_CONFIG
    labels = np.tile([1, 0], 5)
    score = np.tile([1., 0.], 5)
    source = np.repeat(np.arange(5).astype(str), 2)
    result = paired_cluster_interval(labels, score, score, source, DEFAULT_CONFIG)
    assert result['n_valid'] == 1000
    assert result['delta_ci_low'] == result['delta_ci_high'] == 0
    assert result['correct_ci_low'] == result['correct_ci_high'] == 1
    assert paired_cluster_interval(labels[:8], score[:8], score[:8], source[:8], DEFAULT_CONFIG)['delta_ci_low'] is None


def test_alignment_uses_crop_truth_and_keeps_uncertain_in_retention_denominators():
    from hmdb51_e0.e20_metrics import analyze_rows
    from hmdb51_e0.e20_protocol import CONCEPTS
    mapping = [dict(opaque_id=f'f{i}', video_id=f'v{i}', source_group_id=f'g{i}') for i in range(3)]
    score = [dict(opaque_id=f'f{i}', concept_id=c, cosine_score=value)
             for i, value in enumerate((.9, .1, 100.)) for c in CONCEPTS]
    crop, original = [], []
    for i in range(3):
        for cid in CONCEPTS:
            crop.append(dict(opaque_id=f'f{i}', concept_id=cid, view='crop',
                             judgment=('present', 'absent', 'uncertain')[i], reason_code='crop_missing' if i == 2 else 'clear'))
            original.append(dict(opaque_id=f'f{i}', concept_id=cid, view='original',
                                 judgment=('absent', 'present', 'present')[i], reason_code='clear'))
    result = analyze_rows(mapping, score, crop, original, [], [], False)
    for row in result['concept_alignment']:
        assert row['correct_auroc'] == 1 and row['correct_average_precision'] == 1
        assert row['n_uncertain'] == 1 and row['n_present'] == row['n_absent'] == 1
        assert row['coverage_sufficient'] is False
    for row in result['crop_retention']:
        assert row['n_original_present'] == 2 and row['n_retained_present'] == 0
        assert row['original_present_crop_missing'] == 1 and row['retention_rate'] == 0
    assert result['exploratory_continuation_gate'] is False


@pytest.mark.parametrize('coverage,present_videos,auc,delta,review,expected', [
    (.7, 3, .7, .05, True, True),
    (.7, 3, .7, .7-.65, True, True),
    (.69, 3, .8, .1, True, False),
    (.8, 2, .8, .1, True, False),
    (.8, 3, .69, .1, True, False),
    (.8, 3, .8, .049, True, False),
    (.8, 3, .8, .1, False, False),
])
def test_advance_requires_three_usable_concepts_and_completed_review(coverage, present_videos, auc, delta, review, expected):
    from hmdb51_e0.e20_metrics import continuation_decision
    from hmdb51_e0.e20_protocol import DEFAULT_CONFIG
    row = dict(determinable_fraction=coverage, n_present=5, n_absent=5,
               n_present_videos=present_videos, n_absent_videos=3, correct_auroc=auc, auroc_delta=delta)
    result = continuation_decision([row.copy() for _ in range(6)], review, DEFAULT_CONFIG)
    assert result['exploratory_continuation_gate'] is expected
