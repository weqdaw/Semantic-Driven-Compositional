"""Scientific invariants of the instance-retention diagnostic."""
import copy
from pathlib import Path

import numpy as np
import pytest


def fixture_bank():
    # Class 0 has two e0 modes and one e1 mode; the other classes are constant.
    # q=e0 gives hand-computed cache scores .6680289 (instances) and
    # .5595343 (collapsed unit mean), with identical text score 1.
    labels = np.repeat(np.arange(5, dtype=np.int64), 3)
    keys = np.eye(5)[labels]
    keys[1] = np.eye(5)[1]
    return dict(q=np.vstack([keys, np.eye(5)]), T=np.eye(5)), list(range(15)), labels


def test_nonlinear_instance_retention_and_matched_collapsed_slots():
    from hmdb51_e0.e15_features import cache_model, score_branches

    bank, positions, labels = fixture_bank()
    query = np.eye(5)[[0]]
    instance = cache_model(bank, 'IC1', positions, labels)
    collapsed = cache_model(bank, 'IC2', positions, labels)
    final, text, cache, cosines, affinities, votes = score_branches(query, instance)
    other_final, other_text, other_cache, _, _, _ = score_branches(query, collapsed)

    np.testing.assert_array_equal(text, [[1., 0., 0., 0., 0.]])
    np.testing.assert_array_equal(other_text, text)
    np.testing.assert_allclose(cache, [[.6680289238128214, *([.004086771438464067] * 4)]], atol=1e-14)
    assert other_cache[0, 0] == pytest.approx(.5595342993690586, abs=1e-14)
    assert final[0, 0] == pytest.approx(1.0668028923812822, abs=1e-14)
    assert other_final[0, 0] == pytest.approx(1.0559534299369058, abs=1e-14)
    assert cosines.shape == affinities.shape == votes.shape == (1, 15)
    np.testing.assert_array_equal(cosines[0, :3], [1., 0., 1.])
    np.testing.assert_allclose(affinities[0, :3], [1., .004086771438464067, 1.], atol=1e-14)
    np.testing.assert_allclose(votes[0, :3], [1/3, .001362257146154689, 1/3], atol=1e-14)
    np.testing.assert_allclose(collapsed['cache_keys'][:3], np.tile([2/np.sqrt(5), 1/np.sqrt(5), 0, 0, 0], (3, 1)))
    assert instance['cache_keys'].shape == collapsed['cache_keys'].shape == (15, 5)
    np.testing.assert_array_equal(instance['cache_values'].sum(axis=0), [3.] * 5)
    np.testing.assert_array_equal(collapsed['cache_values'], instance['cache_values'])


def test_wrong_labels_only_permute_votes_and_leave_anchor_and_keys_unchanged():
    from hmdb51_e0.e15_features import cache_model, score_branches

    bank, positions, labels = fixture_bank()
    good = cache_model(bank, 'IC1', positions, labels)
    wrong = cache_model(bank, 'IC3', positions, labels)
    for name in ('text_prototypes', 'visual_prototypes', 'support_keys', 'cache_keys', 'true_labels'):
        np.testing.assert_array_equal(wrong[name], good[name])
    np.testing.assert_array_equal(wrong['cache_labels'], np.repeat([1, 2, 3, 4, 0], 3))
    _, good_text, good_cache, *good_evidence = score_branches(np.eye(5), good)
    final, wrong_text, wrong_cache, *wrong_evidence = score_branches(np.eye(5), wrong)
    np.testing.assert_array_equal(wrong_text, good_text)
    np.testing.assert_allclose(wrong_cache, good_cache[:, [4, 0, 1, 2, 3]], atol=1e-14)
    np.testing.assert_allclose(final, wrong_text + .1*wrong_cache, atol=1e-14)
    for a, b in zip(good_evidence, wrong_evidence):
        np.testing.assert_array_equal(a, b)


def test_cache_uses_only_selected_supports_and_is_invariant_to_slot_order():
    from hmdb51_e0.e15_features import cache_model, score_branches

    bank, positions, labels = fixture_bank()
    original = cache_model(bank, 'IC1', positions, labels)
    changed = {k: v.copy() for k, v in bank.items()}
    changed['q'][15:] *= -1
    other = cache_model(changed, 'IC1', positions, labels)
    for name in original:
        np.testing.assert_array_equal(original[name], other[name])
    order = np.arange(15)[::-1]
    reordered = cache_model(bank, 'IC1', order, labels[order])
    for a, b in zip(score_branches(np.eye(5), original)[:3], score_branches(np.eye(5), reordered)[:3]):
        np.testing.assert_allclose(a, b, atol=1e-14)
    assert not np.shares_memory(original['support_keys'], bank['q'])


def test_fixed_config_and_complete_condition_schedule():
    from hmdb51_e0.e15_features import DEFAULT_CONFIG, load_config, schedule, validate_config

    config = load_config(Path(__file__).parents[1] / 'configs/e15.instance-cache.yaml')
    assert config == DEFAULT_CONFIG
    assert schedule(config) == [[r, 3, seed] for r in ('IC1', 'IC2', 'IC3') for seed in range(10)]
    for field, bad in [('alpha', .2), ('beta', 1.), ('support_seeds', [0]), ('final_test_enabled', True), ('shots', [3.])]:
        mutated = copy.deepcopy(config)
        mutated[field] = bad
        with pytest.raises(ValueError):
            validate_config(mutated)


@pytest.mark.parametrize('mutation', ['duplicate', 'negative', 'floating_indices', 'floating_labels', 'unbalanced', 'unknown', 'nonunit', 'nan', 'bad_text', 'degenerate'])
def test_invalid_support_models_are_rejected(mutation):
    from hmdb51_e0.e15_features import cache_model

    bank, positions, labels = fixture_bank()
    rep = 'IC1'
    if mutation == 'duplicate': positions[1] = positions[0]
    elif mutation == 'negative': positions[0] = -1
    elif mutation == 'floating_indices': positions = np.asarray(positions, dtype=float)
    elif mutation == 'floating_labels': labels = labels.astype(float)
    elif mutation == 'unbalanced': labels[0] = 1
    elif mutation == 'unknown': rep = 'IC4'
    elif mutation == 'nonunit': bank['q'][0] *= 2
    elif mutation == 'nan': bank['q'][0, 0] = np.nan
    elif mutation == 'bad_text': bank['T'] = np.eye(4)
    elif mutation == 'degenerate':
        bank['q'][:3] = [[1, 0, 0, 0, 0], [-.5, np.sqrt(3)/2, 0, 0, 0], [-.5, -np.sqrt(3)/2, 0, 0, 0]]
    with pytest.raises(ValueError):
        cache_model(bank, rep, positions, labels)


def test_invalid_query_or_inconsistent_cache_values_cannot_be_scored():
    from hmdb51_e0.e15_features import cache_model, score_branches

    bank, positions, labels = fixture_bank()
    model = cache_model(bank, 'IC1', positions, labels)
    for query in (np.ones((1, 5)), np.full((1, 5), np.nan), np.eye(4), np.eye(5)[0], np.empty((0, 5))):
        with pytest.raises(ValueError):
            score_branches(query, model)
    bad = {k: v.copy() for k, v in model.items()}
    bad['cache_values'][0] = [0., 0., 1., 0., 0.]
    with pytest.raises(ValueError):
        score_branches(np.eye(5), bad)
