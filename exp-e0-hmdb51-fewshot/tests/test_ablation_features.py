import numpy as np
import pytest


def frames():
    return np.concatenate([np.tile([1., 0, 0], (6, 1)), np.tile([0., 1, 0], (5, 1)),
                           np.tile([0., 0, 1], (5, 1))]).astype(np.float32)


def test_repeat_preserves_global_statistics_and_repeats_mean_blocks():
    from hmdb51_e0.ablation_features import ablation_pool
    from hmdb51_e0.temporal_features import temporal_pool
    z = frames()
    vector = ablation_pool(z, 'V1_repeat')
    blocks = vector.reshape(5, 3)
    assert vector.dtype == np.float32 and vector.shape == (15,)
    assert np.linalg.norm(vector) == pytest.approx(1)
    np.testing.assert_array_equal(blocks[0], blocks[2])
    np.testing.assert_array_equal(blocks[0], blocks[3])
    np.testing.assert_array_equal(blocks[0], blocks[4])
    np.testing.assert_allclose(blocks[:2], temporal_pool(z, 'V1').reshape(5, 3)[:2], atol=1e-7)
    assert np.dot(blocks[1], blocks[1]) == pytest.approx(.2, abs=2e-6)
    np.testing.assert_allclose(vector, ablation_pool(z[::-1], 'V1_repeat'), atol=1e-7)


def test_shuffle_permutation_is_label_free_stable_and_replayable():
    from hmdb51_e0.ablation_features import ablation_pool, frame_permutation
    from hmdb51_e0.temporal_features import temporal_pool
    p = frame_permutation('jump/example.avi', 0)
    assert sorted(p.tolist()) == list(range(16))
    np.testing.assert_array_equal(p, frame_permutation('jump/example.avi', 0))
    assert not np.array_equal(p, frame_permutation('jump/other.avi', 0))
    assert not np.array_equal(p, frame_permutation('jump/example.avi', 1))
    z = frames()
    actual = ablation_pool(z, 'V1_shuffle', video_id='jump/example.avi', shuffle_seed=0)
    np.testing.assert_array_equal(actual, temporal_pool(z[p], 'V1'))
    assert not np.allclose(actual, temporal_pool(z, 'V1'))
    np.testing.assert_allclose(actual.reshape(5, 3)[:2], temporal_pool(z, 'V1').reshape(5, 3)[:2], atol=1e-7)


@pytest.mark.parametrize('mutation', ['short', 'nan', 'zero', 'unnormalized', 'condition', 'missing_seed', 'bool_seed', 'negative_seed', 'unsafe_id', 'repeat_seed'])
def test_ablation_rejects_invalid_frames_or_condition_identity(mutation):
    from hmdb51_e0.ablation_features import ablation_pool
    z = frames()
    args = dict(condition='V1_shuffle', video_id='jump/example.avi', shuffle_seed=0)
    if mutation == 'short': z = z[:15]
    elif mutation == 'nan': z[0, 0] = np.nan
    elif mutation == 'zero': z[0] = 0
    elif mutation == 'unnormalized': z *= 2
    elif mutation == 'condition': args['condition'] = 'bad'
    elif mutation == 'missing_seed': args['shuffle_seed'] = None
    elif mutation == 'bool_seed': args['shuffle_seed'] = True
    elif mutation == 'negative_seed': args['shuffle_seed'] = -1
    elif mutation == 'unsafe_id': args['video_id'] = '../test.avi'
    else: args['condition'] = 'V1_repeat'
    with pytest.raises(ValueError):
        ablation_pool(z, **args)
