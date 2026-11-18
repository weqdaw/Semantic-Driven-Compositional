import copy
from pathlib import Path

import numpy as np
import pytest

from hmdb51_e0.clip_features import pool_features, pool_normalized
from hmdb51_e0.feature_cache import FeatureCache
from hmdb51_e0.provenance import digest, read_json, write_json, sha256_file
from test_round2_protocol import config2, parent, reviewed


def test_v1_has_correct_ordered_segments_and_unchanged_v0():
    from hmdb51_e0.temporal_features import temporal_pool
    z = np.concatenate([np.tile([1., 0, 0], (6, 1)), np.tile([0., 1, 0], (5, 1)), np.tile([0., 0, 1], (5, 1))]).astype(np.float32)
    blocks = [z.astype(np.float64).mean(0), z.astype(np.float64).std(0), np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])]
    expected = np.concatenate([b / (np.linalg.norm(b) + 1e-6) for b in blocks])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(temporal_pool(z, 'V1'), expected, atol=1e-7)
    np.testing.assert_array_equal(temporal_pool(z, 'V0'), pool_normalized(z))
    np.testing.assert_allclose(temporal_pool(z[::-1], 'V0'), temporal_pool(z, 'V0'), atol=1e-7)
    assert not np.allclose(temporal_pool(z[::-1], 'V1'), temporal_pool(z, 'V1'))
    constant = np.tile(np.array([.6, .8], dtype=np.float32), (16, 1))
    assert np.count_nonzero(temporal_pool(constant, 'V1')[2:4]) == 0


@pytest.mark.parametrize('mutation', ['short', 'nan', 'zero', 'unnormalized', 'unknown'])
def test_temporal_pool_rejects_invalid_input(mutation):
    from hmdb51_e0.temporal_features import temporal_pool
    z = np.tile(np.array([.6, .8], dtype=np.float32), (16, 1))
    representation = 'V1'
    if mutation == 'short':
        z = z[:15]
    elif mutation == 'nan':
        z[0, 0] = np.nan
    elif mutation == 'zero':
        z[0] = 0
    elif mutation == 'unnormalized':
        z *= 2
    else:
        representation = 'V2'
    with pytest.raises(ValueError):
        temporal_pool(z, representation)


@pytest.fixture
def round2_inputs(parent, config2, tmp_path):
    from hmdb51_e0.round2_protocol import build_grouped_bundle, freeze_grouped_bundle, training_rows, write_round2_supports
    parent_path, parent_bundle = parent
    dataset = tmp_path / 'videos'
    for row in training_rows(parent_bundle):
        file = dataset / row['relative_path']
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(('synthetic-identity-' + row['video_id']).encode())
        row.update(sha256=sha256_file(file), bytes=file.stat().st_size)
    parent_bundle['protocol_id'] = digest({k: parent_bundle[k] for k in ('config', 'rows', 'annotation_hashes', 'decoder')})
    write_json(parent_path, parent_bundle)
    bundle = build_grouped_bundle(parent_bundle, reviewed(parent), config2)
    manifest = freeze_grouped_bundle(bundle, tmp_path / 'grouped')
    supports = write_round2_supports(bundle, config2, tmp_path / 'supports')
    root = Path(__file__).resolve().parents[1]
    identity = dict(protocol_id=parent_bundle['protocol_id'], encoder={'synthetic_test_fixture': True, 'dimension': 6},
                    decoder=parent_bundle['decoder'], video=parent_bundle['config']['video'], pooling=parent_bundle['config']['pooling'],
                    feature_code={name: sha256_file(root / 'src/hmdb51_e0' / name) for name in ('clip_features.py', 'video_io.py', 'feature_cache.py')})
    cache = FeatureCache(tmp_path / 'features', identity)
    entries = []
    rng = np.random.default_rng(40)
    for row in training_rows(parent_bundle):
        entry = dict(video_id=row['video_id'], video_sha256=row['sha256'], frame_indices=list(range(1, 32, 2)))
        raw = rng.normal(size=(16, 6))
        raw[:, row['class_id']] += 2
        z, v = pool_features(raw)
        cache.save(entry, z, v)
        entries.append(entry)
    index = cache.root / 'index-development.json'
    write_json(index, dict(status='success', evaluation_stage='development', identity=identity, entries=entries,
                          frozen_verification={'unchanged': True}))
    paths = dict(dataset_root=dataset, feature_cache_root=tmp_path / 'features', run_root=tmp_path / 'runs', report_root=tmp_path / 'reports')
    return dict(manifest=manifest, supports=supports, parent_index=index, paths=paths, output=tmp_path / 'round2_features', config=config2)


def test_import_reuses_verified_z_without_encoder_and_preserves_parent(round2_inputs, monkeypatch):
    from hmdb51_e0.round2_features import prepare_round2_features, load_round2_features
    from hmdb51_e0 import clip_features
    data = round2_inputs
    monkeypatch.setattr(clip_features, 'FrozenCLIP', lambda *a, **k: pytest.fail('must not instantiate an encoder'))
    before = sha256_file(data['parent_index'])
    index = prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output'])
    vectors, identity = load_round2_features(index, read_json(data['manifest']), data['output'])
    assert sha256_file(data['parent_index']) == before
    assert set(vectors) == {'V0', 'V1'}
    assert len(vectors['V0']) == len(vectors['V1']) == 80
    assert identity['parent_index_sha256'] == before
    vid = next(iter(vectors['V0']))
    parent_index = read_json(data['parent_index'])
    entry = next(e for e in parent_index['entries'] if e['video_id'] == vid)
    original_z, original_v = FeatureCache(data['paths']['feature_cache_root'], parent_index['identity']).load(entry)
    np.testing.assert_array_equal(vectors['V0'][vid], original_v)
    assert vectors['V1'][vid].shape == (30,)
    assert np.isclose(np.linalg.norm(vectors['V1'][vid]), 1)
    assert prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output']) == index


@pytest.mark.parametrize('mutation', ['missing', 'sealed', 'freeze', 'decoder', 'pooling', 'sha', 'code', 'corrupt', 'raw_video'])
def test_import_blocks_invalid_parent_or_input(round2_inputs, mutation):
    from hmdb51_e0.round2_features import prepare_round2_features
    data = round2_inputs
    index = read_json(data['parent_index'])
    if mutation == 'missing':
        index['entries'].pop()
    elif mutation == 'sealed':
        index['evaluation_stage'] = 'final_test'
    elif mutation == 'freeze':
        index['frozen_verification']['unchanged'] = False
    elif mutation == 'decoder':
        index['identity']['decoder'] = {}
    elif mutation == 'pooling':
        index['identity']['pooling'] = {}
    elif mutation == 'sha':
        index['entries'][0]['video_sha256'] = 'f' * 64
    elif mutation == 'code':
        index['identity']['feature_code']['clip_features.py'] = 'f' * 64
    elif mutation == 'corrupt':
        FeatureCache(data['paths']['feature_cache_root'], index['identity']).paths(index['entries'][0])[0].write_bytes(b'broken')
    else:
        row = next(r for r in read_json(data['manifest'])['rows'] if r['official_tag'] == 1)
        (data['paths']['dataset_root'] / row['relative_path']).write_bytes(b'changed')
    write_json(data['parent_index'], index)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output'])


def test_derived_cache_corruption_is_rejected(round2_inputs):
    from hmdb51_e0.round2_features import prepare_round2_features, load_round2_features
    data = round2_inputs
    index = prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output'])
    next(index.parent.glob('*.npz')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt|hash'):
        load_round2_features(index, read_json(data['manifest']), data['output'])


def test_interrupted_import_resumes_verified_entries(round2_inputs, monkeypatch):
    from hmdb51_e0.round2_features import Round2Cache, prepare_round2_features, load_round2_features
    data = round2_inputs
    save = Round2Cache.save
    calls = []
    def transient(self, entry, z):
        calls.append(entry['video_id'])
        if len(calls) == 2:
            raise OSError('transient test I/O failure')
        return save(self, entry, z)
    monkeypatch.setattr(Round2Cache, 'save', transient)
    with pytest.raises(OSError, match='transient'):
        prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output'])
    index = next(data['output'].glob('*/index-development.json'))
    assert read_json(index)['status'] == 'failed'
    monkeypatch.setattr(Round2Cache, 'save', save)
    assert prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output']) == index
    assert read_json(index)['status'] == 'success'
    assert list(index.parent.glob('previous-attempt-*.json'))
    load_round2_features(index, read_json(data['manifest']), data['output'])


@pytest.mark.parametrize('mutation', ['before', 'after', 'encoder_before', 'truthy', 'errors', 'stage', 'failed'])
def test_official_frozen_record_requires_consistent_hashes(mutation):
    from hmdb51_e0.round2_features import validate_frozen_index
    from hmdb51_e0.clip_features import CHECKPOINT_SHA256
    encoder = dict(implementation='openai/CLIP', architecture='ViT-B/32', checkpoint_sha256=CHECKPOINT_SHA256,
                   parameter_sha256_before='b' * 64)
    index = dict(status='success', errors=[], evaluation_stage='development', identity={'encoder': encoder},
                 frozen_verification=dict(before='b' * 64, after='b' * 64, unchanged=True))
    validate_frozen_index(index)
    if mutation == 'before':
        del index['frozen_verification']['before']
    elif mutation == 'after':
        index['frozen_verification']['after'] = 'c' * 64
    elif mutation == 'encoder_before':
        del encoder['parameter_sha256_before']
    elif mutation == 'truthy':
        index['frozen_verification']['unchanged'] = 1
    elif mutation == 'errors':
        index['errors'] = ['unexpected error']
    elif mutation == 'stage':
        index['evaluation_stage'] = 'final_test'
    else:
        index['status'] = 'failed'
    with pytest.raises(ValueError, match='freeze|frozen'):
        validate_frozen_index(index)
