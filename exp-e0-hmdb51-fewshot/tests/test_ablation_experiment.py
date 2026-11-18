import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from hmdb51_e0.provenance import digest, read_json, sha256_file, write_csv, write_json
from test_round2_protocol import config2, parent
from test_round2_features import round2_inputs
from test_round2_runner import ready_round2, run


@pytest.fixture
def inputs(round2_inputs, tmp_path):
    from hmdb51_e0.ablation_features import DEFAULT_CONFIG
    from hmdb51_e0.clip_features import pool_features
    from hmdb51_e0.feature_cache import FeatureCache
    from hmdb51_e0.round2_features import prepare_round2_features
    data = round2_inputs
    # A weak-signal bank yields different predictions across perturbations and
    # supports, so incorrect averaging cannot hide behind uniformly perfect F1.
    parent_index = read_json(data['parent_index'])
    parent_index['identity']['encoder']['fixture'] = 'weak_signal_ablation_validation'
    cache = FeatureCache(data['paths']['feature_cache_root'], parent_index['identity'])
    by_id = {r['video_id']: r for r in read_json(data['manifest'])['rows']}
    rng = np.random.default_rng(718)
    for entry in parent_index['entries']:
        raw = rng.normal(size=(16, 6))
        raw[:, by_id[entry['video_id']]['class_id']] += .35
        z, vector = pool_features(raw)
        cache.save(entry, z, vector)
    data['parent_index'] = cache.root / 'index-development.json'
    write_json(data['parent_index'], parent_index)
    data['index'] = prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output'])
    return dict(**data, reference=run(data), ablation_config=copy.deepcopy(DEFAULT_CONFIG),
                ablation_root=tmp_path / 'ablations')


def test_prepare_replays_all_permutations_and_blocks_corrupt_bank(inputs):
    from hmdb51_e0.ablation_experiment import prepare_ablation_features, load_ablation_features
    from hmdb51_e0.ablation_features import ablation_pool, frame_permutation
    from hmdb51_e0.round2_features import Round2Cache
    before = sha256_file(inputs['index'])
    file = prepare_ablation_features(inputs['reference'], inputs['ablation_config'], inputs['paths'],
                                    inputs['output'], inputs['ablation_root'])
    index, bank = load_ablation_features(file, inputs['reference'])
    assert index['status'] == 'success'
    assert bank['V1_repeat'].shape == (80, 30)
    assert bank['V1_shuffle'].shape == (5, 80, 30)
    assert bank['permutations'].shape == (5, 80, 16)
    assert sha256_file(inputs['index']) == before
    parent_index = read_json(inputs['index'])
    cache = Round2Cache(inputs['output'], parent_index['identity'])
    for i, entry in enumerate(parent_index['entries']):
        z, _, _ = cache.load(entry)
        np.testing.assert_array_equal(bank['V1_repeat'][i], ablation_pool(z, 'V1_repeat'))
        for j, seed in enumerate(index['identity']['config']['shuffle']['seeds']):
            np.testing.assert_array_equal(bank['permutations'][j, i], frame_permutation(entry['video_id'], seed))
            np.testing.assert_array_equal(bank['V1_shuffle'][j, i],
                                          ablation_pool(z, 'V1_shuffle', video_id=entry['video_id'], shuffle_seed=seed))
    # Both byte corruption and a rehashed but invalid permutation must be rejected.
    npz = file.parent / 'features.npz'
    original = npz.read_bytes()
    npz.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash|corrupt'):
        load_ablation_features(file, inputs['reference'])
    npz.write_bytes(original)
    bank['permutations'][0, 0] = np.arange(16)
    with npz.open('wb') as stream:
        np.savez_compressed(stream, **bank)
    index['npz_sha256'] = sha256_file(npz)
    write_json(file, index)
    with pytest.raises(ValueError, match='permutation'):
        load_ablation_features(file, inputs['reference'])


def test_full_ablation_report_recomputes_pairs_and_averages_shuffle_within_support(inputs, tmp_path, monkeypatch):
    from hmdb51_e0.ablation_experiment import prepare_ablation_features, run_ablations
    from hmdb51_e0.ablation_reporting import summarize_ablations, validate_ablation_runs
    from matplotlib.axes import Axes
    file = prepare_ablation_features(inputs['reference'], inputs['ablation_config'], inputs['paths'],
                                    inputs['output'], inputs['ablation_root'])
    directory = run_ablations(inputs['reference'], file, inputs['paths'])
    batch = read_json(directory / 'batch.json')
    assert batch['status'] == 'success' and len(batch['runs']) == 180
    assert sum(r['representation'] == 'V1_repeat' for r in batch['runs']) == 30
    assert {r['shuffle_seed'] for r in batch['runs'] if r['representation'] == 'V1_shuffle'} == set(range(5))
    titles = []
    original_title = Axes.set_title
    def track(self, label, *args, **kwargs):
        titles.append(label)
        return original_title(self, label, *args, **kwargs)
    monkeypatch.setattr(Axes, 'set_title', track)
    result = summarize_ablations(directory, tmp_path / 'summary')
    assert titles and all('PROGRAM VALIDATION' in title for title in titles)
    assert len(result['summary']) == 12
    assert len(result['paired_deltas']) == 120  # 4 comparisons x 3 K x 10 support seeds.
    assert len(result['paired_summary']) == 12
    assert len(result['per_run_metrics']) == 240  # 60 reused + 180 new, not 240 independent datasets.
    assert len({r['macro_f1'] for r in result['per_run_metrics']}) > 10
    assert any(r['macro_f1_std'] > .01 for r in result['summary'])
    for row in result['summary']:
        assert row['n_support_sets'] == 10
        if row['representation'] != 'V1_shuffle': continue
        by_support = [np.mean([r['macro_f1'] for r in result['per_run_metrics']
                              if r['representation'] == 'V1_shuffle' and r['k'] == row['k'] and r['support_seed'] == seed])
                      for seed in range(10)]
        assert row['macro_f1_mean'] == pytest.approx(np.mean(by_support))
        assert row['macro_f1_std'] == pytest.approx(np.std(by_support, ddof=1))
    indexed = {(r['representation'], r['k'], r['support_seed']): r for r in result['per_support_metrics']}
    for row in result['paired_deltas']:
        left = indexed[row['left'], row['k'], row['support_seed']]
        right = indexed[row['right'], row['k'], row['support_seed']]
        assert row['macro_f1_delta'] == pytest.approx(left['macro_f1'] - right['macro_f1'])
    assert (tmp_path / 'summary/permutation_predictions.csv').is_file()
    assert '仅程序验证' in (tmp_path / 'summary/ablation-results.md').read_text(encoding='utf-8')
    # Tamper one thing at a time, restoring each. No failure may be silently omitted.
    original_batch = copy.deepcopy(batch)
    batch['runs'].pop()
    write_json(directory / 'batch.json', batch)
    with pytest.raises(ValueError): validate_ablation_runs(directory)
    write_json(directory / 'batch.json', original_batch)
    first = directory / original_batch['runs'][0]['directory']
    provenance = read_json(first / 'provenance.json')
    wrong = copy.deepcopy(provenance)
    wrong['support_ids'] = wrong['support_ids'][::-1]
    write_json(first / 'provenance.json', wrong)
    with pytest.raises(ValueError): validate_ablation_runs(directory)
    write_json(first / 'provenance.json', provenance)
    prediction_file = first / 'predictions.csv'
    original_predictions = prediction_file.read_bytes()
    with prediction_file.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]['true_class_id'] = str((int(rows[0]['true_class_id']) + 1) % 5)
    write_csv(prediction_file, rows, fields)
    wrong = copy.deepcopy(provenance)
    wrong['predictions_sha256'] = sha256_file(prediction_file)
    write_json(first / 'provenance.json', wrong)
    with pytest.raises(ValueError, match='truth|prediction'):
        validate_ablation_runs(directory)
    prediction_file.write_bytes(original_predictions)
    write_json(first / 'provenance.json', provenance)
    validate_ablation_runs(directory)
    environment_file = directory / 'environment.json'
    original_environment = environment_file.read_bytes()
    environment_file.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='environment'):
        validate_ablation_runs(directory)
    environment_file.write_bytes(original_environment)
    validate_ablation_runs(directory)


def test_cli_rejects_synthetic_and_sealed_stage_and_check_creates_no_outputs(inputs, tmp_path, capsys):
    from hmdb51_e0.ablation_cli import build_parser, main
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(inputs['ablation_config']), encoding='utf-8')
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({**{k: str(v) for k, v in inputs['paths'].items()},
                                     'annotations_root': None, 'weights_file': None}), encoding='utf-8')
    with pytest.raises(SystemExit):
        build_parser().parse_args(['run', '--reference-batch', 'r', '--feature-index', 'f', '--stage', 'final_test'])
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(['prepare', '--help'])
    assert result.value.code == 0
    assert main(['check', '--reference-batch', str(inputs['reference']), '--config', str(config),
                 '--paths', str(paths), '--feature-root', str(inputs['output'])]) == 2
    assert 'official' in capsys.readouterr().err
    assert not inputs['ablation_root'].exists()


@pytest.mark.parametrize('mutation', ['duplicate', 'empty', 'bool', 'negative', 'algorithm', 'sealed', 'unknown'])
def test_config_rejects_ambiguous_or_changed_protocol(mutation):
    from hmdb51_e0.ablation_features import DEFAULT_CONFIG, validate_ablation_config
    config = copy.deepcopy(DEFAULT_CONFIG)
    if mutation == 'duplicate': config['shuffle']['seeds'] = [0, 0]
    elif mutation == 'empty': config['shuffle']['seeds'] = []
    elif mutation == 'bool': config['shuffle']['seeds'] = [True]
    elif mutation == 'negative': config['shuffle']['seeds'] = [-1]
    elif mutation == 'algorithm': config['shuffle']['algorithm'] = 'random'
    elif mutation == 'sealed': config['final_test_enabled'] = True
    else: config['shots'] = [10]
    with pytest.raises(ValueError): validate_ablation_config(config)
