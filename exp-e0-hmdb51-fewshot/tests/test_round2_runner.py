import csv

import numpy as np
import pytest

from hmdb51_e0.provenance import read_json, write_json, sha256_file
from test_round2_features import config2, parent, round2_inputs


@pytest.fixture
def ready_round2(round2_inputs):
    from hmdb51_e0.round2_features import prepare_round2_features
    data = round2_inputs
    data['index'] = prepare_round2_features(data['manifest'], data['parent_index'], data['config'], data['paths'], data['output'])
    return data


def run(data):
    from hmdb51_e0.round2_runner import run_round2
    return run_round2(data['manifest'], data['supports'], data['index'], data['config'], data['paths'], data['output'])


def test_round2_persists_complete_paired_runs_and_validated_report(ready_round2, tmp_path):
    from hmdb51_e0.round2_reporting import summarize_round2
    directory = run(ready_round2)
    batch = read_json(directory / 'batch.json')
    assert batch['status'] == 'success'
    assert len(batch['runs']) == 60
    assert set(r['representation'] for r in batch['runs']) == {'V0', 'V1'}
    assert batch['feature_condition'] == 'validation_only_unverified_features'
    for item in batch['runs']:
        path = directory / item['directory']
        provenance = read_json(path / 'provenance.json')
        assert provenance['n_support'] == 5 * item['k']
        assert provenance['n_support_sources'] <= provenance['n_support']
        assert provenance['feature_dimension'] == (12 if item['representation'] == 'V0' else 30)
        with (path / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(batch['evaluation_ids'])
        assert all(r['representation'] == item['representation'] and r['source_group_id'] for r in rows)
    output = tmp_path / 'summary'
    result = summarize_round2(directory, output, ready_round2['config'])
    assert len(result['summary']) == 6
    assert len(result['paired_deltas']) == 30
    assert len(result['paired_summary']) == 3
    assert '仅程序验证' in (output / 'round2-results.md').read_text(encoding='utf-8')
    assert '2560' in (output / 'round2-results.md').read_text(encoding='utf-8')
    for item in result['paired_summary']:
        values = [r['macro_f1_delta'] for r in result['paired_deltas'] if r['k'] == item['k']]
        assert item['macro_f1_delta_mean'] == pytest.approx(np.mean(values))
        assert item['macro_f1_delta_median'] == pytest.approx(np.median(values))
    second = run(ready_round2)
    for a, b in zip(batch['runs'], read_json(second / 'batch.json')['runs']):
        assert read_json(directory / a['directory'] / 'metrics.json') == read_json(second / b['directory'] / 'metrics.json')


@pytest.mark.parametrize('mutation', ['missing_pair', 'wrong_representation', 'predictions', 'support', 'feature_identity', 'group_stats'])
def test_round2_report_rejects_missing_or_mixed_or_corrupt_runs(ready_round2, tmp_path, mutation):
    from hmdb51_e0.round2_reporting import summarize_round2
    directory = run(ready_round2)
    batch = read_json(directory / 'batch.json')
    first = directory / batch['runs'][0]['directory']
    if mutation == 'missing_pair':
        batch['runs'].pop()
    elif mutation == 'wrong_representation':
        batch['runs'][0]['representation'] = 'V1'
    elif mutation == 'predictions':
        (first / 'predictions.csv').write_text('corrupt\n', encoding='utf-8')
    elif mutation == 'support':
        (first / 'support_set.csv').write_text('corrupt\n', encoding='utf-8')
    else:
        provenance = read_json(first / 'provenance.json')
        if mutation == 'feature_identity':
            provenance['feature_identity'] = 'wrong'
        else:
            provenance['n_support_sources'] += 10
        write_json(first / 'provenance.json', provenance)
    write_json(directory / 'batch.json', batch)
    with pytest.raises(ValueError):
        summarize_round2(directory, tmp_path / 'bad_report', ready_round2['config'])


def test_round2_runner_rejects_old_supports_and_sealed_stage(ready_round2):
    from hmdb51_e0.round2_runner import run_round2
    data = ready_round2
    with pytest.raises(ValueError, match='sealed|development'):
        run_round2(data['manifest'], data['supports'], data['index'], data['config'], data['paths'], data['output'], stage='final_test')
    support = read_json(data['supports'])
    support['protocol_id'] = 'old_protocol'
    write_json(data['supports'], support)
    with pytest.raises(ValueError, match='support'):
        run(data)


def test_synthetic_report_labels_csv_and_every_figure(ready_round2, tmp_path, monkeypatch):
    from hmdb51_e0.round2_reporting import summarize_round2
    from matplotlib.axes import Axes
    titles = []
    original = Axes.set_title
    def track(self, label, *args, **kwargs):
        if label:
            titles.append(label)
        return original(self, label, *args, **kwargs)
    monkeypatch.setattr(Axes, 'set_title', track)
    result = summarize_round2(run(ready_round2), tmp_path / 'labeled', ready_round2['config'])
    assert titles and all('PROGRAM VALIDATION' in title for title in titles)
    assert all('V0' in title or 'V1' in title for title in titles)
    assert all(row['feature_condition'] == 'validation_only_unverified_features' for row in result['summary'])
