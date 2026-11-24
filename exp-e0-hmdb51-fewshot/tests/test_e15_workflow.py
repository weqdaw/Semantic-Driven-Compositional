"""End-to-end identities, evidence reconstruction and production boundaries."""
import copy
import csv

import numpy as np
import pytest

from hmdb51_e0.provenance import read_json, write_json, write_csv, sha256_file
from test_round2_protocol import config2, parent
from test_round2_features import round2_inputs
from test_ablation_experiment import inputs
from test_e1_workflow import e1_data, prepare, SyntheticText


def make_e14(data, tmp_path):
    from hmdb51_e0.e1_experiment import run_e1
    from hmdb51_e0.e11_workflow import prepare_features, run_e11
    from hmdb51_e0.e11_features import DEFAULT_CONFIG as C11
    from hmdb51_e0.e12_workflow import prepare_e12, run_e12
    from hmdb51_e0.e12_features import DEFAULT_CONFIG as C12
    from hmdb51_e0.e13_workflow import prepare_e13, run_e13
    from hmdb51_e0.e13_features import DEFAULT_CONFIG as C13
    from hmdb51_e0.e14_workflow import prepare_e14, run_e14
    from hmdb51_e0.e14_features import DEFAULT_CONFIG as C14

    e1 = run_e1(data['reference'], prepare(data), data['paths'])
    e11 = run_e11(e1, prepare_features(e1, C11, data['paths'], tmp_path/'e11cache'), data['paths'])
    e12 = run_e12(e11, prepare_e12(e11, C12, data['paths'], tmp_path/'e12cache'), data['paths'])
    e13 = run_e13(e12, prepare_e13(e12, C13, data['paths'], tmp_path/'e13cache', text_encoder=SyntheticText()), data['paths'])
    return run_e14(e13, prepare_e14(e13, C14, data['paths'], tmp_path/'e14cache'), data['paths'])


def test_e15_complete_cache_experiment_and_rehashed_tamper_rejection(e1_data, tmp_path):
    # Breaks caught: fake T0 repetitions, wrong reference pairing, collapsed slot
    # mismatches, loss of per-key evidence, and trusting updated hashes alone.
    from hmdb51_e0.e15_features import DEFAULT_CONFIG
    from hmdb51_e0.e15_workflow import check_parent, prepare_e15, run_e15
    from hmdb51_e0.e15_reporting import summarize, validate_runs
    from hmdb51_e0.e11_workflow import tree_hashes

    e14 = make_e14(e1_data, tmp_path)
    before = tree_hashes(e14)
    check = check_parent(e14, DEFAULT_CONFIG)
    assert check['planned_new_evaluations'] == 30
    assert check['reused_reference_evaluations'] == 11
    assert check['gradient_fits_executed'] == 0
    assert check['text_encoding_executed'] is False
    index = prepare_e15(e14, DEFAULT_CONFIG, e1_data['paths'], tmp_path/'e15cache')
    folder = run_e15(e14, index, e1_data['paths'])
    result = summarize(folder, tmp_path/'e15report')
    assert tree_hashes(e14) == before
    assert result['new_evaluations'] == 30 and result['reused_reference_evaluations'] == 11
    assert len(result['summary']) == 5 and len(result['per_run_metrics']) == 41
    assert len(result['per_support_metrics']) == 40
    assert len(result['paired_deltas']) == 70 and len(result['paired_summary']) == 7
    assert len(result['prediction_transitions']) == 40
    assert len(result['transition_summary']) == 4
    zero = next(r for r in result['summary'] if r['representation'] == 'T0')
    assert zero['macro_f1_std'] is None and zero['n_support_sets'] == 0
    assert sum(r['representation'] == 'T0' for r in result['per_run_metrics']) == 1
    assert sum(r['reused_reference'] for r in result['per_run_metrics']) == 11
    for row in result['summary']:
        if row['representation'] == 'T0':
            continue
        values = [r['macro_f1'] for r in result['per_run_metrics'] if r['representation'] == row['representation']]
        assert row['macro_f1_mean'] == pytest.approx(np.mean(values))
        assert row['macro_f1_std'] == pytest.approx(np.std(values, ddof=1))
    by_key = {(r['representation'], r['support_seed']): r for r in result['per_support_metrics']}
    fixed_zero = next(r for r in result['per_run_metrics'] if r['representation'] == 'T0')
    for delta in result['paired_deltas']:
        a = by_key[delta['left'], delta['support_seed']]
        b = fixed_zero if delta['right'] == 'T0' else by_key[delta['right'], delta['support_seed']]
        assert delta['macro_f1_delta'] == pytest.approx(a['macro_f1'] - b['macro_f1'])
        if delta['right'] != 'T0':
            assert a['support_sha256'] == b['support_sha256']
    report = (tmp_path/'e15report/e15-results.md').read_text(encoding='utf-8')
    assert '仅程序验证' in report and '不是统计显著性' in report
    rows = list(csv.DictReader((tmp_path/'e15report/per_video_predictions.csv').open(encoding='utf-8', newline='')))
    assert len(rows) == 41 * result['n_development_videos']
    assert all(r['cache_score_clap'] == '' for r in rows if r['representation'] in ('T0', 'A2'))

    batch = read_json(folder/'batch.json')
    saved = copy.deepcopy(batch)
    batch['runs'].pop()
    write_json(folder/'batch.json', batch)
    with pytest.raises(ValueError, match='schedule|incomplete'):
        validate_runs(folder)
    write_json(folder/'batch.json', saved)
    run = folder/saved['runs'][0]['directory']
    provenance_path = run/'provenance.json'
    original_prov = read_json(provenance_path)
    for filename, array, match in (
        ('cache_model.npz', 'cache_labels', 'model|cache|reconstruction'),
        ('cache_evidence.npz', 'votes', 'evidence|vote|reconstruction'),
        ('cache_evidence.npz', 'support_ids', 'evidence|support|reconstruction'),
    ):
        path = run/filename
        clean_bytes = path.read_bytes()
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        if array == 'cache_labels':
            payload[array] = (payload[array] + 1) % 5
        elif array == 'support_ids':
            payload[array] = payload[array][::-1]
        else:
            payload[array] = payload[array] + .001
        np.savez_compressed(path, **payload)
        prov = copy.deepcopy(original_prov)
        prov['hashes'][filename] = sha256_file(path)
        write_json(provenance_path, prov)
        with pytest.raises(ValueError, match=match):
            validate_runs(folder)
        path.write_bytes(clean_bytes)
        write_json(provenance_path, original_prov)

    path = run/'predictions.csv'
    clean_bytes = path.read_bytes()
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        predictions = list(reader)
    predictions[0]['text_score_clap'] = str(float(predictions[0]['text_score_clap']) + .001)
    write_csv(path, predictions, fields)
    prov = copy.deepcopy(original_prov)
    prov['hashes']['predictions.csv'] = sha256_file(path)
    write_json(provenance_path, prov)
    with pytest.raises(ValueError, match='score|branch|anchor'):
        validate_runs(folder)
    path.write_bytes(clean_bytes)
    write_json(provenance_path, original_prov)


def test_e15_cli_guards_synthetic_and_final_test_before_output_creation(tmp_path, capsys):
    # Breaks caught: accidental formal execution on synthetic features or a
    # final-test entry point, even when input paths themselves are invalid.
    import yaml
    from hmdb51_e0.e15_cli import main, build_parser
    from hmdb51_e0.e15_workflow import run_e15
    parser = build_parser()
    for command in ('check', 'prepare', 'run', 'summarize'):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([command, '--help'])
        assert exc.value.code == 0
    with pytest.raises(SystemExit):
        parser.parse_args(['run', '--e14-batch', 'x', '--feature-index', 'x', '--stage', 'final_test'])
    with pytest.raises(ValueError, match='sealed'):
        run_e15(tmp_path/'absent', tmp_path/'absent', {}, stage='final_test')
    parent_dir = tmp_path/'parent'
    parent_dir.mkdir()
    write_json(parent_dir/'feature_index_snapshot.json', {'feature_condition': 'validation_only_unverified_features'})
    batch_dir = tmp_path/'batch'
    (batch_dir/'parent').mkdir(parents=True)
    write_json(batch_dir/'parent/feature_index_snapshot.json', {'feature_condition': 'validation_only_unverified_features'})
    paths = tmp_path/'paths.yaml'
    paths.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path/'data'), annotations_root=None, weights_file=None,
                     feature_cache_root=str(tmp_path/'old'), run_root=str(tmp_path/'runs'), report_root=str(tmp_path/'reports'))))
    for command in ('check', 'prepare', 'run', 'summarize'):
        args = [command, '--paths', str(paths)]
        args += ['--batch', str(batch_dir)] if command == 'summarize' else ['--e14-batch', str(parent_dir)]
        if command == 'run':
            args += ['--feature-index', str(tmp_path/'absent')]
        if command == 'prepare':
            args += ['--output', str(tmp_path/'prepared')]
        assert main(args) == 2 and 'official' in capsys.readouterr().err
    assert not (tmp_path/'runs').exists()
    assert not (tmp_path/'reports').exists()
    assert not (tmp_path/'prepared').exists()
