from pathlib import Path

import pytest
import yaml

from hmdb51_e0.provenance import read_json, write_csv
from test_round2_protocol import config2, parent, reviewed
from test_round2_features import round2_inputs


def test_round2_cli_exposes_only_explicit_development_steps():
    from hmdb51_e0.round2_cli import build_parser
    parser = build_parser()
    assert parser.parse_args(['run', '--manifest', 'm', '--support-sets', 's', '--feature-index', 'f']).stage == 'development'
    with pytest.raises(SystemExit):
        parser.parse_args(['run', '--manifest', 'm', '--support-sets', 's', '--feature-index', 'f', '--stage', 'final_test'])
    with pytest.raises(SystemExit) as result:
        parser.parse_args(['prepare-features', '--help'])
    assert result.value.code == 0


def local_paths(tmp_path):
    file = tmp_path / 'paths.yaml'
    file.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path / 'videos'), annotations_root=None, weights_file=None,
                                        feature_cache_root=str(tmp_path / 'features'), run_root=str(tmp_path / 'runs'),
                                        report_root=str(tmp_path / 'reports'))), encoding='utf-8')
    return file


def test_cli_audit_then_review_then_freeze(parent, config2, tmp_path, capsys):
    from hmdb51_e0.round2_cli import main
    from hmdb51_e0.source_groups import REVIEW_FIELDS
    paths = local_paths(tmp_path)
    output = tmp_path / 'source_audits'
    assert main(['audit-sources', '--parent-manifest', str(parent[0]), '--output', str(output), '--paths', str(paths)]) == 0
    review = Path(capsys.readouterr().out.strip())
    assert review.is_file()
    args = ['freeze-split', '--parent-manifest', str(parent[0]), '--source-review', str(review),
            '--output', str(tmp_path / 'grouped'), '--paths', str(paths)]
    assert main(args) == 2
    assert 'unresolved' in capsys.readouterr().err
    write_csv(review, reviewed(parent), REVIEW_FIELDS)
    assert main(args) == 0
    manifest = Path(capsys.readouterr().out.strip())
    assert read_json(manifest)['config'] == config2


def test_outputs_inside_dataset_and_legacy_cache_are_blocked(tmp_path):
    from hmdb51_e0.round2_io import safe_output
    paths = dict(dataset_root=tmp_path / 'raw', feature_cache_root=tmp_path / 'legacy_features')
    for output in (tmp_path / 'raw/results', tmp_path / 'legacy_features/round2', tmp_path):
        with pytest.raises(ValueError, match='protected'):
            safe_output(output, paths)
    assert safe_output(tmp_path / 'round2', paths) == (tmp_path / 'round2').resolve()


def test_production_round2_cli_blocks_synthetic_features(round2_inputs, tmp_path, capsys):
    from hmdb51_e0.round2_cli import main
    paths = tmp_path / 'paths.yaml'
    data = round2_inputs
    paths.write_text(yaml.safe_dump({**{key: str(value) for key, value in data['paths'].items()}, 'annotations_root': None, 'weights_file': None}), encoding='utf-8')
    assert main(['prepare-features', '--manifest', str(data['manifest']), '--parent-feature-index', str(data['parent_index']),
                 '--paths', str(paths), '--output', str(data['output'])]) == 2
    assert 'official' in capsys.readouterr().err
