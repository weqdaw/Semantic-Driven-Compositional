import copy
from pathlib import Path

import pytest
import yaml

from hmdb51_e0.pipeline import freeze_manifest
from hmdb51_e0.provenance import digest, read_json, write_json


@pytest.fixture
def config2():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/e0.round2.example.yaml').read_text(encoding='utf-8'))


@pytest.fixture
def parent(tmp_path, config, eligible_rows):
    rows = copy.deepcopy(eligible_rows)
    for row in rows:
        if row['official_tag'] == 1:
            number = int(Path(row['video_id']).stem)
            vid = f"{row['class_name']}/film{number // 2:02}_{row['class_name']}_u_nm_np1_fr_goo_{number % 2}.avi"
            row.update(video_id=vid, relative_path=vid)
    audit = dict(status='success', records=rows, errors=[], annotation_hashes={'a': 'b'}, decoder={'backend': 'fixture'})
    path = freeze_manifest(audit, config, tmp_path / 'parent')
    return path, read_json(path)


def reviewed(parent):
    from hmdb51_e0.source_groups import candidate_records
    records = candidate_records(parent[1])
    for row in records:
        row.update(verification_status='conservative', evidence='Synthetic fixture source map',
                   reviewer='test', reviewed_at='2026-11-16T20:00:00+08:00')
    return records


def test_candidates_are_cross_class_and_unreviewed(parent):
    from hmdb51_e0.source_groups import candidate_records, candidate_source
    rows = candidate_records(parent[1])
    assert len(rows) == 80
    assert {r['verification_status'] for r in rows} == {'unresolved'}
    assert len({r['source_group_id'] for r in rows}) == 8
    assert candidate_source('clap/#FILM_clap_u_nm_np1_fr_goo_1.avi') == candidate_source('walk/film_walk_f_nm_np1_ba_med_9.avi')
    assert candidate_source('clap/unknown.avi') == ''


@pytest.mark.parametrize('video_id, expected', [
    ('jump/likebeckam_jump_l_cm_np1_ri_bad_17.avi', 'likebeckam'),
    ('run/Crash_run_l_cm_np1_fr_med_21.avi', 'crash'),
])
def test_source_candidates_accept_observed_official_l_suffix(video_id, expected):
    from hmdb51_e0.source_groups import candidate_source
    assert candidate_source(video_id) == expected


@pytest.mark.parametrize('mutation', ['unresolved', 'missing', 'duplicate', 'evidence', 'sealed', 'split_candidate', 'sha'])
def test_review_rejects_unreviewed_or_inconsistent_mapping(parent, mutation):
    from hmdb51_e0.source_groups import validate_review
    rows = reviewed(parent)
    if mutation == 'unresolved':
        rows[0]['verification_status'] = 'unresolved'
    elif mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(dict(rows[0]))
    elif mutation == 'evidence':
        rows[0]['evidence'] = ''
    elif mutation == 'sealed':
        rows[0]['video_id'] = next(r['video_id'] for r in parent[1]['rows'] if r['official_tag'] == 2)
    elif mutation == 'split_candidate':
        rows[0]['source_group_id'] = 'different_group'
    else:
        rows[0]['video_sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        validate_review(rows, parent[1])


def test_group_split_is_deterministic_global_and_preserves_sealed(parent, config2, tmp_path):
    from hmdb51_e0.round2_protocol import build_grouped_bundle, validate_grouped_bundle, sample_supports
    first = build_grouped_bundle(parent[1], reviewed(parent), config2)
    second = build_grouped_bundle(parent[1], reviewed(parent), config2)
    assert first['protocol_id'] == second['protocol_id']
    validate_grouped_bundle(first, config2)
    groups = {}
    for row, original in zip(first['rows'], parent[1]['rows']):
        if original['official_tag'] != 1:
            assert row == original
        else:
            groups.setdefault(row['source_group_id'], set()).add(row['experimental_role'])
    assert all(len(roles) == 1 for roles in groups.values())
    assert {next(iter(roles)) for roles in groups.values()} == {'support_pool', 'development'}
    supports = sample_supports(first, config2)
    assert len(supports['records']) == 10 * 5 * (1 + 3 + 5)
    assert supports['source_coverage']
    for seed in range(10):
        for cls in range(5):
            ids = lambda k: [r['video_id'] for r in supports['records'] if r['support_seed'] == seed and r['class_id'] == cls and r['k'] == k]
            assert ids(1) == ids(5)[:1]
            assert ids(3) == ids(5)[:3]


def test_group_split_accepts_approximate_counts_without_splitting(parent, config2):
    from hmdb51_e0.round2_protocol import build_grouped_bundle
    records = reviewed(parent)
    for row in records:
        group_number = int(row['candidate_source_id'].removeprefix('film'))
        row['source_group_id'] = 'large' if group_number < 4 else 'small'
    bundle = build_grouped_bundle(parent[1], records, config2)
    counts = [sum(r['class_id'] == cls and r['experimental_role'] == 'development' for r in bundle['rows']) for cls in range(5)]
    assert counts == [8] * 5


def test_infeasible_grouping_and_tampered_bundle_fail(parent, config2):
    from hmdb51_e0.round2_protocol import build_grouped_bundle, validate_grouped_bundle
    records = reviewed(parent)
    for row in records:
        row['source_group_id'] = 'only_one_source'
    with pytest.raises(ValueError, match='feasible|source|group'):
        build_grouped_bundle(parent[1], records, config2)
    bundle = build_grouped_bundle(parent[1], reviewed(parent), config2)
    bundle['rows'][0]['experimental_role'] = 'development'
    bundle['protocol_id'] = digest({k: v for k, v in bundle.items() if k != 'protocol_id'})
    with pytest.raises(ValueError):
        validate_grouped_bundle(bundle, config2)


@pytest.mark.parametrize('placeholder', ['unknown', 'unresolved', 'tbd', 'none', 'pending'])
def test_review_blocks_unassigned_placeholder_group_ids(parent, placeholder):
    from hmdb51_e0.source_groups import validate_review
    rows = reviewed(parent)
    for row in rows:
        if row['candidate_source_id'] == rows[0]['candidate_source_id']:
            row['source_group_id'] = placeholder
    with pytest.raises(ValueError, match='placeholder|unassigned'):
        validate_review(rows, parent[1])


def test_whole_group_reassignment_cannot_claim_fixed_seed(parent, config2):
    from hmdb51_e0.round2_protocol import build_grouped_bundle, validate_grouped_bundle
    bundle = build_grouped_bundle(parent[1], reviewed(parent), config2)
    support = next(r['source_group_id'] for r in bundle['rows'] if r['experimental_role'] == 'support_pool')
    development = next(r['source_group_id'] for r in bundle['rows'] if r['experimental_role'] == 'development')
    for row in bundle['rows']:
        if row.get('source_group_id') == support:
            row['experimental_role'] = 'development'
        elif row.get('source_group_id') == development:
            row['experimental_role'] = 'support_pool'
    bundle['protocol_id'] = digest({k: v for k, v in bundle.items() if k != 'protocol_id'})
    with pytest.raises(ValueError, match='deterministic|seed'):
        validate_grouped_bundle(bundle, config2)
