"""Separate grouped protocol; no mutation of the first-round E0 contract."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import yaml

from .config import CLASSES, ROOT, validate_config
from .manifest import validate_manifest, MANIFEST_FIELDS
from .provenance import digest, read_json, write_json, write_csv, new_directory
from .source_groups import training_rows, validate_review, REVIEW_FIELDS

FIXED = dict(
    experiment_id='e0_hmdb51_round2', protocol_version='0.2', classes=list(CLASSES),
    source_grouping=dict(unit='original_film_or_upload', review_required=True, unresolved_policy='block', same_candidate_policy='keep_together'),
    sampling=dict(shots=[1, 3, 5], support_seeds=list(range(10)), nested_support_sets=True, unit='video', rng='numpy_generator_pcg64'),
    representations=dict(V0=['mean', 'population_std'], V1=['mean', 'population_std', 'early_mean', 'middle_mean', 'late_mean'],
                         temporal_segments=[6, 5, 5], block_l2_normalize=True, final_l2_normalize=True, epsilon=1e-6),
    classifier=dict(family='multioutput_ridge', **{'lambda': 1.0}, fit_intercept=True, penalize_intercept=False),
    evaluation=dict(current_target='development', final_test_enabled=False, metrics=['accuracy', 'macro_f1', 'per_class_recall', 'confusion_matrix']))
SUPPORT_FIELDS = ['protocol_id', 'support_seed', 'k', 'class_id', 'video_id', 'rank_in_class', 'source_group_id']


def validate_round2_config(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != set(FIXED) | {'development'}:
        raise ValueError('invalid round2 configuration fields')
    for key, value in FIXED.items():
        if config[key] != value:
            raise ValueError(f'unsupported round2 setting: {key}')
    development = config['development']
    keys = {'target_videos_per_class', 'min_videos_per_class', 'min_support_videos_per_class', 'seed', 'search_trials'}
    if not isinstance(development, dict) or set(development) != keys or any(type(v) is not int for v in development.values()):
        raise ValueError('development counts/seed/search_trials must be integers')
    if (development['min_videos_per_class'] < 1 or development['target_videos_per_class'] < development['min_videos_per_class']
            or development['min_support_videos_per_class'] < 5 or development['seed'] < 0
            or not 1 <= development['search_trials'] <= 10000):
        raise ValueError('invalid round2 development constraints')


def load_round2_config(path: Path = ROOT / 'configs/e0.round2.example.yaml') -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_round2_config(config)
    return config


def validate_parent(parent: dict) -> None:
    validate_config(parent['config'])
    payload = {key: parent[key] for key in ('config', 'rows', 'annotation_hashes', 'decoder')}
    if parent.get('status') != 'success' or digest(payload) != parent['protocol_id']:
        raise ValueError('parent E0 manifest identity mismatch')
    validate_manifest(parent['rows'], parent['config'])


def load_parent(path: Path) -> dict:
    parent = read_json(path)
    validate_parent(parent)
    return parent


def _development_groups(rows: list[dict], config: dict) -> set[str]:
    histogram = defaultdict(lambda: [0] * 5)
    for row in rows:
        histogram[row['source_group_id']][row['class_id']] += 1
    groups = tuple(sorted(histogram))
    counts = tuple(tuple(histogram[group]) for group in groups)
    settings = tuple(config['development'][key] for key in ('target_videos_per_class', 'min_videos_per_class',
                                                          'min_support_videos_per_class', 'seed', 'search_trials'))
    return set(_select_groups(groups, counts, settings))


@lru_cache(maxsize=16)
def _select_groups(groups: tuple, counts: tuple, settings: tuple) -> tuple:
    counts = np.asarray(counts, dtype=np.int64)
    total = counts.sum(axis=0)
    target, minimum, support_min, seed, trials = settings
    if np.any(total < minimum + support_min) or np.any((counts > 0).sum(axis=0) < 2):
        raise ValueError('no feasible source-group holdout: too few videos or source groups')
    objective = lambda n: (int(np.maximum(minimum - n, 0).sum()), int(np.abs(n - target).sum()))
    rng = np.random.Generator(np.random.PCG64(seed))
    best = None
    for _ in range(trials):
        selected = np.zeros(len(groups), dtype=bool)
        n = np.zeros(5, dtype=np.int64)
        # Randomized, count-only local search; never inspects features or model scores.
        while True:
            changed = False
            for index in rng.permutation(len(groups)):
                candidate = n + (-counts[index] if selected[index] else counts[index])
                if np.all(total - candidate >= support_min) and objective(candidate) < objective(n):
                    selected[index] = not selected[index]
                    n = candidate
                    changed = True
            if not changed:
                break
        if np.all(n >= minimum):
            chosen = tuple(group for group, flag in zip(groups, selected) if flag)
            score = (*objective(n), chosen)
            if best is None or score < best[0]:
                best = (score, chosen)
    if best is None:
        raise ValueError('no feasible group split found by the fixed search; revise grouping/constraints before fitting')
    return best[1]


def group_statistics(rows: list[dict]) -> dict:
    output = {}
    for name in CLASSES:
        output[name] = {}
        for role in ('support_pool', 'development'):
            part = [r for r in rows if r['class_name'] == name and r['experimental_role'] == role]
            output[name][role] = dict(videos=len(part), source_groups=len({r['source_group_id'] for r in part}))
    return output


def build_grouped_bundle(parent: dict, review: list[dict], config: dict) -> dict:
    validate_round2_config(config)
    validate_parent(parent)
    review = validate_review(review, parent)
    reviewed = {r['video_id']: r for r in review}
    rows = []
    for original in parent['rows']:
        row = dict(original)
        if row['video_id'] in reviewed:
            record = reviewed[row['video_id']]
            row.update(source_group_id=record['source_group_id'], source_verification_status=record['verification_status'])
        rows.append(row)
    selected = _development_groups([r for r in rows if r['video_id'] in reviewed], config)
    for row in rows:
        if row['video_id'] in reviewed:
            row['experimental_role'] = 'development' if row['source_group_id'] in selected else 'support_pool'
    payload = dict(kind='hmdb51_e0_round2_grouped_manifest', config=config, parent_manifest=parent,
                   source_review=review, source_review_sha256=digest(review), rows=rows,
                   split_algorithm='count_only_randomized_group_local_search_v1')
    bundle = {**payload, 'protocol_id': digest(payload)}
    validate_grouped_bundle(bundle, config)
    return bundle


def validate_grouped_bundle(bundle: dict, config: dict) -> None:
    validate_round2_config(config)
    if (bundle.get('kind') != 'hmdb51_e0_round2_grouped_manifest' or bundle['config'] != config
            or digest({k: v for k, v in bundle.items() if k != 'protocol_id'}) != bundle['protocol_id']):
        raise ValueError('grouped manifest identity/config mismatch')
    parent = bundle['parent_manifest']
    validate_parent(parent)
    review = validate_review(bundle['source_review'], parent)
    if digest(review) != bundle['source_review_sha256']:
        raise ValueError('source review hash mismatch')
    by_id = {r['video_id']: r for r in review}
    if len(bundle['rows']) != len(parent['rows']):
        raise ValueError('grouped manifest coverage mismatch')
    roles = defaultdict(set)
    for row, original in zip(bundle['rows'], parent['rows']):
        record = by_id.get(original['video_id'])
        if record is None:
            if row != original:
                raise ValueError('sealed/unused/excluded parent record changed')
            continue
        expected = {**original, 'experimental_role': row['experimental_role'],
                    'source_group_id': record['source_group_id'], 'source_verification_status': record['verification_status']}
        if row != expected or row['experimental_role'] not in ('support_pool', 'development'):
            raise ValueError('grouped training identity changed')
        roles[row['source_group_id']].add(row['experimental_role'])
    if any(len(value) != 1 for value in roles.values()):
        raise ValueError('source group crosses support/development roles')
    stats = group_statistics(bundle['rows'])
    for name in CLASSES:
        if (stats[name]['development']['videos'] < config['development']['min_videos_per_class']
                or stats[name]['support_pool']['videos'] < config['development']['min_support_videos_per_class']):
            raise ValueError('grouped split does not meet per-class constraints')
    selected = {r['source_group_id'] for r in bundle['rows'] if r['experimental_role'] == 'development'}
    if selected != _development_groups(training_rows(bundle), config):
        raise ValueError('grouped roles differ from deterministic split/seed policy')


def freeze_grouped_bundle(bundle: dict, output: Path) -> Path:
    validate_grouped_bundle(bundle, bundle['config'])
    directory = new_directory(output, 'round2-grouped-manifest')
    write_json(directory / 'bundle.json', bundle)
    write_csv(directory / 'video_manifest.csv', bundle['rows'], MANIFEST_FIELDS + ['source_group_id', 'source_verification_status'])
    write_csv(directory / 'source_review_snapshot.csv', bundle['source_review'], REVIEW_FIELDS)
    write_json(directory / 'source_group_audit.json', dict(
        protocol_id=bundle['protocol_id'], group_statistics=group_statistics(bundle['rows']),
        review_status_counts=dict(Counter(r['verification_status'] for r in bundle['source_review'])),
        support_development_group_overlap=[], final_test_enabled=False,
        interpretation='Disjoint reviewed source IDs; conservative mappings do not prove real-world source independence.'))
    return directory / 'bundle.json'


def load_grouped_bundle(path: Path, config: dict) -> dict:
    bundle = read_json(path)
    validate_grouped_bundle(bundle, config)
    return bundle


def sample_supports(bundle: dict, config: dict) -> dict:
    validate_grouped_bundle(bundle, config)
    records, coverage = [], []
    for seed in config['sampling']['support_seeds']:
        rng = np.random.Generator(np.random.PCG64(seed))
        for class_id, name in enumerate(CLASSES):
            pool = sorted((r for r in bundle['rows'] if r['class_name'] == name and r['experimental_role'] == 'support_pool'), key=lambda r: r['video_id'])
            permuted = [pool[int(i)] for i in rng.permutation(len(pool))]
            for k in config['sampling']['shots']:
                part = permuted[:k]
                if len(part) != k:
                    raise ValueError('insufficient support videos for fixed K grid')
                records.extend(dict(protocol_id=bundle['protocol_id'], support_seed=seed, k=k, class_id=class_id,
                                    video_id=r['video_id'], rank_in_class=rank, source_group_id=r['source_group_id'])
                               for rank, r in enumerate(part, 1))
                coverage.append(dict(support_seed=seed, k=k, class_name=name, n_videos=k,
                                     n_source_groups=len({r['source_group_id'] for r in part})))
    payload = dict(protocol_id=bundle['protocol_id'], records=records, source_coverage=coverage)
    return {**payload, 'sha256': digest(payload)}


def write_round2_supports(bundle: dict, config: dict, output: Path) -> Path:
    supports = sample_supports(bundle, config)
    directory = new_directory(output, 'round2-supports')
    write_json(directory / 'support_sets.json', supports)
    write_csv(directory / 'support_sets.csv', supports['records'], SUPPORT_FIELDS)
    write_csv(directory / 'source_coverage.csv', supports['source_coverage'], list(supports['source_coverage'][0]))
    return directory / 'support_sets.json'


def load_round2_supports(path: Path, bundle: dict, config: dict) -> dict:
    supports = read_json(path)
    if supports != sample_supports(bundle, config):
        raise ValueError('round2 support identities/hash/coverage mismatch')
    return supports
