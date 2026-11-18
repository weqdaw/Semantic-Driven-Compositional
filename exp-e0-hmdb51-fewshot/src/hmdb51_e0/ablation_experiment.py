"""Prepare and fit supplementary controls without modifying historical inputs."""
from __future__ import annotations

from pathlib import Path
import shutil
import traceback
import numpy as np

from .ablation_features import ablation_code, ablation_pool, frame_permutation, validate_ablation_config
from .config import CLASSES, ROOT
from .metrics import classification_metrics
from .provenance import digest, environment, new_directory, now, read_json, sha256_file, source_identity, write_csv, write_json
from .ridge import fit_ridge
from .round2_features import Round2Cache, load_round2_features
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS
from .round2_reporting import validated_round2_runs
from .round2_runner import PREDICTIONS


def reference_context(directory: Path) -> tuple[dict, list[dict], dict, dict]:
    directory = Path(directory)
    config = read_json(directory / 'batch.json')['config']
    batch, runs = validated_round2_runs(directory, config)
    return batch, runs, read_json(directory / 'manifest_snapshot.json'), read_json(directory / 'support_sets_snapshot.json')


def reference_hashes(directory: Path, batch: dict) -> dict:
    files = ['batch.json', 'manifest_snapshot.json', 'support_sets_snapshot.json',
             'feature_index_snapshot.json', 'parent_feature_index_snapshot.json', 'environment.json']
    files += [f"{r['directory']}/{name}" for r in batch['runs'] for name in
              ('provenance.json', 'resolved_config.json', 'support_set.csv', 'predictions.csv', 'metrics.json')]
    return {name: sha256_file(Path(directory) / name) for name in files}


def condition_schedule(config: dict) -> list[tuple[str, int | None]]:
    return [('V1_repeat', None), *[('V1_shuffle', seed) for seed in config['shuffle']['seeds']]]


def run_schedule(config: dict, reference_config: dict) -> list[list]:
    return [[condition, perturbation, k, seed] for condition, perturbation in condition_schedule(config)
            for k in reference_config['sampling']['shots'] for seed in reference_config['sampling']['support_seeds']]


def check_reference(directory: Path, config: dict, feature_root: Path) -> dict:
    """Read-only preflight. Neither computes controls nor fits a classifier."""
    validate_ablation_config(config)
    batch, _, bundle, _ = reference_context(directory)
    index_path = Path(feature_root) / batch['feature_identity'] / 'index-development.json'
    if sha256_file(index_path) != batch['feature_index_sha256']:
        raise ValueError('reference feature index differs from frozen batch')
    vectors, identity = load_round2_features(index_path, bundle, feature_root)
    return dict(status='verified', evaluation_stage='development', final_test_enabled=False,
                protocol_id=batch['protocol_id'], feature_condition=batch['feature_condition'],
                parent_feature_identity=digest(identity), feature_index=str(index_path.resolve()),
                n_training_videos=len(vectors['V0']), n_development_videos=len(batch['evaluation_ids']),
                reused_baseline_runs=len(batch['runs']), planned_new_runs=len(run_schedule(config, batch['config'])),
                shuffle_seeds=config['shuffle']['seeds'], fits_executed=0)


def prepare_ablation_features(reference: Path, config: dict, paths: dict, feature_root: Path, output: Path) -> Path:
    check = check_reference(reference, config, feature_root)
    batch, _, bundle, _ = reference_context(reference)
    hashes = reference_hashes(reference, batch)
    parent = read_json(Path(reference) / 'feature_index_snapshot.json')
    identity = dict(schema='round2_ablation_bank_v1', protocol_id=batch['protocol_id'],
                    parent_feature_identity=batch['feature_identity'], reference_evidence_sha256=digest(hashes),
                    encoder=parent['identity']['encoder'], config=config, feature_code=ablation_code(), numpy=np.__version__)
    parent_entries = parent['entries']
    output = safe_output(output, paths, protected=(reference, feature_root))
    folder = new_directory(output, 'e0-ablation-features')
    index_path = folder / 'index-development.json'
    index = dict(status='running', errors=[], started_at=now(), evaluation_stage='development', identity=identity,
                 feature_condition=batch['feature_condition'], frozen_verification=parent['frozen_verification'],
                 entries=parent_entries, reference_hashes=hashes, reference_batch=str(Path(reference).resolve()))
    write_json(index_path, index)
    try:
        cache = Round2Cache(feature_root, parent['identity'])
        repeat, shuffled, permutations = [], [[] for _ in config['shuffle']['seeds']], [[] for _ in config['shuffle']['seeds']]
        permutation_rows = []
        for entry in parent_entries:
            vid = entry['video_id']
            z, _, _ = cache.load(entry)
            repeat.append(ablation_pool(z, 'V1_repeat'))
            for j, seed in enumerate(config['shuffle']['seeds']):
                permutation = frame_permutation(vid, seed)
                permutations[j].append(permutation)
                shuffled[j].append(ablation_pool(z, 'V1_shuffle', video_id=vid, shuffle_seed=seed))
                permutation_rows.append(dict(video_id=vid, shuffle_seed=seed,
                                             permutation_zero_based=' '.join(map(str, permutation.tolist())),
                                             original_frame_indices=' '.join(map(str, entry['frame_indices'])),
                                             shuffled_frame_indices=' '.join(str(entry['frame_indices'][i]) for i in permutation)))
        np.savez_compressed(folder / 'features.npz', V1_repeat=np.stack(repeat), V1_shuffle=np.array(shuffled),
                            permutations=np.array(permutations, dtype=np.int16))
        write_csv(folder / 'permutations.csv', permutation_rows, list(permutation_rows[0]))
        if reference_hashes(reference, batch) != hashes:
            raise ValueError('reference artifacts changed during preparation')
        index.update(status='success', npz_sha256=sha256_file(folder / 'features.npz'),
                     permutations_csv_sha256=sha256_file(folder / 'permutations.csv'),
                     feature_dimension=identity['encoder']['dimension'] * 5, preflight=check)
    except Exception as exc:
        index['status'] = 'failed'
        index['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        index['finished_at'] = now()
        write_json(index_path, index)
    return index_path


def validate_ablation_index(index: dict, reference: Path, batch: dict, *, current_code: bool) -> None:
    from .round2_features import validate_frozen_index
    validate_frozen_index(index)
    identity = index['identity']
    validate_ablation_config(identity['config'])
    parent = read_json(Path(reference) / 'feature_index_snapshot.json')
    hashes = reference_hashes(reference, batch)
    if (identity['schema'] != 'round2_ablation_bank_v1' or identity['protocol_id'] != batch['protocol_id']
            or identity['parent_feature_identity'] != batch['feature_identity']
            or identity['reference_evidence_sha256'] != digest(hashes) or index['reference_hashes'] != hashes
            or identity['encoder'] != parent['identity']['encoder']
            or index['entries'] != parent['entries'] or index['feature_condition'] != batch['feature_condition']
            or index['frozen_verification'] != parent['frozen_verification']
            or index['feature_dimension'] != 5 * identity['encoder']['dimension']):
        raise ValueError('ablation index/reference/freeze identity mismatch')
    if current_code and (identity['feature_code'] != ablation_code() or identity['numpy'] != np.__version__):
        raise ValueError('ablation implementation/NumPy identity changed; prepare a new bank')


def load_ablation_features(index_path: Path, reference: Path) -> tuple[dict, dict]:
    batch, _, _, _ = reference_context(reference)
    index_path = Path(index_path)
    index = read_json(index_path)
    validate_ablation_index(index, reference, batch, current_code=True)
    npz, permutation_csv = index_path.parent / 'features.npz', index_path.parent / 'permutations.csv'
    if sha256_file(npz) != index['npz_sha256'] or sha256_file(permutation_csv) != index['permutations_csv_sha256']:
        raise ValueError('corrupt ablation bank/permutation file hash')
    try:
        with np.load(npz, allow_pickle=False) as content:
            if set(content.files) != {'V1_repeat', 'V1_shuffle', 'permutations'}:
                raise ValueError('ablation array names mismatch')
            bank = {key: content[key] for key in content.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f'corrupt ablation NPZ: {exc}') from exc
    n, d, seeds = len(index['entries']), index['feature_dimension'], index['identity']['config']['shuffle']['seeds']
    for name, shape in (('V1_repeat', (n, d)), ('V1_shuffle', (len(seeds), n, d))):
        array = bank[name]
        if (array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all()
                or not np.allclose(np.linalg.norm(array, axis=-1), 1, atol=1e-5)):
            raise ValueError('invalid ablation feature shape/dtype/normalization')
    if bank['permutations'].shape != (len(seeds), n, 16) or bank['permutations'].dtype != np.int16:
        raise ValueError('invalid permutation shape/dtype')
    for j, seed in enumerate(seeds):
        for i, entry in enumerate(index['entries']):
            if not np.array_equal(bank['permutations'][j, i], frame_permutation(entry['video_id'], seed)):
                raise ValueError('saved permutation differs from fixed video/seed rule')
    return index, bank


def run_ablations(reference: Path, index_path: Path, paths: dict, *, stage: str = 'development') -> Path:
    if stage != 'development':
        raise ValueError('development only; final_test remains sealed')
    reference = Path(reference)
    parent_batch, _, bundle, supports = reference_context(reference)
    index, bank = load_ablation_features(index_path, reference)
    config = index['identity']['config']
    output = safe_output(paths['run_root'], paths, protected=(reference, Path(index_path).parent))
    folder = new_directory(output, 'e0-round2-ablations')
    batch = dict(status='running', errors=[], started_at=now(), evaluation_stage=stage, config=config,
                 protocol_id=parent_batch['protocol_id'], feature_identity=digest(index['identity']),
                 feature_condition=parent_batch['feature_condition'], reference_config=parent_batch['config'],
                 reference_evidence_sha256=index['identity']['reference_evidence_sha256'],
                 feature_index_sha256=sha256_file(index_path), evaluation_ids=parent_batch['evaluation_ids'],
                 expected_runs=run_schedule(config, parent_batch['config']), runs=[], source=source_identity(ROOT))
    write_json(folder / 'batch.json', batch)
    try:
        for name, expected_hash in index['reference_hashes'].items():
            target = folder / 'reference' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reference / name, target)
            if sha256_file(target) != expected_hash:
                raise ValueError('reference changed during snapshot')
        shutil.copyfile(index_path, folder / 'feature_index_snapshot.json')
        if sha256_file(folder / 'feature_index_snapshot.json') != batch['feature_index_sha256']:
            raise ValueError('ablation index changed during snapshot')
        shutil.copyfile(Path(index_path).parent / 'permutations.csv', folder / 'permutations.csv')
        if sha256_file(folder / 'permutations.csv') != index['permutations_csv_sha256']:
            raise ValueError('permutation CSV changed during snapshot')
        for name, expected_hash in batch['source']['files'].items():
            target = folder / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            if sha256_file(target) != expected_hash:
                raise ValueError('code changed during snapshot')
        write_json(folder / 'environment.json', environment())
        batch['environment_sha256'] = sha256_file(folder / 'environment.json')
        by_id = {r['video_id']: r for r in bundle['rows']}
        positions = {entry['video_id']: i for i, entry in enumerate(index['entries'])}
        evaluation = batch['evaluation_ids']
        truth = np.array([by_id[vid]['class_id'] for vid in evaluation])
        for representation, shuffle_seed, k, seed in batch['expected_runs']:
            vectors = bank['V1_repeat'] if representation == 'V1_repeat' else bank['V1_shuffle'][config['shuffle']['seeds'].index(shuffle_seed)]
            support = sorted((r for r in supports['records'] if r['k'] == k and r['support_seed'] == seed),
                             key=lambda r: (r['class_id'], r['rank_in_class']))
            ids = [r['video_id'] for r in support]
            groups = sorted({r['source_group_id'] for r in support})
            if len(set(ids)) != 5 * k or set(ids) & set(evaluation) or set(groups) & {by_id[v]['source_group_id'] for v in evaluation}:
                raise ValueError('support count or source/evaluation overlap')
            run_dir = new_directory(folder, f'{representation.lower()}-p{shuffle_seed}-k{k}-s{seed}')
            record = dict(directory=run_dir.name, representation=representation, shuffle_seed=shuffle_seed, k=k, support_seed=seed)
            batch['runs'].append(record)
            provenance = dict(**record, run_id=run_dir.name, status='running', started_at=now(),
                              evaluation_stage=stage, protocol_id=batch['protocol_id'], feature_identity=batch['feature_identity'],
                              source=batch['source'], support_ids=ids, support_sha256=digest(sorted(ids)),
                              n_support=len(ids), support_source_groups=groups, n_support_sources=len(groups),
                              evaluation_ids=evaluation, feature_dimension=index['feature_dimension'])
            write_json(run_dir / 'resolved_config.json', dict(ablation=config, reference=parent_batch['config']))
            write_csv(run_dir / 'support_set.csv', support, SUPPORT_FIELDS)
            try:
                weights, intercept = fit_ridge(vectors[[positions[v] for v in ids]],
                                              np.array([r['class_id'] for r in support]), parent_batch['config']['classifier']['lambda'])
                scores = vectors[[positions[v] for v in evaluation]].astype(np.float64) @ weights + intercept
                predictions = [dict(representation=representation, shuffle_seed=shuffle_seed,
                                    source_group_id=by_id[vid]['source_group_id'], run_id=run_dir.name,
                                    evaluation_stage=stage, support_seed=seed, k=k, video_id=vid,
                                    true_class_id=int(truth[i]), predicted_class_id=int(scores[i].argmax()),
                                    **{f'score_{name}': float(scores[i, j]) for j, name in enumerate(CLASSES)})
                               for i, vid in enumerate(evaluation)]
                metrics = dict(**classification_metrics(truth, scores), evaluation_stage=stage,
                               representation=representation, shuffle_seed=shuffle_seed, n_support=len(ids),
                               n_support_sources=len(groups), feature_dimension=index['feature_dimension'])
                write_csv(run_dir / 'predictions.csv', predictions, ['shuffle_seed', *PREDICTIONS])
                write_json(run_dir / 'metrics.json', metrics)
                np.savez_compressed(run_dir / 'ridge.npz', W=weights, b=intercept)
                provenance.update(status='success', predictions_sha256=sha256_file(run_dir / 'predictions.csv'),
                                  metrics_sha256=sha256_file(run_dir / 'metrics.json'),
                                  support_csv_sha256=sha256_file(run_dir / 'support_set.csv'), ridge_sha256=sha256_file(run_dir / 'ridge.npz'))
                (run_dir / 'run.log').write_text('success; development only; no encoder update\n', encoding='utf-8')
            except Exception:
                provenance['status'] = 'failed'
                (run_dir / 'run.log').write_text(traceback.format_exc(), encoding='utf-8')
                raise
            finally:
                provenance['finished_at'] = now()
                write_json(run_dir / 'provenance.json', provenance)
            write_json(folder / 'batch.json', batch)
        if sha256_file(index_path) != batch['feature_index_sha256'] or reference_hashes(reference, parent_batch) != index['reference_hashes']:
            raise ValueError('inputs changed during fitting')
        batch['status'] = 'success'
    except Exception as exc:
        batch['status'] = 'failed'
        batch['errors'].append(f'{type(exc).__name__}: {exc}')
        (folder / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        batch['finished_at'] = now()
        write_json(folder / 'batch.json', batch)
    return folder
