"""Paired V0/V1 development runs on one frozen group split and frame bank."""
from __future__ import annotations

from pathlib import Path
import traceback
import numpy as np

from .config import CLASSES, ROOT
from .metrics import classification_metrics
from .pipeline import PREDICTION_FIELDS, official_clip_condition
from .provenance import digest, environment, new_directory, now, read_json, sha256_file, source_identity, write_csv, write_json
from .ridge import fit_ridge
from .round2_features import load_round2_features
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS, group_statistics, load_grouped_bundle, load_round2_supports

PREDICTIONS = ['representation', 'source_group_id', *PREDICTION_FIELDS]


def expected_pairs(config: dict) -> list[list]:
    return [[representation, k, seed] for k in config['sampling']['shots']
            for seed in config['sampling']['support_seeds'] for representation in ('V0', 'V1')]


def run_round2(manifest: Path, supports_path: Path, index_path: Path, config: dict, paths: dict,
               feature_root: Path, *, stage: str = 'development') -> Path:
    if stage != 'development':
        raise ValueError('round2 only supports development; final_test remains sealed')
    bundle = load_grouped_bundle(manifest, config)
    supports = load_round2_supports(supports_path, bundle, config)
    vectors, identity = load_round2_features(index_path, bundle, feature_root)
    safe_output(paths['run_root'], paths)
    directory = new_directory(paths['run_root'], 'e0-round2-development')
    evaluation = sorted(r['video_id'] for r in bundle['rows'] if r['experimental_role'] == 'development')
    by_id = {r['video_id']: r for r in bundle['rows']}
    dimension = identity['encoder']['dimension']
    batch = dict(status='running', started_at=now(), evaluation_stage=stage, config=config,
                 protocol_id=bundle['protocol_id'], feature_identity=digest(identity),
                 feature_condition='official_openai_clip_vit_b32' if official_clip_condition(identity) else 'validation_only_unverified_features',
                 parent_protocol_id=bundle['parent_manifest']['protocol_id'], source=source_identity(ROOT),
                 evaluation_ids=evaluation, expected_pairs=expected_pairs(config), runs=[], errors=[],
                 group_statistics=group_statistics(bundle['rows']),
                 manifest_file_sha256=sha256_file(manifest), support_file_sha256=sha256_file(supports_path),
                 feature_index_sha256=sha256_file(index_path))
    write_json(directory / 'batch.json', batch)
    try:
        for source, target in ((manifest, 'manifest_snapshot.json'), (supports_path, 'support_sets_snapshot.json'),
                               (index_path, 'feature_index_snapshot.json'),
                               (Path(index_path).parent / 'parent_feature_index_snapshot.json', 'parent_feature_index_snapshot.json')):
            (directory / target).write_bytes(Path(source).read_bytes())
        write_json(directory / 'environment.json', environment())
        truth = np.array([by_id[vid]['class_id'] for vid in evaluation])
        for representation, k, seed in batch['expected_pairs']:
            support = sorted((r for r in supports['records'] if r['k'] == k and r['support_seed'] == seed),
                             key=lambda r: (r['class_id'], r['rank_in_class']))
            ids = [r['video_id'] for r in support]
            if len(ids) != 5 * k or len(set(ids)) != 5 * k or set(ids) & set(evaluation):
                raise ValueError('round2 support/video overlap or count mismatch')
            support_groups = sorted({r['source_group_id'] for r in support})
            if set(support_groups) & {by_id[vid]['source_group_id'] for vid in evaluation}:
                raise ValueError('round2 source overlap')
            run_dir = new_directory(directory, f'{representation.lower()}-k{k}-s{seed}')
            record = dict(directory=run_dir.name, representation=representation, k=k, support_seed=seed)
            batch['runs'].append(record)
            provenance = dict(**record, run_id=run_dir.name, status='running', started_at=now(),
                              evaluation_stage=stage, protocol_id=bundle['protocol_id'], feature_identity=digest(identity),
                              support_ids=ids, support_sha256=digest(sorted(ids)), evaluation_ids=evaluation,
                              support_source_groups=support_groups, n_support_sources=len(support_groups), n_support=len(ids),
                              feature_dimension=dimension * (2 if representation == 'V0' else 5), source=batch['source'])
            write_json(run_dir / 'resolved_config.json', config)
            write_json(run_dir / 'provenance.json', provenance)
            write_csv(run_dir / 'support_set.csv', support, SUPPORT_FIELDS)
            try:
                weights, intercept = fit_ridge(np.stack([vectors[representation][vid] for vid in ids]),
                                              np.array([r['class_id'] for r in support]), config['classifier']['lambda'])
                scores = np.stack([vectors[representation][vid] for vid in evaluation]).astype(np.float64) @ weights + intercept
                predictions = [dict(representation=representation, source_group_id=by_id[vid]['source_group_id'],
                                    run_id=run_dir.name, evaluation_stage=stage, support_seed=seed, k=k, video_id=vid,
                                    true_class_id=int(truth[i]), predicted_class_id=int(scores[i].argmax()),
                                    **{f'score_{name}': float(scores[i, j]) for j, name in enumerate(CLASSES)})
                               for i, vid in enumerate(evaluation)]
                metrics = dict(**classification_metrics(truth, scores), evaluation_stage=stage, representation=representation,
                               n_support=len(ids), n_support_sources=len(support_groups), feature_dimension=provenance['feature_dimension'])
                write_csv(run_dir / 'predictions.csv', predictions, PREDICTIONS)
                write_json(run_dir / 'metrics.json', metrics)
                np.savez_compressed(run_dir / 'ridge.npz', W=weights, b=intercept)
                provenance.update(status='success', predictions_sha256=sha256_file(run_dir / 'predictions.csv'),
                                  metrics_sha256=sha256_file(run_dir / 'metrics.json'), support_csv_sha256=sha256_file(run_dir / 'support_set.csv'))
                (run_dir / 'run.log').write_text(f'{representation}; success; {len(ids)} support; {len(evaluation)} development\n', encoding='utf-8')
            except Exception:
                provenance['status'] = 'failed'
                (run_dir / 'run.log').write_text(traceback.format_exc(), encoding='utf-8')
                raise
            finally:
                provenance['finished_at'] = now()
                write_json(run_dir / 'provenance.json', provenance)
            write_json(directory / 'batch.json', batch)
        for path, key in ((manifest, 'manifest_file_sha256'), (supports_path, 'support_file_sha256'), (index_path, 'feature_index_sha256')):
            if sha256_file(path) != batch[key]:
                raise ValueError('input artifact changed during round2 fitting')
        batch['status'] = 'success'
    except Exception as exc:
        batch['status'] = 'failed'
        batch['errors'].append(f'{type(exc).__name__}: {exc}')
        (directory / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        batch['finished_at'] = now()
        write_json(directory / 'batch.json', batch)
    return directory
