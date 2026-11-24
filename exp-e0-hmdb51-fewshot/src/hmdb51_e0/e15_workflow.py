"""Snapshot a completed E1.4 and evaluate controlled support-instance caches."""
from pathlib import Path
import shutil
import traceback

import numpy as np

from .config import CLASSES, ROOT
from .e11_workflow import tree_hashes, copy_evidence, code_hashes
from .e13_workflow import OFFICIAL
from .e14_reporting import validate_runs as validate_e14
from .e15_features import validate_config, schedule, cache_model, score_branches
from .metrics import classification_metrics
from .provenance import (
    digest, environment, new_directory, now, read_json, sha256_file,
    source_identity, write_csv, write_json,
)
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS

RUN_FILES = (
    'resolved_config.json', 'support_set.csv', 'cache_model.npz',
    'cache_evidence.npz', 'predictions.csv', 'metrics.json',
)
PRED_FIELDS = [
    'representation', 'rotation_seed', 'k', 'support_seed', 'alpha', 'beta', 'lambda',
    'run_id', 'evaluation_stage', 'video_id', 'source_group_id', 'true_class_id', 'predicted_class_id',
    *['score_'+c for c in CLASSES], *['text_score_'+c for c in CLASSES],
    *['cache_score_'+c for c in CLASSES], *['auxiliary_score_'+c for c in CLASSES],
]


def parent_context(directory: Path):
    """Return only independently validated ancestors and the 11 fixed references."""
    directory = Path(directory)
    batch, runs = validate_e14(directory)
    index = read_json(directory/'feature_index_snapshot.json')
    with np.load(directory/'parent/features.npz', allow_pickle=False) as content:
        bank = {key: content[key] for key in content.files}
    base = directory/'parent/parent/parent/parent/reference'
    references = []
    for run in runs:
        if run['representation'] not in ('T0', 'A2'):
            continue
        references.append(dict(
            representation=run['representation'], rotation_seed=run['rotation_seed'],
            k=run['k'], support_seed=run['support_seed'], alpha=None, beta=None,
            **{'lambda': run['lambda']}, metrics=run['metrics'], support_sha256=run['support_sha256'],
            prediction_relative_path=run['prediction_path'].relative_to(directory).as_posix(),
        ))
    identities = {(r['representation'], r['rotation_seed'], r['k'], r['support_seed']) for r in references}
    if (len(references) != 11 or identities != {('T0', None, 0, None), *[('A2', None, 3, s) for s in range(10)]}):
        raise ValueError('incomplete E1.4 T0/A2 references')
    return dict(
        batch=batch, index=index, bank=bank, references=references,
        bundle=read_json(base/'manifest_snapshot.json'), supports=read_json(base/'support_sets_snapshot.json'),
    )


def check_parent(directory: Path, config: dict):
    validate_config(config)
    context = parent_context(directory)
    return dict(
        status='verified', evaluation_stage='development', final_test_enabled=False,
        protocol_id=context['batch']['protocol_id'], parent_feature_identity=context['batch']['feature_identity'],
        feature_condition=context['batch']['feature_condition'], n_cached_videos=len(context['bank']['q']),
        dimension=context['bank']['q'].shape[1], n_development_videos=len(context['batch']['evaluation_ids']),
        planned_new_evaluations=30, reused_reference_evaluations=11, gradient_fits_executed=0,
        text_encoding_executed=False, image_encoding_executed=False,
        alpha=config['alpha'], beta=config['beta'], aggregation=config['aggregation'],
    )


def prepare_e15(parent: Path, config: dict, paths: dict, output: Path):
    validate_config(config)
    parent = Path(parent)
    hashes = tree_hashes(parent)
    context = parent_context(parent)
    folder = new_directory(safe_output(output, paths, protected=(parent,)), 'e15-instance-cache-inputs')
    target = folder/'index-development.json'
    identity = dict(
        schema='e15_instance_cache_v1', config=config, protocol_id=context['batch']['protocol_id'],
        parent_feature_identity=context['batch']['feature_identity'], parent_evidence_sha256=digest(hashes),
        code=code_hashes(), numpy=np.__version__,
    )
    index = dict(
        status='running', errors=[], evaluation_stage='development', identity=identity,
        parent_hashes=hashes, feature_condition=context['batch']['feature_condition'], entries=context['index']['entries'],
    )
    write_json(target, index)
    try:
        copy_evidence(parent, folder/'parent', hashes)
        if tree_hashes(parent) != hashes or code_hashes() != identity['code']:
            raise ValueError('E1.5 input/code changed during preparation')
        index['status'] = 'success'
    except Exception as exc:
        index['status'] = 'failed'
        index['errors'].append(str(exc))
        raise
    finally:
        write_json(target, index)
    return target


def load_prepared(index_path: Path, parent: Path | None = None, *, current_code=True):
    index_path = Path(index_path)
    index = read_json(index_path)
    identity = index['identity']
    validate_config(identity['config'])
    if (index['status'] != 'success' or index['errors'] or index['evaluation_stage'] != 'development'
            or identity['schema'] != 'e15_instance_cache_v1'
            or digest(index['parent_hashes']) != identity['parent_evidence_sha256']):
        raise ValueError('incomplete/mixed E1.5 prepared inputs')
    if (tree_hashes(index_path.parent/'parent') != index['parent_hashes']
            or (parent is not None and tree_hashes(parent) != index['parent_hashes'])):
        raise ValueError('E1.5 parent evidence differs from prepared snapshot')
    if current_code and code_hashes() != identity['code']:
        raise ValueError('E1.5 code changed; prepare a new input snapshot')
    context = parent_context(index_path.parent/'parent')
    if (identity['protocol_id'] != context['batch']['protocol_id']
            or identity['parent_feature_identity'] != context['batch']['feature_identity']
            or index['entries'] != context['index']['entries']
            or index['feature_condition'] != context['batch']['feature_condition']):
        raise ValueError('mixed E1.5 protocol/feature/entry identity')
    return index, context


def get_support(context, seed):
    return sorted(
        [r for r in context['supports']['records'] if r['k'] == 3 and r['support_seed'] == seed],
        key=lambda r: (r['class_id'], r['rank_in_class']),
    )


def validate_support(support, development_ids, by_id):
    ids = [r['video_id'] for r in support]
    groups = sorted({r['source_group_id'] for r in support})
    if (len(ids) != 15 or len(set(ids)) != 15 or set(ids) & set(development_ids)
            or set(groups) & {by_id[v]['source_group_id'] for v in development_ids}
            or any(sum(r['class_id'] == c for r in support) != 3 for c in range(5))
            or any(r['class_id'] != by_id[r['video_id']]['class_id']
                   or r['source_group_id'] != by_id[r['video_id']]['source_group_id'] for r in support)):
        raise ValueError('E1.5 support identity/source overlap/count mismatch')
    return ids, groups


def evidence_payload(development_ids, support_ids, similarities, affinities, votes):
    return dict(
        query_ids=np.asarray(development_ids), support_ids=np.asarray(support_ids),
        cosine_similarities=similarities, affinities=affinities, votes=votes,
    )


def run_e15(parent: Path, index_path: Path, paths: dict, *, stage='development'):
    if stage != 'development':
        raise ValueError('E1.5 development only; final test sealed')
    parent, index_path = Path(parent), Path(index_path)
    index, context = load_prepared(index_path, parent)
    config = index['identity']['config']
    folder = new_directory(
        safe_output(paths['run_root'], paths, protected=(parent, index_path.parent)), 'e15-instance-cache-development',
    )
    batch = dict(
        status='running', errors=[], started_at=now(), evaluation_stage=stage, config=config,
        protocol_id=index['identity']['protocol_id'], feature_identity=digest(index['identity']),
        feature_condition=index['feature_condition'], feature_index_sha256=sha256_file(index_path),
        evaluation_ids=context['batch']['evaluation_ids'], expected_runs=schedule(config), runs=[],
        source=source_identity(ROOT), gradient_fits_executed=0,
    )
    write_json(folder/'batch.json', batch)
    try:
        copy_evidence(index_path.parent/'parent', folder/'parent', index['parent_hashes'])
        shutil.copyfile(index_path, folder/'feature_index_snapshot.json')
        copy_evidence(ROOT, folder/'code_snapshot', batch['source']['files'])
        write_json(folder/'environment.json', environment())
        batch['environment_sha256'] = sha256_file(folder/'environment.json')
        bank = context['bank']
        positions = {r['video_id']: j for j, r in enumerate(index['entries'])}
        by_id = {r['video_id']: r for r in context['bundle']['rows']}
        dev = batch['evaluation_ids']
        queries = bank['q'][[positions[v] for v in dev]]
        truth = np.asarray([by_id[v]['class_id'] for v in dev])
        for rep, k, seed in batch['expected_runs']:
            run = new_directory(folder, f'{rep.lower()}-k{k}-s{seed}')
            record = dict(
                directory=run.name, representation=rep, rotation_seed=None, k=k, support_seed=seed,
                alpha=config['alpha'], beta=config['beta'],
            )
            prov = dict(
                **record, status='running', protocol_id=batch['protocol_id'], feature_identity=batch['feature_identity'],
                evaluation_stage=stage, evaluation_ids=dev,
            )
            write_json(run/'provenance.json', prov)
            try:
                support = get_support(context, seed)
                ids, groups = validate_support(support, dev, by_id)
                model = cache_model(bank, rep, [positions[v] for v in ids], np.asarray([r['class_id'] for r in support]), config)
                scores, text, cache, similarities, affinities, votes = score_branches(queries, model, config)
                write_json(run/'resolved_config.json', config)
                write_csv(run/'support_set.csv', support, SUPPORT_FIELDS)
                np.savez_compressed(run/'cache_model.npz', **model)
                np.savez_compressed(run/'cache_evidence.npz', **evidence_payload(dev, ids, similarities, affinities, votes))
                predictions = [dict(
                    representation=rep, rotation_seed=None, k=k, support_seed=seed, alpha=config['alpha'], beta=config['beta'],
                    run_id=run.name, evaluation_stage=stage, video_id=v, source_group_id=by_id[v]['source_group_id'],
                    true_class_id=int(truth[i]), predicted_class_id=int(scores[i].argmax()),
                    **{'score_'+c: float(scores[i, j]) for j, c in enumerate(CLASSES)},
                    **{'text_score_'+c: float(text[i, j]) for j, c in enumerate(CLASSES)},
                    **{'cache_score_'+c: float(cache[i, j]) for j, c in enumerate(CLASSES)},
                ) for i, v in enumerate(dev)]
                write_csv(run/'predictions.csv', predictions, PRED_FIELDS)
                write_json(run/'metrics.json', classification_metrics(truth, scores))
                prov.update(
                    status='success', support_ids=ids, support_sha256=digest(sorted(ids)), support_source_groups=groups,
                    feature_dimension=bank['q'].shape[1], n_cache_keys=15,
                    hashes={name: sha256_file(run/name) for name in RUN_FILES},
                )
                (run/'run.log').write_text('success; fixed text anchor; per-instance kernel votes; no encoding or gradient training\n', encoding='utf-8')
            except Exception:
                prov['status'] = 'failed'
                (run/'run.log').write_text(traceback.format_exc(), encoding='utf-8')
                raise
            finally:
                write_json(run/'provenance.json', prov)
            batch['runs'].append(record)
            write_json(folder/'batch.json', batch)
        if (sha256_file(index_path) != batch['feature_index_sha256'] or tree_hashes(parent) != index['parent_hashes']
                or code_hashes() != index['identity']['code']):
            raise ValueError('E1.5 inputs/code changed during evaluation')
        batch['status'] = 'success'
    except Exception as exc:
        batch['status'] = 'failed'
        batch['errors'].append(f'{type(exc).__name__}: {exc}')
        (folder/'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        batch['finished_at'] = now()
        write_json(folder/'batch.json', batch)
    return folder
