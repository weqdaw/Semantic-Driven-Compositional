"""Explicit E1.1 cache preparation and development fitting, preserving E1 evidence."""
from __future__ import annotations

from pathlib import Path
import shutil
import traceback
import numpy as np

from .config import CLASSES, ROOT
from .e1_experiment import load_bank
from .e1_reporting import validate_e1_runs
from .e11_features import derive_bank, schedule, validate_config
from .metrics import classification_metrics
from .provenance import digest, environment, new_directory, now, read_json, sha256_file, source_identity, write_csv, write_json
from .ridge import fit_ridge
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS

PREDICTION_FIELDS=['representation','rotation_seed','run_id','evaluation_stage','k','support_seed',
                   'video_id','source_group_id','true_class_id','predicted_class_id',*['score_'+c for c in CLASSES]]
RUN_FILES=('resolved_config.json','support_set.csv','predictions.csv','metrics.json','ridge.npz')


def tree_hashes(directory: Path) -> dict:
    directory=Path(directory).resolve()
    paths=sorted(directory.rglob('*'))
    if not directory.is_dir() or any(p.is_symlink() or not p.resolve().is_relative_to(directory) for p in paths):
        raise ValueError('unsafe or absent evidence tree')
    return {p.relative_to(directory).as_posix():sha256_file(p) for p in paths if p.is_file()}


def code_hashes() -> dict:
    return source_identity(ROOT)['files']


def parent_context(directory: Path):
    batch,runs,baseline=validate_e1_runs(directory)
    index=read_json(Path(directory)/'feature_index_snapshot.json')
    bank=load_bank(index,Path(directory)/'feature_bank.npz')
    bundle=read_json(Path(directory)/'reference/manifest_snapshot.json')
    supports=read_json(Path(directory)/'reference/support_sets_snapshot.json')
    return batch,runs,baseline,index,bank,bundle,supports


def check_parent(directory: Path, config: dict) -> dict:
    validate_config(config)
    batch,_,_,index,bank,_,_=parent_context(directory)
    return dict(status='verified',evaluation_stage='development',final_test_enabled=False,
                protocol_id=batch['protocol_id'],feature_condition=batch['feature_condition'],
                parent_feature_identity=batch['feature_identity'],
                n_cached_videos=len(index['entries']),n_development_videos=len(batch['evaluation_ids']),
                n_concepts=len(bank['E']),dimensions={'G1':5*len(bank['E']),
                'VC1':5*(bank['E'].shape[1]+len(bank['E'])),'VG1':5*(bank['E'].shape[1]+len(bank['E']))},
                planned_new_runs=len(schedule(config,batch['reference_config'])),reused_baseline_runs=60,
                fits_executed=0,image_encoding_executed=False,text_encoding_executed=False)


def copy_evidence(source: Path, target: Path, hashes: dict) -> None:
    for name,expected in hashes.items():
        destination=(target/name).resolve()
        if not destination.is_relative_to(target.resolve()):
            raise ValueError('unsafe evidence snapshot path')
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,destination)
        if sha256_file(destination)!=expected:
            raise ValueError('input changed during evidence snapshot')


def prepare_features(parent: Path, config: dict, paths: dict, output: Path) -> Path:
    validate_config(config); parent=Path(parent)
    batch,_,_,old_index,old_bank,_,_=parent_context(parent)
    hashes=tree_hashes(parent)
    output=safe_output(output,paths,protected=(parent,))
    folder=new_directory(output,'e11-diagnostic-features')
    identity=dict(schema='e11_geometry_fusion_v1',config=config,protocol_id=batch['protocol_id'],
                  parent_feature_identity=batch['feature_identity'],parent_evidence_sha256=digest(hashes),
                  code=code_hashes(),numpy=np.__version__)
    index=dict(status='running',errors=[],evaluation_stage='development',identity=identity,
               feature_condition=batch['feature_condition'],entries=old_index['entries'],parent_hashes=hashes)
    target=folder/'index-development.json'; write_json(target,index)
    try:
        copy_evidence(parent,folder/'parent',hashes)
        bank=derive_bank(old_bank,config)
        np.savez_compressed(folder/'features.npz',**bank)
        if tree_hashes(parent)!=hashes:
            raise ValueError('E1 evidence changed during preparation')
        index.update(status='success',npz_sha256=sha256_file(folder/'features.npz'))
    except Exception as exc:
        index['status']='failed'; index['errors'].append(str(exc)); raise
    finally:
        write_json(target,index)
    return target


def load_features(index_path: Path, parent: Path | None=None, *, current_code=True):
    index_path=Path(index_path); index=read_json(index_path)
    if index.get('status')!='success' or index.get('errors') or index['evaluation_stage']!='development':
        raise ValueError('incomplete or non-development E1.1 feature bank')
    identity=index['identity']; validate_config(identity['config'])
    if identity['schema']!='e11_geometry_fusion_v1' or digest(index['parent_hashes'])!=identity['parent_evidence_sha256']:
        raise ValueError('E1.1 parent evidence identity mismatch')
    if tree_hashes(index_path.parent/'parent')!=index['parent_hashes']:
        raise ValueError('corrupt E1 parent snapshot')
    if parent is not None and tree_hashes(parent)!=index['parent_hashes']:
        raise ValueError('E1 source batch differs from prepared parent')
    batch,_,_,old_index,old_bank,_,_=parent_context(index_path.parent/'parent')
    if (index['entries']!=old_index['entries'] or identity['protocol_id']!=batch['protocol_id'] or
            identity['parent_feature_identity']!=batch['feature_identity'] or index['feature_condition']!=batch['feature_condition']):
        raise ValueError('mixed E1.1 parent/entry/feature identity')
    if current_code and identity['code']!=code_hashes():
        raise ValueError('E1.1 code changed; explicitly prepare a new feature cache')
    bank_path=index_path.parent/'features.npz'
    if sha256_file(bank_path)!=index['npz_sha256']:
        raise ValueError('corrupt E1.1 bank bytes')
    with np.load(bank_path,allow_pickle=False) as archive:
        bank={k:archive[k] for k in archive.files}
    expected=derive_bank(old_bank,identity['config'])
    if set(bank)!=set(expected) or any(bank[k].dtype!=v.dtype or not np.array_equal(bank[k],v) for k,v in expected.items()):
        raise ValueError('E1.1 bank differs from deterministic geometry/fusion derivation')
    return index,bank


def select_vectors(bank: dict, config: dict, representation: str, rotation_seed: int | None):
    return bank[representation] if rotation_seed is None else bank[representation][config['rotation_seeds'].index(rotation_seed)]


def run_e11(parent: Path, index_path: Path, paths: dict, *, stage='development') -> Path:
    if stage!='development': raise ValueError('E1.1 is development only; final test sealed')
    parent,index_path=Path(parent),Path(index_path)
    index,bank=load_features(index_path,parent)
    old=read_json(parent/'batch.json'); config=index['identity']['config']
    reference=old['reference_config']
    bundle=read_json(parent/'reference/manifest_snapshot.json')
    supports=read_json(parent/'reference/support_sets_snapshot.json')
    root=safe_output(paths['run_root'],paths,protected=(parent,index_path.parent))
    folder=new_directory(root,'e11-diagnostic-development')
    batch=dict(status='running',errors=[],started_at=now(),evaluation_stage=stage,
               config=config,reference_config=reference,protocol_id=old['protocol_id'],
               feature_condition=index['feature_condition'],feature_identity=digest(index['identity']),
               feature_index_sha256=sha256_file(index_path),evaluation_ids=old['evaluation_ids'],
               expected_runs=schedule(config,reference),runs=[],source=source_identity(ROOT))
    write_json(folder/'batch.json',batch)
    try:
        copy_evidence(index_path.parent/'parent',folder/'parent',index['parent_hashes'])
        shutil.copyfile(index_path,folder/'feature_index_snapshot.json')
        shutil.copyfile(index_path.parent/'features.npz',folder/'features.npz')
        copy_evidence(ROOT,folder/'code_snapshot',batch['source']['files'])
        write_json(folder/'environment.json',environment())
        batch['environment_sha256']=sha256_file(folder/'environment.json')
        positions={e['video_id']:i for i,e in enumerate(index['entries'])}
        by_id={r['video_id']:r for r in bundle['rows']}; dev=batch['evaluation_ids']
        truth=np.array([by_id[v]['class_id'] for v in dev])
        for rep,rotation,k,seed in batch['expected_runs']:
            vectors=select_vectors(bank,config,rep,rotation)
            support=sorted([r for r in supports['records'] if r['k']==k and r['support_seed']==seed],key=lambda r:(r['class_id'],r['rank_in_class']))
            ids=[r['video_id'] for r in support]; groups=sorted({r['source_group_id'] for r in support})
            if len(set(ids))!=5*k or set(ids)&set(dev) or set(groups)&{by_id[v]['source_group_id'] for v in dev}:
                raise ValueError('E1.1 support/source overlap or count mismatch')
            run=new_directory(folder,f'{rep.lower()}-q{rotation}-k{k}-s{seed}')
            record=dict(directory=run.name,representation=rep,rotation_seed=rotation,k=k,support_seed=seed)
            provenance=dict(**record,status='running',evaluation_stage=stage,protocol_id=batch['protocol_id'],
                            feature_identity=batch['feature_identity'],support_ids=ids,support_sha256=digest(sorted(ids)),
                            support_source_groups=groups,evaluation_ids=dev,feature_dimension=vectors.shape[1])
            write_json(run/'provenance.json',provenance)
            write_json(run/'resolved_config.json',dict(e11=config,reference=reference))
            write_csv(run/'support_set.csv',support,SUPPORT_FIELDS)
            w,b=fit_ridge(vectors[[positions[v] for v in ids]],np.array([r['class_id'] for r in support]),reference['classifier']['lambda'])
            scores=vectors[[positions[v] for v in dev]].astype(np.float64)@w+b
            predictions=[dict(representation=rep,rotation_seed=rotation,run_id=run.name,evaluation_stage=stage,
                              k=k,support_seed=seed,video_id=vid,source_group_id=by_id[vid]['source_group_id'],
                              true_class_id=int(truth[i]),predicted_class_id=int(scores[i].argmax()),
                              **{f'score_{c}':float(scores[i,j]) for j,c in enumerate(CLASSES)}) for i,vid in enumerate(dev)]
            write_csv(run/'predictions.csv',predictions,PREDICTION_FIELDS)
            write_json(run/'metrics.json',classification_metrics(truth,scores))
            np.savez_compressed(run/'ridge.npz',W=w,b=b)
            provenance.update(status='success',hashes={name:sha256_file(run/name) for name in RUN_FILES})
            write_json(run/'provenance.json',provenance)
            (run/'run.log').write_text('success; development only; frozen cached features\n',encoding='utf-8')
            batch['runs'].append(record); write_json(folder/'batch.json',batch)
        if (sha256_file(index_path)!=batch['feature_index_sha256'] or
                sha256_file(index_path.parent/'features.npz')!=index['npz_sha256'] or
                tree_hashes(parent)!=index['parent_hashes'] or code_hashes()!=index['identity']['code']):
            raise ValueError('E1.1 inputs changed during fitting')
        batch['status']='success'
    except Exception as exc:
        batch['status']='failed'; batch['errors'].append(f'{type(exc).__name__}: {exc}')
        (folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8'); raise
    finally:
        batch['finished_at']=now(); write_json(folder/'batch.json',batch)
    return folder
