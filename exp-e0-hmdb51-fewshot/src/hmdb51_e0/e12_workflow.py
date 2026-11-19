"""Independent E1.2 preparation, per-support scaling and explicit development fitting."""
from pathlib import Path
import shutil
import traceback
import numpy as np

from .config import ROOT,CLASSES
from .e11_reporting import validate_runs as validate_e11
from .e11_workflow import tree_hashes,copy_evidence,code_hashes
from .e12_features import build_vectors,schedule,validate_config
from .metrics import classification_metrics
from .provenance import digest,environment,new_directory,now,read_json,sha256_file,source_identity,write_csv,write_json
from .ridge import fit_ridge
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS

RUN_FILES=('resolved_config.json','support_set.csv','scalers.npz','scaling.json','predictions.csv','metrics.json','ridge.npz')
PREDICTION_FIELDS=['representation','control_seed','k','support_seed','lambda','run_id','evaluation_stage',
                   'video_id','source_group_id','true_class_id','predicted_class_id',*['score_'+c for c in CLASSES]]


def parent_context(directory: Path):
    directory=Path(directory)
    batch,runs=validate_e11(directory)
    index=read_json(directory/'feature_index_snapshot.json')
    e1=read_json(directory/'parent/batch.json')
    if len(read_json(directory/'parent/concept_catalog.json')['concepts'])!=16:
        raise ValueError('E1.2 requires the unchanged 16-concept E1 catalogue')
    with np.load(directory/'features.npz',allow_pickle=False) as data:
        raw={k:data[k] for k in ('V1','G1')}
    with np.load(directory/'parent/feature_bank.npz',allow_pickle=False) as data:
        raw.update({k:data[k] for k in ('C1','R1')})
    references=[]
    for r in runs:
        if r['k']==3:
            references.append(dict(representation=r['representation'],control_seed=r['rotation_seed'],k=3,support_seed=r['support_seed'],
                                   metrics=r['metrics'],support_sha256=r['support_sha256'],
                                   prediction_relative_path=r['prediction_path'].relative_to(directory).as_posix()))
    for r in e1['runs']:
        if r['representation']=='R1' and r['k']==3:
            folder=directory/'parent'/r['directory']
            references.append(dict(representation='R1',control_seed=r['projection_seed'],k=3,support_seed=r['support_seed'],
                                   metrics=read_json(folder/'metrics.json'),support_sha256=read_json(folder/'provenance.json')['support_sha256'],
                                   prediction_relative_path=(folder/'predictions.csv').relative_to(directory).as_posix()))
    if len(references)!=180: raise ValueError('incomplete E1.2 ancestor reference runs')
    return dict(batch=batch,reference_config=e1['reference_config'],index=index,raw=raw,references=references,
                bundle=read_json(directory/'parent/reference/manifest_snapshot.json'),
                supports=read_json(directory/'parent/reference/support_sets_snapshot.json'))


def check_parent(directory: Path,config: dict):
    validate_config(config); context=parent_context(directory); batch=context['batch']
    return dict(status='verified',evaluation_stage='development',final_test_enabled=False,
                protocol_id=batch['protocol_id'],feature_condition=batch['feature_condition'],
                parent_feature_identity=batch['feature_identity'],n_cached_videos=len(context['index']['entries']),
                n_development_videos=len(batch['evaluation_ids']),shots=[3],planned_new_runs=len(schedule(config)),
                reused_reference_runs=len(context['references']),fits_executed=0,image_encoding_executed=False,text_encoding_executed=False,
                scaler_fit_source='current_15_support_videos_only')


def prepare_e12(parent: Path,config: dict,paths: dict,output: Path):
    validate_config(config); parent=Path(parent); before=tree_hashes(parent)
    context=parent_context(parent)
    output=safe_output(output,paths,protected=(parent,))
    folder=new_directory(output,'e12-support-scale-inputs'); target=folder/'index-development.json'
    identity=dict(schema='e12_support_scale_v1',config=config,protocol_id=context['batch']['protocol_id'],
                  parent_feature_identity=context['batch']['feature_identity'],parent_evidence_sha256=digest(before),
                  code=code_hashes(),numpy=np.__version__)
    index=dict(status='running',errors=[],evaluation_stage='development',identity=identity,parent_hashes=before,
               feature_condition=context['batch']['feature_condition'],entries=context['index']['entries'])
    write_json(target,index)
    try:
        copy_evidence(parent,folder/'parent',before)
        if tree_hashes(parent)!=before or code_hashes()!=identity['code']:
            raise ValueError('E1.2 inputs changed during preparation')
        index['status']='success'
    except Exception as exc:
        index['status']='failed'; index['errors'].append(str(exc)); raise
    finally: write_json(target,index)
    return target


def load_prepared(index_path: Path,parent: Path | None=None,*,current_code=True):
    index_path=Path(index_path); index=read_json(index_path); identity=index['identity']
    validate_config(identity['config'])
    if (index['status']!='success' or index['errors'] or index['evaluation_stage']!='development' or
            identity['schema']!='e12_support_scale_v1' or digest(index['parent_hashes'])!=identity['parent_evidence_sha256']):
        raise ValueError('incomplete or mixed E1.2 preparation')
    if tree_hashes(index_path.parent/'parent')!=index['parent_hashes']:
        raise ValueError('corrupt E1.1 input snapshot')
    if parent is not None and tree_hashes(parent)!=index['parent_hashes']:
        raise ValueError('E1.1 batch differs from prepared input')
    if current_code and code_hashes()!=identity['code']:
        raise ValueError('E1.2 code changed; prepare a new input snapshot')
    context=parent_context(index_path.parent/'parent'); batch=context['batch']
    if (identity['protocol_id']!=batch['protocol_id'] or identity['parent_feature_identity']!=batch['feature_identity'] or
            index['entries']!=context['index']['entries'] or index['feature_condition']!=batch['feature_condition']):
        raise ValueError('mixed E1.2 protocol/feature/entry identity')
    return index,context


def get_support(context,seed):
    return sorted([r for r in context['supports']['records'] if r['k']==3 and r['support_seed']==seed],key=lambda r:(r['class_id'],r['rank_in_class']))


def run_e12(parent: Path,index_path: Path,paths: dict,*,stage='development'):
    if stage!='development': raise ValueError('E1.2 development only; final test sealed')
    parent,index_path=Path(parent),Path(index_path)
    index,context=load_prepared(index_path,parent); config=index['identity']['config']
    folder=new_directory(safe_output(paths['run_root'],paths,protected=(parent,index_path.parent)),'e12-support-scale-development')
    batch=dict(status='running',errors=[],started_at=now(),evaluation_stage=stage,config=config,
               reference_config=context['reference_config'],protocol_id=index['identity']['protocol_id'],
               feature_identity=digest(index['identity']),feature_condition=index['feature_condition'],
               feature_index_sha256=sha256_file(index_path),evaluation_ids=context['batch']['evaluation_ids'],
               expected_runs=schedule(config),runs=[],source=source_identity(ROOT))
    write_json(folder/'batch.json',batch)
    try:
        copy_evidence(index_path.parent/'parent',folder/'parent',index['parent_hashes'])
        shutil.copyfile(index_path,folder/'feature_index_snapshot.json')
        copy_evidence(ROOT,folder/'code_snapshot',batch['source']['files'])
        write_json(folder/'environment.json',environment()); batch['environment_sha256']=sha256_file(folder/'environment.json')
        positions={r['video_id']:i for i,r in enumerate(index['entries'])}
        by_id={r['video_id']:r for r in context['bundle']['rows']}; dev=batch['evaluation_ids']
        truth=np.array([by_id[v]['class_id'] for v in dev])
        for rep,control,k,seed,lambda_ in batch['expected_runs']:
            run=new_directory(folder,f'{rep.lower()}-p{control}-k{k}-s{seed}')
            record=dict(directory=run.name,representation=rep,control_seed=control,k=k,support_seed=seed,**{'lambda':lambda_})
            provenance=dict(**record,status='running',evaluation_stage=stage,protocol_id=batch['protocol_id'],
                            feature_identity=batch['feature_identity'],evaluation_ids=dev)
            write_json(run/'provenance.json',provenance)
            try:
                support=get_support(context,seed); ids=[r['video_id'] for r in support]; groups=sorted({r['source_group_id'] for r in support})
                if len(set(ids))!=15 or set(ids)&set(dev) or set(groups)&{by_id[v]['source_group_id'] for v in dev}:
                    raise ValueError('E1.2 support/source overlap or count mismatch')
                vectors,scalers,diagnostics=build_vectors(context['raw'],rep,control,[positions[v] for v in ids])
                provenance.update(support_ids=ids,support_sha256=digest(sorted(ids)),support_source_groups=groups,feature_dimension=vectors.shape[1])
                write_json(run/'resolved_config.json',dict(e12=config,reference=context['reference_config'],**{'lambda':lambda_}))
                write_csv(run/'support_set.csv',support,SUPPORT_FIELDS)
                np.savez_compressed(run/'scalers.npz',**scalers)
                write_json(run/'scaling.json',dict(scaler_fit_ids=ids,blocks=diagnostics,feature_dimension=vectors.shape[1],post_fusion_row_l2=False))
                w,b=fit_ridge(vectors[[positions[v] for v in ids]],np.array([r['class_id'] for r in support]),lambda_)
                scores=vectors[[positions[v] for v in dev]]@w+b
                predictions=[dict(representation=rep,control_seed=control,k=k,support_seed=seed,run_id=run.name,evaluation_stage=stage,
                                  video_id=vid,source_group_id=by_id[vid]['source_group_id'],true_class_id=int(truth[i]),
                                  predicted_class_id=int(scores[i].argmax()),**{'lambda':lambda_},
                                  **{'score_'+c:float(scores[i,j]) for j,c in enumerate(CLASSES)}) for i,vid in enumerate(dev)]
                write_csv(run/'predictions.csv',predictions,PREDICTION_FIELDS)
                write_json(run/'metrics.json',classification_metrics(truth,scores)); np.savez_compressed(run/'ridge.npz',W=w,b=b)
                provenance.update(status='success',hashes={name:sha256_file(run/name) for name in RUN_FILES})
                (run/'run.log').write_text('success; E1.2 support-only scaling; development\n',encoding='utf-8')
            except Exception:
                provenance['status']='failed'; (run/'run.log').write_text(traceback.format_exc(),encoding='utf-8'); raise
            finally: write_json(run/'provenance.json',provenance)
            batch['runs'].append(record); write_json(folder/'batch.json',batch)
        if (sha256_file(index_path)!=batch['feature_index_sha256'] or tree_hashes(parent)!=index['parent_hashes'] or
                code_hashes()!=index['identity']['code']): raise ValueError('E1.2 inputs changed during fitting')
        batch['status']='success'
    except Exception as exc:
        batch['status']='failed'; batch['errors'].append(f'{type(exc).__name__}: {exc}')
        (folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8'); raise
    finally:
        batch['finished_at']=now(); write_json(folder/'batch.json',batch)
    return folder
