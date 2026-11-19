"""Reuse validated E1.3 features; explicitly evaluate text-anchored corrections."""
from pathlib import Path
import shutil
import traceback
import numpy as np

from .config import ROOT,CLASSES
from .e11_workflow import tree_hashes,copy_evidence,code_hashes
from .e13_reporting import validate_runs as validate_e13
from .e13_workflow import OFFICIAL
from .e14_features import validate_config,schedule,score_model,score_branches
from .metrics import classification_metrics
from .provenance import digest,environment,new_directory,now,read_json,sha256_file,source_identity,write_csv,write_json
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS

RUN_FILES=('resolved_config.json','support_set.csv','score_model.npz','predictions.csv','metrics.json')
PRED_FIELDS=['representation','rotation_seed','k','support_seed','lambda','run_id','evaluation_stage','video_id',
             'source_group_id','true_class_id','predicted_class_id',*['score_'+c for c in CLASSES],
             *['text_score_'+c for c in CLASSES],*['auxiliary_score_'+c for c in CLASSES]]


def parent_context(directory: Path):
    directory=Path(directory);batch,runs=validate_e13(directory)
    index=read_json(directory/'feature_index_snapshot.json')
    with np.load(directory/'features.npz',allow_pickle=False) as data:bank={k:data[k] for k in data.files}
    base=directory/'parent/parent/parent/reference'
    refs=[]
    for r in runs:
        if r['representation'] in ('T0','P0','P1'):
            refs.append(dict(representation=r['representation'],rotation_seed=r['rotation_seed'],k=r['k'],support_seed=r['support_seed'],
                             metrics=r['metrics'],support_sha256=r['support_sha256'],
                             prediction_relative_path=r['prediction_path'].relative_to(directory).as_posix()))
    if len(refs)!=21:raise ValueError('incomplete E1.3 references')
    return dict(batch=batch,index=index,bank=bank,references=refs,bundle=read_json(base/'manifest_snapshot.json'),supports=read_json(base/'support_sets_snapshot.json'))


def check_parent(directory: Path,config: dict):
    validate_config(config);context=parent_context(directory)
    return dict(status='verified',evaluation_stage='development',final_test_enabled=False,
                protocol_id=context['batch']['protocol_id'],parent_feature_identity=context['batch']['feature_identity'],
                feature_condition=context['batch']['feature_condition'],n_cached_videos=len(context['bank']['q']),dimension=context['bank']['q'].shape[1],
                n_development_videos=len(context['batch']['evaluation_ids']),planned_new_evaluations=80,reused_reference_evaluations=21,
                gradient_fits_executed=0,text_encoding_executed=False,image_encoding_executed=False)


def prepare_e14(parent: Path,config: dict,paths: dict,output: Path):
    validate_config(config);parent=Path(parent);hashes=tree_hashes(parent);context=parent_context(parent)
    folder=new_directory(safe_output(output,paths,protected=(parent,)),'e14-text-anchor-inputs');target=folder/'index-development.json'
    identity=dict(schema='e14_text_anchor_v1',config=config,protocol_id=context['batch']['protocol_id'],
                  parent_feature_identity=context['batch']['feature_identity'],parent_evidence_sha256=digest(hashes),code=code_hashes(),numpy=np.__version__)
    index=dict(status='running',errors=[],evaluation_stage='development',identity=identity,parent_hashes=hashes,
               feature_condition=context['batch']['feature_condition'],entries=context['index']['entries'])
    write_json(target,index)
    try:
        copy_evidence(parent,folder/'parent',hashes)
        if tree_hashes(parent)!=hashes or code_hashes()!=identity['code']:raise ValueError('E1.4 input changed during preparation')
        index['status']='success'
    except Exception as exc:index['status']='failed';index['errors'].append(str(exc));raise
    finally:write_json(target,index)
    return target


def load_prepared(index_path: Path,parent: Path | None=None,*,current_code=True):
    index_path=Path(index_path);index=read_json(index_path);identity=index['identity'];validate_config(identity['config'])
    if (index['status']!='success' or index['errors'] or index['evaluation_stage']!='development' or identity['schema']!='e14_text_anchor_v1' or
            digest(index['parent_hashes'])!=identity['parent_evidence_sha256']):raise ValueError('incomplete/mixed E1.4 prepared inputs')
    if tree_hashes(index_path.parent/'parent')!=index['parent_hashes'] or (parent is not None and tree_hashes(parent)!=index['parent_hashes']):
        raise ValueError('E1.4 parent evidence differs from prepared snapshot')
    if current_code and code_hashes()!=identity['code']:raise ValueError('E1.4 code changed; prepare a new input snapshot')
    context=parent_context(index_path.parent/'parent')
    if (identity['protocol_id']!=context['batch']['protocol_id'] or identity['parent_feature_identity']!=context['batch']['feature_identity'] or
            index['entries']!=context['index']['entries'] or index['feature_condition']!=context['batch']['feature_condition']):
        raise ValueError('mixed E1.4 protocol/feature/entry identity')
    return index,context


def get_support(context,seed):
    return sorted([r for r in context['supports']['records'] if r['k']==3 and r['support_seed']==seed],key=lambda r:(r['class_id'],r['rank_in_class']))


def run_e14(parent: Path,index_path: Path,paths: dict,*,stage='development'):
    if stage!='development':raise ValueError('E1.4 development only; final test sealed')
    parent,index_path=Path(parent),Path(index_path);index,context=load_prepared(index_path,parent);config=index['identity']['config']
    folder=new_directory(safe_output(paths['run_root'],paths,protected=(parent,index_path.parent)),'e14-text-anchor-development')
    batch=dict(status='running',errors=[],started_at=now(),evaluation_stage=stage,config=config,protocol_id=index['identity']['protocol_id'],
               feature_identity=digest(index['identity']),feature_condition=index['feature_condition'],feature_index_sha256=sha256_file(index_path),
               evaluation_ids=context['batch']['evaluation_ids'],expected_runs=schedule(config),runs=[],source=source_identity(ROOT),gradient_fits_executed=0)
    write_json(folder/'batch.json',batch)
    try:
        copy_evidence(index_path.parent/'parent',folder/'parent',index['parent_hashes'])
        shutil.copyfile(index_path,folder/'feature_index_snapshot.json');copy_evidence(ROOT,folder/'code_snapshot',batch['source']['files'])
        write_json(folder/'environment.json',environment());batch['environment_sha256']=sha256_file(folder/'environment.json')
        bank=context['bank'];positions={r['video_id']:j for j,r in enumerate(index['entries'])};by_id={r['video_id']:r for r in context['bundle']['rows']}
        dev=batch['evaluation_ids'];queries=bank['q'][[positions[v] for v in dev]];truth=np.asarray([by_id[v]['class_id'] for v in dev])
        for rep,rotation,k,seed,lambda_ in batch['expected_runs']:
            run=new_directory(folder,f'{rep.lower()}-q{rotation}-k{k}-s{seed}')
            record=dict(directory=run.name,representation=rep,rotation_seed=rotation,k=k,support_seed=seed,**{'lambda':lambda_})
            prov=dict(**record,status='running',protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],evaluation_stage=stage,evaluation_ids=dev)
            write_json(run/'provenance.json',prov)
            try:
                support=get_support(context,seed);ids=[r['video_id'] for r in support];groups=sorted({r['source_group_id'] for r in support})
                if len(set(ids))!=15 or set(ids)&set(dev) or set(groups)&{by_id[v]['source_group_id'] for v in dev}:raise ValueError('E1.4 support/source overlap/count')
                model=score_model(bank,rep,rotation,[positions[v] for v in ids],np.asarray([r['class_id'] for r in support],dtype=np.int64))
                scores,text,aux=score_branches(queries,model,lambda_)
                write_json(run/'resolved_config.json',dict(protocol=config,**{'lambda':lambda_}));write_csv(run/'support_set.csv',support,SUPPORT_FIELDS)
                np.savez_compressed(run/'score_model.npz',**model)
                predictions=[dict(representation=rep,rotation_seed=rotation,k=3,support_seed=seed,run_id=run.name,evaluation_stage=stage,
                                  video_id=v,source_group_id=by_id[v]['source_group_id'],true_class_id=int(truth[i]),predicted_class_id=int(scores[i].argmax()),
                                  **{'lambda':lambda_},**{'score_'+c:float(scores[i,j]) for j,c in enumerate(CLASSES)},
                                  **{'text_score_'+c:float(text[i,j]) for j,c in enumerate(CLASSES)},
                                  **{'auxiliary_score_'+c:float(aux[i,j]) for j,c in enumerate(CLASSES)}) for i,v in enumerate(dev)]
                write_csv(run/'predictions.csv',predictions,PRED_FIELDS);write_json(run/'metrics.json',classification_metrics(truth,scores))
                prov.update(status='success',support_ids=ids,support_sha256=digest(sorted(ids)),support_source_groups=groups,
                            feature_dimension=bank['q'].shape[1],hashes={n:sha256_file(run/n) for n in RUN_FILES})
                (run/'run.log').write_text('success; fixed text anchor; raw additive scores; no gradient training\n',encoding='utf-8')
            except Exception:prov['status']='failed';(run/'run.log').write_text(traceback.format_exc(),encoding='utf-8');raise
            finally:write_json(run/'provenance.json',prov)
            batch['runs'].append(record);write_json(folder/'batch.json',batch)
        if sha256_file(index_path)!=batch['feature_index_sha256'] or tree_hashes(parent)!=index['parent_hashes'] or code_hashes()!=index['identity']['code']:
            raise ValueError('E1.4 inputs changed during evaluation')
        batch['status']='success'
    except Exception as exc:
        batch['status']='failed';batch['errors'].append(f'{type(exc).__name__}: {exc}');(folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:batch['finished_at']=now();write_json(folder/'batch.json',batch)
    return folder
