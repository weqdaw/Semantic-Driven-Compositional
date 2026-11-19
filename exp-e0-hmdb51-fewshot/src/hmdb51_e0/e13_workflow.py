"""Explicit text preparation and gradient-free E1.3 prototype evaluation."""
from pathlib import Path
import importlib.util
import shutil
import traceback
import numpy as np

from .clip_features import CHECKPOINT_SHA256
from .config import ROOT,CLASSES
from .e1_features import validate_text_alignment
from .e1_experiment import frame_array_sha256
from .e11_workflow import tree_hashes,copy_evidence,code_hashes
from .e12_reporting import validate_runs as validate_e12
from .e13_features import validate_config,derive_bank,schedule,prototype_model
from .metrics import classification_metrics
from .pipeline import official_clip_condition
from .provenance import digest,environment,new_directory,now,read_json,sha256_file,source_identity,write_csv,write_json
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS

OFFICIAL='official_openai_clip_vit_b32_class_text_prototypes'
RUN_FILES=('resolved_config.json','support_set.csv','prototypes.npz','predictions.csv','metrics.json')
PRED_FIELDS=['representation','rotation_seed','k','support_seed','run_id','evaluation_stage','video_id','source_group_id',
             'true_class_id','predicted_class_id',*['score_'+c for c in CLASSES]]


def parent_context(directory: Path):
    directory=Path(directory); batch,_,_=validate_e12(directory)
    base=directory/'parent/parent'
    index=read_json(base/'feature_index_snapshot.json')
    with np.load(base/'feature_bank.npz',allow_pickle=False) as data: z=data['Z']
    return dict(batch=batch,index=index,z=z,image_encoder=index['identity']['encoder'],reference_text_encoder=index['identity']['text_encoder'],
                bundle=read_json(base/'reference/manifest_snapshot.json'),supports=read_json(base/'reference/support_sets_snapshot.json'))


def validate_class_text(context,text,frozen):
    validate_text_alignment(context['image_encoder'],text,frozen)
    if official_clip_condition({'encoder':context['image_encoder']}):
        if (text.get('text_encoder')!='native_openai_clip' or
                text.get('tokenizer')!=context['reference_text_encoder'].get('tokenizer')):
            raise ValueError('class-text tokenizer/BPE differs from original E1 ancestor')


def check_parent(directory: Path,config: dict,paths: dict):
    validate_config(config); context=parent_context(directory)
    if official_clip_condition({'encoder':context['image_encoder']}):
        weight=paths.get('weights_file')
        if weight is None or sha256_file(weight)!=CHECKPOINT_SHA256: raise ValueError('E1.3 requires the original official checkpoint')
        if any(importlib.util.find_spec(name) is None for name in ('clip','torch')): raise ValueError('native text dependencies unavailable')
    return dict(status='verified',evaluation_stage='development',final_test_enabled=False,
                protocol_id=context['batch']['protocol_id'],parent_feature_identity=context['batch']['feature_identity'],
                n_cached_videos=len(context['z']),n_development_videos=len(context['batch']['evaluation_ids']),
                dimension=context['z'].shape[2],n_fixed_prompts=5,planned_evaluations=81,zero_shot_evaluations=1,
                support_evaluations=80,gradient_fits_executed=0,text_encoding_executed=False,image_encoding_executed=False)


def prepare_e13(parent: Path,config: dict,paths: dict,output: Path,*,text_encoder=None,device='cuda'):
    validate_config(config); parent=Path(parent); hashes=tree_hashes(parent); context=parent_context(parent)
    folder=new_directory(safe_output(output,paths,protected=(parent,)),'e13-class-text-features')
    target=folder/'index-development.json'; index=dict(status='running',errors=[],evaluation_stage='development')
    code=code_hashes(); write_json(target,index)
    try:
        if text_encoder is None:
            from .e1_text import FrozenTextCLIP
            text_encoder=FrozenTextCLIP(paths['weights_file'],device=device)
        e=np.asarray(text_encoder.encode_text(config['prompts']),dtype=np.float32); frozen=text_encoder.verify_frozen()
        validate_class_text(context,text_encoder.identity,frozen)
        bank=derive_bank(context['z'],e,config)
        identity=dict(schema='e13_class_text_bank_v1',config=config,protocol_id=context['batch']['protocol_id'],
                      parent_feature_identity=context['batch']['feature_identity'],parent_evidence_sha256=digest(hashes),
                      image_encoder=context['image_encoder'],text_encoder=text_encoder.identity,text_frozen=frozen,
                      text_array_sha256=frame_array_sha256(e),code=code,numpy=np.__version__)
        condition=OFFICIAL if official_clip_condition({'encoder':context['image_encoder']}) else 'validation_only_unverified_features'
        copy_evidence(parent,folder/'parent',hashes)
        np.savez_compressed(folder/'features.npz',**bank)
        if tree_hashes(parent)!=hashes or code_hashes()!=code: raise ValueError('E1.3 inputs changed during preparation')
        index.update(status='success',identity=identity,parent_hashes=hashes,feature_condition=condition,
                     entries=context['index']['entries'],npz_sha256=sha256_file(folder/'features.npz'))
    except Exception as exc:
        index['status']='failed';index['errors'].append(str(exc));raise
    finally: write_json(target,index)
    return target


def load_features(index_path: Path,parent: Path | None=None,*,current_code=True):
    index_path=Path(index_path); index=read_json(index_path)
    if index.get('status')!='success' or index.get('errors') or index['evaluation_stage']!='development': raise ValueError('incomplete E1.3 cache')
    identity=index['identity'];validate_config(identity['config'])
    if identity['schema']!='e13_class_text_bank_v1' or digest(index['parent_hashes'])!=identity['parent_evidence_sha256']:
        raise ValueError('E1.3 evidence identity mismatch')
    if tree_hashes(index_path.parent/'parent')!=index['parent_hashes'] or (parent is not None and tree_hashes(parent)!=index['parent_hashes']):
        raise ValueError('E1.3 parent snapshot differs from prepared input')
    if current_code and identity['code']!=code_hashes(): raise ValueError('E1.3 code changed; explicitly prepare a new cache')
    context=parent_context(index_path.parent/'parent')
    expected_condition=OFFICIAL if official_clip_condition({'encoder':context['image_encoder']}) else 'validation_only_unverified_features'
    if (identity['protocol_id']!=context['batch']['protocol_id'] or identity['parent_feature_identity']!=context['batch']['feature_identity'] or
            identity['image_encoder']!=context['image_encoder'] or index['entries']!=context['index']['entries'] or index['feature_condition']!=expected_condition):
        raise ValueError('mixed E1.3 image/protocol/entry identity')
    validate_class_text(context,identity['text_encoder'],identity['text_frozen'])
    path=index_path.parent/'features.npz'
    if sha256_file(path)!=index['npz_sha256']: raise ValueError('corrupt E1.3 bank')
    with np.load(path,allow_pickle=False) as data: bank={k:data[k] for k in data.files}
    if frame_array_sha256(bank['E'])!=identity['text_array_sha256']: raise ValueError('encoded class-text bank identity mismatch')
    expected=derive_bank(context['z'],bank['E'],identity['config'])
    if set(bank)!=set(expected) or any(bank[k].dtype!=v.dtype or not np.array_equal(bank[k],v) for k,v in expected.items()):
        raise ValueError('E1.3 bank differs from frame/text/direction derivation')
    return index,bank,context


def get_support(context,seed):
    return [] if seed is None else sorted([r for r in context['supports']['records'] if r['k']==3 and r['support_seed']==seed],key=lambda r:(r['class_id'],r['rank_in_class']))


def run_e13(parent: Path,index_path: Path,paths: dict,*,stage='development'):
    if stage!='development': raise ValueError('E1.3 development only; final test sealed')
    parent,index_path=Path(parent),Path(index_path); index,bank,context=load_features(index_path,parent)
    config=index['identity']['config']; root=safe_output(paths['run_root'],paths,protected=(parent,index_path.parent))
    folder=new_directory(root,'e13-class-text-development')
    batch=dict(status='running',errors=[],started_at=now(),evaluation_stage=stage,config=config,protocol_id=index['identity']['protocol_id'],
               feature_identity=digest(index['identity']),feature_condition=index['feature_condition'],feature_index_sha256=sha256_file(index_path),
               evaluation_ids=context['batch']['evaluation_ids'],expected_runs=schedule(config),runs=[],source=source_identity(ROOT),gradient_fits_executed=0)
    write_json(folder/'batch.json',batch)
    try:
        copy_evidence(index_path.parent/'parent',folder/'parent',index['parent_hashes'])
        shutil.copyfile(index_path,folder/'feature_index_snapshot.json');shutil.copyfile(index_path.parent/'features.npz',folder/'features.npz')
        copy_evidence(ROOT,folder/'code_snapshot',batch['source']['files'])
        write_json(folder/'environment.json',environment());batch['environment_sha256']=sha256_file(folder/'environment.json')
        positions={r['video_id']:i for i,r in enumerate(index['entries'])};by_id={r['video_id']:r for r in context['bundle']['rows']}
        dev=batch['evaluation_ids'];truth=np.array([by_id[v]['class_id'] for v in dev])
        for rep,rotation,k,seed in batch['expected_runs']:
            run=new_directory(folder,f'{rep.lower()}-q{rotation}-k{k}-s{seed}')
            record=dict(directory=run.name,representation=rep,rotation_seed=rotation,k=k,support_seed=seed)
            prov=dict(**record,status='running',protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],evaluation_stage=stage,evaluation_ids=dev)
            write_json(run/'provenance.json',prov)
            try:
                support=get_support(context,seed);ids=[r['video_id'] for r in support];groups=sorted({r['source_group_id'] for r in support})
                if len(set(ids))!=5*k or set(ids)&set(dev) or set(groups)&{by_id[v]['source_group_id'] for v in dev}: raise ValueError('E1.3 support/source overlap/count')
                prototypes,visual=prototype_model(bank,rep,rotation,[positions[v] for v in ids],np.asarray([r['class_id'] for r in support],dtype=np.int64))
                scores=bank['q'][[positions[v] for v in dev]]@prototypes.T
                write_json(run/'resolved_config.json',config);write_csv(run/'support_set.csv',support,SUPPORT_FIELDS)
                np.savez_compressed(run/'prototypes.npz',prototypes=prototypes,visual_prototypes=visual)
                predictions=[dict(representation=rep,rotation_seed=rotation,k=k,support_seed=seed,run_id=run.name,evaluation_stage=stage,
                                  video_id=v,source_group_id=by_id[v]['source_group_id'],true_class_id=int(truth[i]),predicted_class_id=int(scores[i].argmax()),
                                  **{'score_'+c:float(scores[i,j]) for j,c in enumerate(CLASSES)}) for i,v in enumerate(dev)]
                write_csv(run/'predictions.csv',predictions,PRED_FIELDS);write_json(run/'metrics.json',classification_metrics(truth,scores))
                prov.update(status='success',support_ids=ids,support_sha256=digest(sorted(ids)),support_source_groups=groups,
                            feature_dimension=bank['q'].shape[1],hashes={name:sha256_file(run/name) for name in RUN_FILES})
                (run/'run.log').write_text('success; no gradient training; frozen CLIP development\n',encoding='utf-8')
            except Exception:
                prov['status']='failed';(run/'run.log').write_text(traceback.format_exc(),encoding='utf-8');raise
            finally:write_json(run/'provenance.json',prov)
            batch['runs'].append(record);write_json(folder/'batch.json',batch)
        if (sha256_file(index_path)!=batch['feature_index_sha256'] or sha256_file(index_path.parent/'features.npz')!=index['npz_sha256'] or
                tree_hashes(parent)!=index['parent_hashes'] or code_hashes()!=index['identity']['code']): raise ValueError('E1.3 inputs changed during evaluation')
        batch['status']='success'
    except Exception as exc:
        batch['status']='failed';batch['errors'].append(f'{type(exc).__name__}: {exc}')
        (folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:batch['finished_at']=now();write_json(folder/'batch.json',batch)
    return folder
