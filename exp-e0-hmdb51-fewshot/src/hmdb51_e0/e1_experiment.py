"""Independent E1 preparation and explicit development fitting."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import hashlib
import shutil
import traceback
import numpy as np
import yaml

from .ablation_experiment import reference_context,reference_hashes
from .clip_features import CHECKPOINT_SHA256
from .config import CLASSES,ROOT
from .e1_features import (concept_pool,cosine_scores,e1_code,load_catalog,random_directions,
                         validate_catalog,validate_e1_config,validate_text_alignment)
from .metrics import classification_metrics
from .pipeline import official_clip_condition
from .provenance import digest,environment,new_directory,now,read_json,sha256_file,source_identity,write_csv,write_json
from .ridge import fit_ridge
from .round2_features import Round2Cache,load_round2_features,validate_frozen_index
from .round2_io import safe_output
from .round2_protocol import SUPPORT_FIELDS
from .round2_runner import PREDICTIONS

OFFICIAL='official_openai_clip_vit_b32_text_concepts'


def frame_array_sha256(z: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(z,dtype='<f4',order='C').tobytes()).hexdigest()


def schedule(config: dict,reference_config: dict) -> list[list]:
    conditions=[('C0',None),('C1',None),*[(r,s) for r in ('R0','R1') for s in config['random_projection']['seeds']]]
    return [[r,p,k,s] for r,p in conditions for k in reference_config['sampling']['shots']
            for s in reference_config['sampling']['support_seeds']]


def check_e1(reference: Path,config: dict,catalog_path: Path,paths: dict,feature_root: Path) -> dict:
    validate_e1_config(config); catalog=load_catalog(catalog_path)
    batch,_,bundle,_=reference_context(reference)
    original_index=Path(feature_root)/batch['feature_identity']/'index-development.json'
    if sha256_file(original_index)!=batch['feature_index_sha256']:
        raise ValueError('reference feature bank differs from frozen second round')
    vectors,identity=load_round2_features(original_index,bundle,feature_root)
    official=official_clip_condition(identity)
    if official:
        checkpoint=paths.get('weights_file')
        if checkpoint is None or sha256_file(checkpoint)!=CHECKPOINT_SHA256:
            raise ValueError('E1 text branch requires the original official checkpoint')
        if importlib.util.find_spec('clip') is None or importlib.util.find_spec('torch') is None:
            raise ValueError('native CLIP text dependencies unavailable')
    m=len(catalog['concepts'])
    return dict(status='verified',evaluation_stage='development',final_test_enabled=False,
                protocol_id=batch['protocol_id'],image_feature_condition=batch['feature_condition'],
                parent_feature_identity=batch['feature_identity'],n_training_videos=len(vectors['V0']),
                n_development_videos=len(batch['evaluation_ids']),n_concepts=m,
                dimensions=dict(C0=2*m,C1=5*m,R0=2*m,R1=5*m),
                planned_new_runs=len(schedule(config,batch['config'])),reused_baseline_runs=60,
                text_encoding_executed=False,fits_executed=0)


def prepare_e1_features(reference: Path,config: dict,catalog_path: Path,paths: dict,
                        feature_root: Path,output: Path,*,text_encoder=None,device: str='cuda') -> Path:
    preflight=check_e1(reference,config,catalog_path,paths,feature_root)
    parent,_,bundle,_=reference_context(reference)
    hashes=reference_hashes(reference,parent)
    original=read_json(Path(reference)/'feature_index_snapshot.json')
    catalog=load_catalog(catalog_path)
    if text_encoder is None:
        from .e1_text import FrozenTextCLIP
        text_encoder=FrozenTextCLIP(paths['weights_file'],device=device)
    output=safe_output(output,paths,protected=(reference,feature_root))
    folder=new_directory(output,'e1-concept-features')
    target=folder/'index-development.json'
    identity=dict(schema='e1_concept_bank_v1',protocol_id=parent['protocol_id'],config=config,
                  catalog_sha256=digest(catalog),catalog_file_sha256=sha256_file(catalog_path),
                  parent_feature_identity=parent['feature_identity'],reference_evidence_sha256=digest(hashes),
                  encoder=original['identity']['encoder'],text_encoder=text_encoder.identity,
                  feature_code=e1_code(),numpy=np.__version__)
    index=dict(status='running',errors=[],started_at=now(),evaluation_stage='development',identity=identity,
               entries=original['entries'],reference_hashes=hashes,reference_batch=str(Path(reference).resolve()),
               frozen_verification=original['frozen_verification'],preflight=preflight)
    write_json(target,index)
    try:
        prompts=[c['description_en'] for c in catalog['concepts']]
        text=text_encoder.encode_text(prompts)
        frozen=text_encoder.verify_frozen()
        validate_text_alignment(identity['encoder'],identity['text_encoder'],frozen)
        index.update(text_frozen_verification=frozen,feature_condition=OFFICIAL if official_clip_condition(identity) else 'validation_only_unverified_features')
        m,d=len(prompts),identity['encoder']['dimension']
        if text.shape!=(m,d) or text.dtype!=np.float32 or not np.isfinite(text).all() or not np.allclose(np.linalg.norm(text,axis=1),1,atol=1e-5):
            raise ValueError('text embedding dimension/dtype/unit normalization mismatch')
        random=np.stack([random_directions(d,m,s) for s in config['random_projection']['seeds']])
        frame_bank,c,c0,c1=[],[],[],[]
        r0,r1=[[] for _ in random],[[] for _ in random]
        cache=Round2Cache(feature_root,original['identity'])
        for entry in index['entries']:
            z,_,_=cache.load(entry)
            frame_bank.append(z)
            scores=cosine_scores(z,text)
            c.append(scores); c0.append(concept_pool(scores,'C0')); c1.append(concept_pool(scores,'C1'))
            for j,directions in enumerate(random):
                values=cosine_scores(z,directions)
                r0[j].append(concept_pool(values,'R0')); r1[j].append(concept_pool(values,'R1'))
        identity['frame_array_sha256']=[frame_array_sha256(z) for z in frame_bank]
        np.savez_compressed(folder/'features.npz',Z=np.stack(frame_bank),E=text,P=random,C=np.stack(c),C0=np.stack(c0),C1=np.stack(c1),R0=np.array(r0),R1=np.array(r1))
        write_json(folder/'concept_catalog.json',catalog)
        shutil.copyfile(catalog_path,folder/'concept_catalog.yaml')
        if sha256_file(folder/'concept_catalog.yaml')!=identity['catalog_file_sha256']:
            raise ValueError('concept catalog changed during preparation')
        diagnostics=[]
        for j,concept in enumerate(catalog['concepts']):
            values=np.stack(c)[:,:,j]
            diagnostics.append(dict(concept_id=concept['id'],label_zh=concept['label_zh'],description_en=concept['description_en'],
                                    cosine_mean=float(values.mean()),cosine_std=float(values.std()),
                                    cosine_min=float(values.min()),cosine_max=float(values.max()),human_validation='not_performed'))
        write_csv(folder/'concept_diagnostics.csv',diagnostics,list(diagnostics[0]))
        dev=parent['evaluation_ids']
        rng=np.random.Generator(np.random.PCG64(config['audit_template']['seed']))
        selected=sorted(rng.choice(dev,size=min(len(dev),config['audit_template']['n_development_videos']),replace=False).tolist())
        audit=[dict(audit_id=f'video{i+1:03}',video_id=vid,concept_id=concept['id'],label_zh=concept['label_zh'],
                    description_en=concept['description_en'],visibility_note=concept['visibility_note'],
                    judgment='',reviewer='',reviewed_at='',evidence_note='')
               for i,vid in enumerate(selected) for concept in catalog['concepts']]
        write_csv(folder/'concept_audit_template.csv',audit,list(audit[0]))
        artifacts={name:sha256_file(folder/name) for name in ('concept_catalog.json','concept_catalog.yaml','concept_diagnostics.csv','concept_audit_template.csv')}
        if reference_hashes(reference,parent)!=hashes:
            raise ValueError('reference evidence changed during E1 preparation')
        index.update(status='success',n_concepts=m,frame_dimension=d,npz_sha256=sha256_file(folder/'features.npz'),
                     artifacts=artifacts,encoded_prompts_sha256=digest(prompts),human_concept_validation='not_performed')
    except Exception as exc:
        index['status']='failed'; index['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        index['finished_at']=now(); write_json(target,index)
    return target


def validate_e1_index(index: dict,reference: Path,parent: dict,folder: Path,*,current_code: bool) -> dict:
    validate_frozen_index(index)
    identity=index['identity']; validate_e1_config(identity['config'])
    catalog=read_json(Path(folder)/'concept_catalog.json'); validate_catalog(catalog)
    original=read_json(Path(reference)/'feature_index_snapshot.json')
    hashes=reference_hashes(reference,parent)
    validate_text_alignment(identity['encoder'],identity['text_encoder'],index['text_frozen_verification'])
    expected_condition=OFFICIAL if official_clip_condition(identity) else 'validation_only_unverified_features'
    if (identity['schema']!='e1_concept_bank_v1' or identity['protocol_id']!=parent['protocol_id']
            or identity['parent_feature_identity']!=parent['feature_identity'] or identity['encoder']!=original['identity']['encoder']
            or index['entries']!=original['entries'] or identity['reference_evidence_sha256']!=digest(hashes)
            or index['reference_hashes']!=hashes or index['frozen_verification']!=original['frozen_verification']
            or identity['catalog_sha256']!=digest(catalog) or index['n_concepts']!=len(catalog['concepts'])
            or index['frame_dimension']!=identity['encoder']['dimension'] or index['feature_condition']!=expected_condition
            or index['encoded_prompts_sha256']!=digest([c['description_en'] for c in catalog['concepts']])
            or index['human_concept_validation']!='not_performed'):
        raise ValueError('E1 concept/reference/index identity mismatch')
    for name,expected in index['artifacts'].items():
        path=(Path(folder)/name).resolve()
        if not path.is_relative_to(Path(folder).resolve()) or sha256_file(path)!=expected:
            raise ValueError('corrupt E1 catalog/diagnostic/audit artifact')
    if (sha256_file(Path(folder)/'concept_catalog.yaml')!=identity['catalog_file_sha256']
            or yaml.safe_load((Path(folder)/'concept_catalog.yaml').read_text(encoding='utf-8'))!=catalog):
        raise ValueError('E1 YAML/JSON concept catalog mismatch')
    if current_code and (identity['feature_code']!=e1_code() or identity['numpy']!=np.__version__):
        raise ValueError('E1 code/NumPy changed; prepare a new feature bank')
    return catalog


def load_bank(index: dict,path: Path) -> dict:
    if sha256_file(path)!=index['npz_sha256']: raise ValueError('corrupt E1 NPZ hash')
    with np.load(path,allow_pickle=False) as content:
        if set(content.files)!={'Z','E','P','C','C0','C1','R0','R1'}: raise ValueError('E1 array names mismatch')
        bank={k:content[k] for k in content.files}
    n,m,d=len(index['entries']),index['n_concepts'],index['frame_dimension']
    seeds=index['identity']['config']['random_projection']['seeds']
    shapes=dict(Z=(n,16,d),E=(m,d),P=(len(seeds),m,d),C=(n,16,m),C0=(n,2*m),C1=(n,5*m),R0=(len(seeds),n,2*m),R1=(len(seeds),n,5*m))
    for name,shape in shapes.items():
        value=bank[name]
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('E1 feature shape/dtype/finite mismatch')
        if name!='C' and not np.allclose(np.linalg.norm(value,axis=-1),1,atol=1e-5):
            raise ValueError('E1 directions/pooling unit norm mismatch')
    if np.max(np.abs(bank['C']))>1.00001: raise ValueError('invalid concept cosine range')
    if [frame_array_sha256(z) for z in bank['Z']]!=index['identity']['frame_array_sha256']:
        raise ValueError('source frame array hashes differ from verified parent Z')
    for j,seed in enumerate(seeds):
        if not np.array_equal(bank['P'][j],random_directions(d,m,seed)):
            raise ValueError('random projection differs from saved fixed seed')
    for i in range(n):
        if not np.allclose(bank['C'][i],cosine_scores(bank['Z'][i],bank['E']),atol=1e-7,rtol=0):
            raise ValueError('concept cosine scores differ from source frames/text directions')
        for rep in ('C0','C1'):
            if not np.allclose(bank[rep][i],concept_pool(bank['C'][i],rep),atol=1e-7,rtol=0):
                raise ValueError('concept pooling differs from stored cosine sequence')
        for j in range(len(seeds)):
            projected=cosine_scores(bank['Z'][i],bank['P'][j])
            for rep in ('R0','R1'):
                if not np.allclose(bank[rep][j,i],concept_pool(projected,rep),atol=1e-7,rtol=0):
                    raise ValueError('random control pooling differs from source frames/projection')
    return bank


def load_e1_features(index_path: Path,reference: Path) -> tuple[dict,dict]:
    parent,_,_,_=reference_context(reference)
    index=read_json(index_path)
    validate_e1_index(index,reference,parent,Path(index_path).parent,current_code=True)
    return index,load_bank(index,Path(index_path).parent/'features.npz')


def run_e1(reference: Path,index_path: Path,paths: dict,*,stage: str='development') -> Path:
    if stage!='development': raise ValueError('E1 development only; final_test remains sealed')
    reference,index_path=Path(reference),Path(index_path)
    parent,_,bundle,supports=reference_context(reference)
    index,bank=load_e1_features(index_path,reference)
    config=index['identity']['config']
    output=safe_output(paths['run_root'],paths,protected=(reference,index_path.parent))
    folder=new_directory(output,'e1-concept-development')
    batch=dict(status='running',errors=[],started_at=now(),evaluation_stage=stage,config=config,
               reference_config=parent['config'],protocol_id=parent['protocol_id'],feature_identity=digest(index['identity']),
               feature_condition=index['feature_condition'],feature_index_sha256=sha256_file(index_path),
               evaluation_ids=parent['evaluation_ids'],expected_runs=schedule(config,parent['config']),runs=[],source=source_identity(ROOT))
    write_json(folder/'batch.json',batch)
    try:
        for name,expected in index['reference_hashes'].items():
            target=folder/'reference'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(reference/name,target)
            if sha256_file(target)!=expected: raise ValueError('reference changed during E1 snapshot')
        shutil.copyfile(index_path,folder/'feature_index_snapshot.json')
        shutil.copyfile(index_path.parent/'features.npz',folder/'feature_bank.npz')
        for name in index['artifacts']: shutil.copyfile(index_path.parent/name,folder/name)
        validate_e1_index(index,folder/'reference',parent,folder,current_code=True)
        load_bank(index,folder/'feature_bank.npz')
        for name,expected in batch['source']['files'].items():
            target=folder/'code_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
            if sha256_file(target)!=expected: raise ValueError('source changed during E1 snapshot')
        write_json(folder/'environment.json',environment()); batch['environment_sha256']=sha256_file(folder/'environment.json')
        positions={e['video_id']:i for i,e in enumerate(index['entries'])}
        by_id={r['video_id']:r for r in bundle['rows']}
        dev=batch['evaluation_ids']; truth=np.array([by_id[v]['class_id'] for v in dev])
        for representation,projection_seed,k,seed in batch['expected_runs']:
            vectors=bank[representation] if projection_seed is None else bank[representation][config['random_projection']['seeds'].index(projection_seed)]
            support=sorted([r for r in supports['records'] if r['k']==k and r['support_seed']==seed],key=lambda r:(r['class_id'],r['rank_in_class']))
            ids=[r['video_id'] for r in support]; groups=sorted({r['source_group_id'] for r in support})
            if len(set(ids))!=5*k or set(ids)&set(dev) or set(groups)&{by_id[v]['source_group_id'] for v in dev}:
                raise ValueError('E1 support/source/evaluation count or overlap')
            run_dir=new_directory(folder,f'{representation.lower()}-p{projection_seed}-k{k}-s{seed}')
            record=dict(directory=run_dir.name,representation=representation,projection_seed=projection_seed,k=k,support_seed=seed)
            batch['runs'].append(record)
            provenance=dict(**record,run_id=run_dir.name,status='running',started_at=now(),
                            evaluation_stage=stage,protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],
                            source=batch['source'],evaluation_ids=dev,support_ids=ids,support_sha256=digest(sorted(ids)),
                            n_support=len(ids),support_source_groups=groups,n_support_sources=len(groups),feature_dimension=vectors.shape[1])
            write_json(run_dir/'resolved_config.json',dict(e1=config,reference=parent['config'],catalog_sha256=index['identity']['catalog_sha256']))
            write_csv(run_dir/'support_set.csv',support,SUPPORT_FIELDS)
            try:
                w,b=fit_ridge(vectors[[positions[v] for v in ids]],np.array([r['class_id'] for r in support]),parent['config']['classifier']['lambda'])
                scores=vectors[[positions[v] for v in dev]].astype(np.float64) @ w+b
                predictions=[dict(representation=representation,projection_seed=projection_seed,source_group_id=by_id[vid]['source_group_id'],
                                  run_id=run_dir.name,evaluation_stage=stage,support_seed=seed,k=k,video_id=vid,
                                  true_class_id=int(truth[i]),predicted_class_id=int(scores[i].argmax()),
                                  **{f'score_{name}':float(scores[i,j]) for j,name in enumerate(CLASSES)}) for i,vid in enumerate(dev)]
                metrics=dict(**classification_metrics(truth,scores),evaluation_stage=stage,representation=representation,
                             projection_seed=projection_seed,n_support=len(ids),n_support_sources=len(groups),feature_dimension=vectors.shape[1])
                write_csv(run_dir/'predictions.csv',predictions,['projection_seed',*PREDICTIONS])
                write_json(run_dir/'metrics.json',metrics); np.savez_compressed(run_dir/'ridge.npz',W=w,b=b)
                provenance.update(status='success',**{key:sha256_file(run_dir/name) for name,key in
                                   (('predictions.csv','predictions_sha256'),('metrics.json','metrics_sha256'),
                                    ('support_set.csv','support_csv_sha256'),('ridge.npz','ridge_sha256'))})
                (run_dir/'run.log').write_text('success; E1 development; encoder frozen\n',encoding='utf-8')
            except Exception:
                provenance['status']='failed'; (run_dir/'run.log').write_text(traceback.format_exc(),encoding='utf-8'); raise
            finally:
                provenance['finished_at']=now(); write_json(run_dir/'provenance.json',provenance)
            write_json(folder/'batch.json',batch)
        if sha256_file(index_path)!=batch['feature_index_sha256'] or reference_hashes(reference,parent)!=index['reference_hashes']:
            raise ValueError('E1 inputs changed during fitting')
        batch['status']='success'
    except Exception as exc:
        batch['status']='failed'; batch['errors'].append(f'{type(exc).__name__}: {exc}')
        (folder/'failure.log').write_text(traceback.format_exc(),encoding='utf-8'); raise
    finally:
        batch['finished_at']=now(); write_json(folder/'batch.json',batch)
    return folder
