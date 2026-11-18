"""Concept similarities and matched nonsemantic controls; no task-label input."""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import yaml

from .config import CLASSES, ROOT
from .pipeline import official_clip_condition
from .provenance import sha256_file
from .temporal_features import temporal_pool

DEFAULT_CONFIG = dict(experiment_id='e1_hmdb51_concept_baselines',version='1.0',evaluation_stage='development',
    final_test_enabled=False,concept_score='cosine_float64_to_float32_without_frame_renormalization',
    pooling='mean_population_std_and_optional_6_5_5_segment_means',block_l2_normalize=True,
    final_l2_normalize=True,epsilon=1e-6,random_projection=dict(algorithm='pcg64_normal_unit_rows_v1',seeds=[0,1,2,3,4]),
    aggregation='mean_projection_metrics_within_support_then_support_sample_sd',
    audit_template=dict(n_development_videos=30,seed=20261118))


def validate_e1_config(config: dict) -> None:
    if not isinstance(config,dict) or set(config)!=set(DEFAULT_CONFIG):
        raise ValueError('invalid E1 configuration keys')
    if any(config[k]!=v for k,v in DEFAULT_CONFIG.items() if k!='random_projection') or config['final_test_enabled'] is not False:
        raise ValueError('unsupported E1 protocol setting')
    random=config['random_projection']
    if (not isinstance(random,dict) or set(random)!={'algorithm','seeds'}
            or random['algorithm']!='pcg64_normal_unit_rows_v1'):
        raise ValueError('invalid random projection configuration')
    seeds=random['seeds']
    if not isinstance(seeds,list) or any(type(s) is not int for s in seeds) or seeds!=[0,1,2,3,4]:
        raise ValueError('E1 version 1.0 fixes exactly five projection seeds: 0,1,2,3,4')


def load_e1_config(path: Path) -> dict:
    config=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_e1_config(config)
    return config


def validate_catalog(catalog: dict) -> None:
    keys={'catalog_id','version','status','provenance','development_context','final_test_used','concepts'}
    if (not isinstance(catalog,dict) or set(catalog)!=keys or catalog['final_test_used'] is not False
            or any(not isinstance(catalog[k],str) or not catalog[k].strip() for k in keys-{'concepts','final_test_used'})):
        raise ValueError('invalid concept catalog metadata')
    concepts=catalog['concepts']
    if not isinstance(concepts,list) or not 2<=len(concepts)<=64:
        raise ValueError('concept catalog requires 2..64 concepts')
    ids,prompts=set(),set()
    for concept in concepts:
        if (not isinstance(concept,dict) or set(concept)!={'id','label_zh','description_en','visibility_note'}
                or any(not isinstance(v,str) or not v.strip() for v in concept.values())
                or re.fullmatch(r'[a-z][a-z0-9_]*',concept['id']) is None):
            raise ValueError('invalid concept entry')
        prompt=concept['description_en'].strip().casefold()
        if concept['id'] in ids or prompt in prompts or set(re.findall(r'[a-z]+',prompt)) & set(CLASSES):
            raise ValueError('duplicate concept or direct task label prompt')
        ids.add(concept['id']); prompts.add(prompt)


def load_catalog(path: Path) -> dict:
    catalog=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_catalog(catalog)
    return catalog


def cosine_scores(z: np.ndarray, directions: np.ndarray) -> np.ndarray:
    temporal_pool(z,'V0')  # Validate unchanged normalized 16 x d image features.
    directions=np.asarray(directions)
    if (directions.ndim!=2 or directions.shape[1]!=z.shape[1] or directions.shape[0]<1
            or not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-5)):
        raise ValueError('directions must be finite unit text/random rows in the same embedding space')
    return (z.astype(np.float64) @ directions.astype(np.float64).T).astype(np.float32)


def concept_pool(scores: np.ndarray, representation: str) -> np.ndarray:
    values=np.asarray(scores,dtype=np.float64)
    if (values.ndim!=2 or values.shape[0]!=16 or values.shape[1]<1 or not np.isfinite(values).all()
            or np.max(np.abs(values))>1.00001 or representation not in ('C0','C1','R0','R1')):
        raise ValueError('invalid finite 16 x m cosine sequence or E1 representation')
    blocks=[values.mean(0),values.std(0,ddof=0)]
    if representation in ('C1','R1'):
        blocks.extend([values[:6].mean(0),values[6:11].mean(0),values[11:].mean(0)])
    vector=np.concatenate([b/(np.linalg.norm(b)+1e-6) for b in blocks])
    norm=np.linalg.norm(vector)
    if norm<=1e-6 or not np.isfinite(norm): raise ValueError('zero or nonfinite concept feature')
    return (vector/norm).astype(np.float32)


def random_directions(dimension: int, n_concepts: int, seed: int) -> np.ndarray:
    if any(type(v) is not int or v<1 for v in (dimension,n_concepts)) or type(seed) is not int or seed<0:
        raise ValueError('invalid projection dimension, concept count or seed')
    matrix=np.random.Generator(np.random.PCG64(seed)).normal(size=(n_concepts,dimension))
    return (matrix/np.linalg.norm(matrix,axis=1,keepdims=True)).astype(np.float32)


def validate_text_alignment(image: dict, text: dict, frozen: dict) -> None:
    if image.get('synthetic_test_fixture') is True and text.get('synthetic_test_fixture') is True:
        if image.get('dimension')==text.get('dimension') and frozen.get('unchanged') is True: return
        raise ValueError('synthetic text/image fixture mismatch')
    keys=('architecture','checkpoint_sha256','dimension','parameter_sha256_before','clip_source_sha256')
    if (not official_clip_condition({'encoder':image}) or text.get('text_encoder')!='native_openai_clip'
            or any(image.get(k)!=text.get(k) for k in keys) or frozen.get('unchanged') is not True):
        raise ValueError('text/image checkpoint or embedding alignment mismatch')
    before=image.get('parameter_sha256_before')
    tokenizer=text.get('tokenizer',{})
    if (not isinstance(before,str) or re.fullmatch(r'[0-9a-f]{64}',before) is None
            or frozen.get('before')!=before or frozen.get('after')!=before
            or tokenizer.get('context_length')!=77 or tokenizer.get('truncate') is not False
            or any(re.fullmatch(r'[0-9a-f]{64}',str(tokenizer.get(k))) is None for k in ('bpe_sha256','tokenizer_source_sha256'))):
        raise ValueError('text freeze/tokenizer verification mismatch')


def e1_code() -> dict:
    names=('e1_features.py','e1_text.py','e1_experiment.py','e1_reporting.py','e1_cli.py',
           'clip_features.py','temporal_features.py','ridge.py','metrics.py','ablation_experiment.py','round2_features.py')
    return {name:sha256_file(ROOT/'src/hmdb51_e0'/name) for name in names}
