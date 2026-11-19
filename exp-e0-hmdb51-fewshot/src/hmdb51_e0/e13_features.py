"""Fixed class-text references and support-only cosine prototypes."""
from pathlib import Path
import numpy as np
import yaml

from .config import CLASSES
from .e11_features import orthogonal_matrix

DEFAULT_CONFIG=dict(experiment_id='e13_class_text_prototypes',version=1,classes=list(CLASSES),
                    prompts=['a photo of a person '+v+'.' for v in ('clapping','jumping','running','walking','waving')],
                    shots=[3],support_seeds=list(range(10)),rotation_seeds=list(range(5)),
                    wrong_text_rows=[1,2,3,4,0],fusion='unit_visual_plus_unit_text_then_unit',
                    video_pool='unit_mean_16_unit_frames',final_test_enabled=False)


def validate_config(config):
    if config!=DEFAULT_CONFIG: raise ValueError('E1.3 requires the fixed prompts, mapping and 81-evaluation protocol')


def load_config(path: Path):
    config=yaml.safe_load(Path(path).read_text(encoding='utf-8')); validate_config(config)
    return config


def unit_rows(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not x.shape[1] or not np.isfinite(x).all(): raise ValueError('expected finite feature rows')
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    if np.any(norms<=1e-6): raise ValueError('degenerate feature/prototype normalization')
    return x/norms


def schedule(config):
    validate_config(config)
    conditions=[('P0',None),('P1',None),('P2',None),*[('P3',s) for s in config['rotation_seeds']]]
    return [['T0',None,0,None]]+[[r,p,3,s] for r,p in conditions for s in config['support_seeds']]


def derive_bank(z,e,config):
    validate_config(config); z=np.asarray(z,dtype=np.float64); e=np.asarray(e)
    if (z.ndim!=3 or z.shape[1]!=16 or not len(z) or not np.isfinite(z).all() or
            not np.allclose(np.linalg.norm(z,axis=2),1,atol=1e-5,rtol=0) or e.shape!=(5,z.shape[2]) or
            not np.isfinite(e).all() or not np.allclose(np.linalg.norm(e,axis=1),1,atol=1e-5,rtol=0)):
        raise ValueError('E1.3 requires matching unit frame/text embeddings')
    t=unit_rows(e); q=np.stack([orthogonal_matrix(int(z.shape[2]),s) for s in config['rotation_seeds']])
    d=np.stack([t@matrix for matrix in q])
    if not all(np.allclose(x@x.T,t@t.T,atol=1e-12,rtol=0) for x in d): raise ValueError('text Gram mismatch')
    return dict(q=unit_rows(z.mean(1)),E=e.astype(np.float32),T=t,Q=q,D=d,
                wrong_text_rows=np.asarray(config['wrong_text_rows'],dtype=np.int64))


def prototype_model(bank,representation,rotation_seed,support_positions,support_labels):
    if representation not in ('T0','P0','P1','P2','P3') or ((representation=='P3')!=(rotation_seed is not None)):
        raise ValueError('invalid E1.3 condition/rotation')
    if rotation_seed is not None and (type(rotation_seed) is not int or rotation_seed not in range(5)):
        raise ValueError('invalid E1.3 rotation seed')
    positions=np.asarray(support_positions); labels=np.asarray(support_labels)
    if representation=='T0':
        if positions.size or labels.size: raise ValueError('T0 has no support examples')
        return bank['T'].copy(),np.empty((0,bank['q'].shape[1]),dtype=np.float64)
    if (positions.shape!=(15,) or positions.dtype.kind not in 'iu' or len(set(positions.tolist()))!=15 or
            positions.min()<0 or positions.max()>=len(bank['q']) or labels.shape!=(15,) or labels.dtype.kind not in 'iu' or
            any(np.count_nonzero(labels==c)!=3 for c in range(5))):
        raise ValueError('prototype requires exactly three distinct support videos per class')
    sample=bank['q'][positions]
    visual=unit_rows(np.stack([sample[labels==c].mean(0) for c in range(5)]))
    if representation=='P0': return visual.copy(),visual
    text=bank['T'] if representation=='P1' else (bank['T'][bank['wrong_text_rows']] if representation=='P2' else bank['D'][rotation_seed])
    return unit_rows(visual+text),visual
