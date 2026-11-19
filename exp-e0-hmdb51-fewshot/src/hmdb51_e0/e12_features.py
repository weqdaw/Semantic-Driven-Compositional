"""E1.2 support-only scalar variance matching; no per-video renormalization."""
from pathlib import Path
import numpy as np
import yaml

DEFAULT_CONFIG=dict(experiment_id='e12_support_scale_audit',version=1,shots=[3],support_seeds=list(range(10)),
                    control_seeds=list(range(5)),scale_epsilon=1e-6,normalization='support_centered_rms_norm',
                    fusion='concatenate_divide_sqrt2_no_row_l2',final_test_enabled=False,
                    continuation=dict(min_macro_f1_gain=.02,min_positive_supports=8))
BLOCKS={'V1_lambda2':('V1',),'SV1':('V1',),'SC1':('C1',),'SG1':('G1',),'SR1':('R1',),
        'SVC1':('V1','C1'),'SVG1':('V1','G1'),'SVR1':('V1','R1')}
RANDOM=('SG1','SR1','SVG1','SVR1')


def validate_config(config):
    if config!=DEFAULT_CONFIG: raise ValueError('E1.2 requires the fixed 3-shot/240-fit diagnostic protocol')


def load_config(path: Path):
    config=yaml.safe_load(Path(path).read_text(encoding='utf-8')); validate_config(config)
    return config


def schedule(config):
    validate_config(config)
    return [[rep,control,3,seed,2. if rep=='V1_lambda2' else 1.] for rep in BLOCKS
            for control in (config['control_seeds'] if rep in RANDOM else [None]) for seed in config['support_seeds']]


def build_vectors(raw: dict, representation: str, control_seed: int | None, support_positions: list[int]):
    if representation not in BLOCKS or ((representation in RANDOM) != (control_seed is not None)):
        raise ValueError('invalid E1.2 representation/control identity')
    if control_seed is not None and (type(control_seed) is not int or control_seed not in range(5)):
        raise ValueError('invalid E1.2 direction seed')
    positions=np.asarray(support_positions)
    if positions.ndim!=1 or not positions.size or positions.dtype.kind not in 'iu' or len(set(positions.tolist()))!=len(positions):
        raise ValueError('invalid or repeated support positions')
    transformed=[]; params={}; diagnostics=[]; n=None
    for block in BLOCKS[representation]:
        x=np.asarray(raw[block][control_seed] if block in ('G1','R1') else raw[block],dtype=np.float64)
        if (x.ndim!=2 or not x.shape[1] or not np.isfinite(x).all() or positions.min()<0 or positions.max()>=len(x)
                or (n is not None and n!=len(x))):
            raise ValueError('invalid finite/raw block/support shape')
        n=len(x); sample=x[positions]; mean=sample.mean(0)
        variance=float(np.mean(np.sum((sample-mean)**2,axis=1)))
        if representation=='V1_lambda2': result=x.copy()
        else:
            scale=float(np.sqrt(variance))
            if not np.isfinite(scale) or scale<=DEFAULT_CONFIG['scale_epsilon']:
                raise ValueError(f'degenerate support scale for {block}: {scale}')
            params[block+'_mean']=mean; params[block+'_scale']=np.asarray(scale,dtype=np.float64)
            result=(x-mean)/scale
        after=result[positions]-result[positions].mean(0)
        diagnostics.append(dict(block=block,dimension=x.shape[1],n_support=len(positions),
                                normalization_applied=representation!='V1_lambda2',before_centered_variance=variance,
                                after_centered_variance=float(np.mean(np.sum(after**2,axis=1)))))
        transformed.append(result)
    vectors=transformed[0] if len(transformed)==1 else np.concatenate(transformed,axis=1)/np.sqrt(2.)
    if not np.isfinite(vectors).all(): raise ValueError('nonfinite E1.2 transformed features')
    return vectors,params,diagnostics
