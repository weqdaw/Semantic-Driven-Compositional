"""Fixed text scores plus controlled visual residuals; no combined normalization."""
from pathlib import Path
import numpy as np
import yaml
from .e13_features import prototype_model

DEFAULT_CONFIG=dict(experiment_id='e14_text_anchor_scores',version=1,shots=[3],support_seeds=list(range(10)),
                    rotation_seeds=list(range(5)),lambdas={'A1':1.,'A2':.1,'A3':.1,'A4':.1},
                    wrong_visual_rows=[1,2,3,4,0],score_rule='text_plus_lambda_auxiliary_no_normalization',
                    final_test_enabled=False,continuation={'min_macro_f1_gain':.02,'min_positive_supports':8})


def validate_config(config):
    if config!=DEFAULT_CONFIG:raise ValueError('E1.4 requires fixed coefficients/mapping and80 evaluations')


def load_config(path: Path):
    config=yaml.safe_load(Path(path).read_text(encoding='utf-8'));validate_config(config);return config


def schedule(config):
    validate_config(config)
    return [[rep,rot,3,seed,config['lambdas'][rep]] for rep in ('A1','A2','A3','A4')
            for rot in (config['rotation_seeds'] if rep=='A4' else [None]) for seed in config['support_seeds']]


def score_model(bank,representation,rotation_seed,support_positions,support_labels):
    if representation not in DEFAULT_CONFIG['lambdas'] or ((representation=='A4')!=(rotation_seed is not None)):
        raise ValueError('invalid E1.4 condition/rotation')
    if rotation_seed is not None and (type(rotation_seed) is not int or rotation_seed not in range(5)):
        raise ValueError('invalid visual rotation seed')
    visual,_=prototype_model(bank,'P0',None,support_positions,support_labels)
    auxiliary=visual[DEFAULT_CONFIG['wrong_visual_rows']] if representation=='A3' else (visual@bank['Q'][rotation_seed] if representation=='A4' else visual.copy())
    text=bank['T'].copy();weights=text+DEFAULT_CONFIG['lambdas'][representation]*auxiliary
    return dict(text_prototypes=text,visual_prototypes=visual,auxiliary_prototypes=auxiliary,score_weights=weights)


def score_branches(q,model,lambda_):
    q=np.asarray(q,dtype=np.float64)
    t,a,w=model['text_prototypes'],model['auxiliary_prototypes'],model['score_weights']
    if (q.ndim!=2 or t.shape!=(5,q.shape[1]) or a.shape!=t.shape or w.shape!=t.shape or
            not np.isfinite(lambda_) or lambda_<=0 or not all(np.isfinite(x).all() for x in (q,t,a,w)) or
            not np.allclose(w,t+lambda_*a,atol=1e-12,rtol=1e-12)):
        raise ValueError('invalid anchored score model or branch coefficient')
    text=q@t.T;aux=q@a.T;final=text+lambda_*aux
    if not np.isfinite(final).all():raise ValueError('nonfinite E1.4 scores')
    return final,text,aux
