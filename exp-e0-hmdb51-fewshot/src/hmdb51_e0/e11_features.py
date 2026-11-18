"""E1.1 Gram-preserving controls and fixed equal-energy feature fusion."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml

from .e1_features import concept_pool, cosine_scores
from .temporal_features import temporal_pool

DEFAULT_CONFIG = {
    'experiment_id': 'e11_geometry_fusion_diagnostics', 'version': 1,
    'rotation_seeds': [0, 1, 2, 3, 4],
    'geometry': 'common_right_orthogonal_E_times_Q',
    'fusion': 'separate_block_l2_equal_energy_then_final_l2',
    'primary_shot': 3, 'final_test_enabled': False,
    'audit': {'seed': 20261118, 'videos': 10, 'frame_positions_zero_based': [2, 7, 13],
              'concept_ids': ['pose01', 'pose02', 'pose03', 'pose10', 'pose11', 'pose12']},
}


def validate_config(config: dict) -> None:
    if config != DEFAULT_CONFIG:
        raise ValueError('E1.1 requires the fixed preregistered diagnostic configuration')


def load_config(path: Path) -> dict:
    config=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_config(config)
    return config


def orthogonal_matrix(dimension: int, seed: int) -> np.ndarray:
    if type(dimension) is not int or dimension < 1 or type(seed) is not int or seed < 0:
        raise ValueError('invalid orthogonal dimension/seed')
    q,r=np.linalg.qr(np.random.Generator(np.random.PCG64(seed)).normal(size=(dimension,dimension)))
    # Sign correction fixes the QR convention; this is O(d), not necessarily SO(d).
    return q * np.where(np.diag(r) < 0, -1., 1.)


def equal_fusion(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a,b=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or
            not a.shape[0] or not a.shape[1] or not b.shape[1] or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('fusion requires finite matching nonempty feature rows')
    norms=[np.linalg.norm(x,axis=1,keepdims=True) for x in (a,b)]
    if any(np.any(n <= 1e-6) for n in norms):
        raise ValueError('fusion rejects zero/degenerate feature blocks')
    joined=np.concatenate((a/norms[0],b/norms[1]),axis=1)
    return (joined/np.linalg.norm(joined,axis=1,keepdims=True)).astype(np.float32)


def derive_bank(parent: dict, config: dict) -> dict:
    validate_config(config)
    z,e,c1=parent['Z'],parent['E'],parent['C1']
    q=np.stack([orthogonal_matrix(int(e.shape[1]),s) for s in config['rotation_seeds']])
    directions=np.stack([(e.astype(np.float64)@matrix).astype(np.float32) for matrix in q])
    if not all(np.allclose(d@d.T,e@e.T,atol=1e-6,rtol=0) for d in directions):
        raise ValueError('rotated text geometry failed Gram preservation')
    g1=np.stack([np.stack([concept_pool(cosine_scores(video,d),'C1') for video in z]) for d in directions])
    v1=np.stack([temporal_pool(video,'V1') for video in z])
    return dict(Q=q,D=directions,V1=v1,G1=g1,VC1=equal_fusion(v1,c1),
                VG1=np.stack([equal_fusion(v1,control) for control in g1]))


def schedule(config: dict, reference_config: dict) -> list[list]:
    validate_config(config)
    conditions=[('G1',p) for p in config['rotation_seeds']]+[('VC1',None)]+[('VG1',p) for p in config['rotation_seeds']]
    return [[r,p,k,s] for r,p in conditions for k in reference_config['sampling']['shots']
            for s in reference_config['sampling']['support_seeds']]
