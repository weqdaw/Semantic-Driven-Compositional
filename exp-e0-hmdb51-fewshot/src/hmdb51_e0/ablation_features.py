"""Additive, label-free controls over the unchanged second-round frame bank."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml

from .config import ROOT, safe_video_path
from .provenance import digest, sha256_file
from .temporal_features import temporal_pool

SHUFFLE_ALGORITHM = 'sha256_video_seed_pcg64_permutation_v1'
CONDITIONS = ('V1_repeat', 'V1_shuffle')
DEFAULT_CONFIG = dict(
    experiment_id='e0_hmdb51_round2_ablations', protocol_version='0.3',
    evaluation_stage='development', final_test_enabled=False,
    representations=dict(V1_repeat=['mean', 'population_std', 'mean', 'mean', 'mean'],
                         V1_shuffle=['mean', 'population_std', 'early_mean', 'middle_mean', 'late_mean']),
    temporal_segments=[6, 5, 5], block_l2_normalize=True, final_l2_normalize=True,
    epsilon=1e-6, statistics_dtype='float64', output_dtype='float32',
    shuffle=dict(algorithm=SHUFFLE_ALGORITHM, seeds=[0, 1, 2, 3, 4]),
    aggregation='mean_shuffle_metrics_within_support_then_support_sample_sd')


def validate_ablation_config(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError('invalid ablation config keys')
    if any(config[key] != value for key, value in DEFAULT_CONFIG.items() if key != 'shuffle'):
        raise ValueError('unsupported ablation protocol setting')
    shuffle = config['shuffle']
    if (not isinstance(shuffle, dict) or set(shuffle) != {'algorithm', 'seeds'}
            or shuffle['algorithm'] != SHUFFLE_ALGORITHM):
        raise ValueError('unsupported shuffle algorithm')
    seeds = shuffle['seeds']
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or seeds != sorted(set(seeds))):
        raise ValueError('shuffle seeds must be sorted unique nonnegative integers')


def load_ablation_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_ablation_config(config)
    return config


def frame_permutation(video_id: str, shuffle_seed: int) -> np.ndarray:
    safe_video_path(Path('.'), video_id)
    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError('shuffle seed must be a nonnegative integer')
    seed = int(digest(dict(algorithm=SHUFFLE_ALGORITHM, video_id=video_id, shuffle_seed=shuffle_seed)), 16)
    return np.random.Generator(np.random.PCG64(seed)).permutation(16).astype(np.int16)


def ablation_pool(z: np.ndarray, condition: str, *, video_id: str | None = None,
                  shuffle_seed: int | None = None) -> np.ndarray:
    temporal_pool(z, 'V0')  # Same finite/unit/frame-count validation as the parent.
    if condition == 'V1_shuffle':
        return temporal_pool(z[frame_permutation(video_id, shuffle_seed)], 'V1')
    if condition != 'V1_repeat' or shuffle_seed is not None:
        raise ValueError('invalid ablation condition or repeat shuffle seed')
    values = z.astype(np.float64)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
    blocks = [mean, std, mean, mean, mean]
    vector = np.concatenate([block / (np.linalg.norm(block) + 1e-6) for block in blocks])
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def ablation_code() -> dict:
    names = ('ablation_features.py', 'ablation_experiment.py', 'ablation_reporting.py', 'ablation_cli.py',
             'temporal_features.py', 'clip_features.py', 'ridge.py', 'metrics.py')
    return {name: sha256_file(ROOT / 'src/hmdb51_e0' / name) for name in names}
