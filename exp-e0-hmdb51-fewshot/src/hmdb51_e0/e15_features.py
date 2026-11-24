"""Fixed nonlinear instance-cache votes supplement the unchanged text anchor."""
import json
from pathlib import Path

import numpy as np
import yaml

from .e13_features import unit_rows


DEFAULT_CONFIG = dict(
    experiment_id='e15_instance_cache', version=1, shots=[3],
    support_seeds=list(range(10)), alpha=.1, beta=5.5,
    aggregation='per_class_mean', wrong_label_shift=1,
    score_rule='text_plus_alpha_cache_no_normalization', final_test_enabled=False,
    continuation={'min_macro_f1_gain': .02, 'min_positive_supports': 8},
)
_CONFIG_SIGNATURE = json.dumps(DEFAULT_CONFIG, sort_keys=True, allow_nan=False)
_MODEL_FIELDS = {
    'text_prototypes', 'visual_prototypes', 'support_keys', 'cache_keys',
    'true_labels', 'cache_labels', 'cache_values',
}


def validate_config(config):
    try:
        signature = json.dumps(config, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid E1.5 configuration') from exc
    if signature != _CONFIG_SIGNATURE:
        raise ValueError('E1.5 requires fixed alpha/beta, mean voting and30 evaluations')


def load_config(path: Path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_config(config)
    return config


def schedule(config):
    validate_config(config)
    return [[rep, 3, seed] for rep in ('IC1', 'IC2', 'IC3') for seed in config['support_seeds']]


def _unit_matrix(value, *, rows=None, dimension=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError('E1.5 requires real finite unit feature rows')
    array = raw.astype(np.float64, copy=False)
    if (array.ndim != 2 or not array.shape[0] or not array.shape[1] or
            (rows is not None and array.shape[0] != rows) or
            (dimension is not None and array.shape[1] != dimension) or
            not np.isfinite(array).all() or
            not np.allclose(np.linalg.norm(array, axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('E1.5 requires matching finite unit feature rows')
    return array


def _labels(value):
    array = np.asarray(value)
    if (array.shape != (15,) or array.dtype.kind not in 'iu' or
            any(np.count_nonzero(array == c) != 3 for c in range(5))):
        raise ValueError('E1.5 requires three integer-labeled support videos per class')
    return array.astype(np.int64, copy=True)


def cache_model(bank, representation, support_indices, labels, config=DEFAULT_CONFIG):
    validate_config(config)
    if representation not in ('IC1', 'IC2', 'IC3'):
        raise ValueError('invalid E1.5 instance-cache condition')
    positions = np.asarray(support_indices)
    q = np.asarray(bank['q'])
    if (q.ndim != 2 or positions.shape != (15,) or positions.dtype.kind not in 'iu' or
            len(set(positions.tolist())) != 15 or positions.min() < 0 or positions.max() >= len(q)):
        raise ValueError('E1.5 requires15 distinct valid support positions')
    true_labels = _labels(labels)
    support = _unit_matrix(q[positions], rows=15).copy()
    text = _unit_matrix(bank['T'], rows=5, dimension=support.shape[1]).copy()
    visual = unit_rows(np.stack([support[true_labels == c].mean(0) for c in range(5)]))
    keys = visual[true_labels].copy() if representation == 'IC2' else support.copy()
    cache_labels = ((true_labels + config['wrong_label_shift']) % 5
                    if representation == 'IC3' else true_labels.copy())
    values = np.eye(5, dtype=np.float64)[cache_labels]
    return dict(text_prototypes=text, visual_prototypes=visual, support_keys=support,
                cache_keys=keys, true_labels=true_labels, cache_labels=cache_labels,
                cache_values=values)


def score_branches(queries, model, config=DEFAULT_CONFIG):
    validate_config(config)
    if set(model) != _MODEL_FIELDS:
        raise ValueError('incomplete E1.5 cache model')
    q = _unit_matrix(queries)
    dimension = q.shape[1]
    text_keys = _unit_matrix(model['text_prototypes'], rows=5, dimension=dimension)
    _unit_matrix(model['visual_prototypes'], rows=5, dimension=dimension)
    _unit_matrix(model['support_keys'], rows=15, dimension=dimension)
    keys = _unit_matrix(model['cache_keys'], rows=15, dimension=dimension)
    _labels(model['true_labels'])
    labels = _labels(model['cache_labels'])
    values = np.asarray(model['cache_values'])
    if (values.shape != (15, 5) or values.dtype.kind not in 'iuf' or
            not np.array_equal(values, np.eye(5, dtype=np.float64)[labels])):
        raise ValueError('E1.5 cache values differ from one-hot cache labels')
    text = q @ text_keys.T
    cosines = q @ keys.T
    affinities = np.exp(-config['beta'] * (1. - cosines))
    votes = affinities / 3.
    cache = votes @ values
    final = text + config['alpha'] * cache
    if not all(np.isfinite(value).all() for value in (text, cosines, affinities, votes, cache, final)):
        raise ValueError('nonfinite E1.5 branch score or cache evidence')
    return final, text, cache, cosines, affinities, votes
