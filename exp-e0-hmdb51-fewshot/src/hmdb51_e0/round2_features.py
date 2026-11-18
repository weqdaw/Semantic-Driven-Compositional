"""Verified import of legacy Z into a separate shared Z/V0/V1 cache."""
from __future__ import annotations

import os
from pathlib import Path
import re
import uuid
import numpy as np

from .config import ROOT, safe_video_path
from .feature_cache import FeatureCache
from .pipeline import load_feature_vectors, official_clip_condition
from .provenance import digest, read_json, write_json, sha256_file, now
from .round2_io import safe_output
from .round2_protocol import load_grouped_bundle, validate_grouped_bundle
from .source_groups import training_rows
from .temporal_features import temporal_pool
from .video_io import midpoint_indices

CODE_FILES = ('clip_features.py', 'video_io.py', 'feature_cache.py', 'temporal_features.py', 'round2_features.py')


def feature_code() -> dict:
    return {name: sha256_file(ROOT / 'src/hmdb51_e0' / name) for name in CODE_FILES}


def expected_entry(row: dict) -> dict:
    return dict(video_id=row['video_id'], video_sha256=row['sha256'],
                frame_indices=midpoint_indices(row['frame_count']).tolist())


def validate_frozen_index(index: dict) -> None:
    frozen = index.get('frozen_verification')
    if (index.get('status') != 'success' or index.get('errors', []) != []
            or index.get('evaluation_stage') != 'development' or not isinstance(frozen, dict)
            or frozen.get('unchanged') is not True):
        raise ValueError('feature index success/stage/frozen verification failed')
    if official_clip_condition(index.get('identity', {})):
        before = index['identity']['encoder'].get('parameter_sha256_before')
        if (not isinstance(before, str) or re.fullmatch(r'[0-9a-f]{64}', before) is None
                or frozen.get('before') != before or frozen.get('after') != before):
            raise ValueError('official encoder freeze hashes are missing or inconsistent')


class Round2Cache:
    def __init__(self, root: Path, identity: dict):
        self.identity = identity
        self.root = Path(root) / digest(identity)

    def paths(self, entry: dict) -> tuple[Path, Path]:
        safe_video_path(Path('.'), entry['video_id'])
        stem = digest(dict(identity=self.identity, video=entry))
        return self.root / f'{stem}.npz', self.root / f'{stem}.json'

    @staticmethod
    def validate(z, v0, v1):
        if any(value.dtype != np.float32 for value in (z, v0, v1)):
            raise ValueError('round2 cache must be float32')
        for name, vector, multiplier in (('V0', v0, 2), ('V1', v1, 5)):
            expected = temporal_pool(z, name)
            if (vector.shape != (multiplier * z.shape[1],) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)
                    or not np.allclose(vector, expected, atol=2e-5)):
                raise ValueError('round2 feature shape/normalization/pooling mismatch')

    def save(self, entry: dict, z: np.ndarray) -> None:
        v0, v1 = temporal_pool(z, 'V0'), temporal_pool(z, 'V1')
        self.validate(z, v0, v1)
        npz, meta = self.paths(entry)
        if npz.exists() or meta.exists():
            # Resume only complete, verified entries; never replace a corrupt one.
            loaded = self.load(entry)
            if not np.array_equal(loaded[0], z):
                raise ValueError('existing derived cache has different Z')
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = npz.with_suffix(f'.{uuid.uuid4().hex}.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, Z=z, V0=v0, V1=v1)
        os.replace(temporary, npz)
        write_json(meta, dict(identity=self.identity, video=entry, npz_sha256=sha256_file(npz),
                             frame_dimension=z.shape[1], dtype='float32'))

    def load(self, entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        npz, meta = self.paths(entry)
        try:
            metadata = read_json(meta)
            if metadata['identity'] != self.identity or metadata['video'] != entry:
                raise ValueError('round2 cache identity mismatch')
            if metadata['npz_sha256'] != sha256_file(npz):
                raise ValueError('corrupt round2 cache: NPZ hash mismatch')
            with np.load(npz, allow_pickle=False) as content:
                if set(content.files) != {'Z', 'V0', 'V1'}:
                    raise ValueError('round2 cache array names mismatch')
                z, v0, v1 = content['Z'], content['V0'], content['V1']
            self.validate(z, v0, v1)
            if metadata['frame_dimension'] != z.shape[1] or metadata['dtype'] != 'float32':
                raise ValueError('round2 cache dimension/dtype metadata mismatch')
            return z, v0, v1
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt/invalid round2 cache for {entry['video_id']}: {exc}") from exc


def prepare_round2_features(manifest: Path, parent_index_path: Path, config: dict, paths: dict, output: Path) -> Path:
    bundle = load_grouped_bundle(manifest, config)
    parent = bundle['parent_manifest']
    output = safe_output(output, paths)
    parent_hash = sha256_file(parent_index_path)
    parent_bytes = Path(parent_index_path).read_bytes()
    index = read_json(parent_index_path)
    validate_frozen_index(index)
    identity = index['identity']
    for key, value in (('video', parent['config']['video']), ('pooling', parent['config']['pooling']), ('decoder', parent['decoder'])):
        if identity.get(key) != value:
            raise ValueError(f'parent frame condition mismatch: {key}')
    if not set(('clip_features.py', 'video_io.py', 'feature_cache.py')).issubset(identity.get('feature_code', {})):
        raise ValueError('parent feature implementation hashes missing')
    rows = training_rows(bundle)
    required = {r['video_id'] for r in rows}
    # Legacy loader verifies parent protocol, complete development index, code,
    # frozen record, video SHA/indices, metadata, NPZ hashes and V0 pooling.
    load_feature_vectors(parent_index_path, parent, paths, required, 'development')
    if len(index['entries']) != len(required) or {e['video_id'] for e in index['entries']} != required:
        raise ValueError('parent index must cover precisely eligible training videos')
    derived_identity = dict(schema='round2_Z_V0_V1_v1', protocol_id=bundle['protocol_id'],
                            parent_index_sha256=parent_hash, parent_feature_identity=digest(identity),
                            encoder=identity['encoder'], decoder=identity['decoder'], video=identity['video'],
                            representations=config['representations'], feature_code=feature_code(), numpy=np.__version__)
    cache = Round2Cache(output, derived_identity)
    target = cache.root / 'index-development.json'
    prepared = dict(status='running', evaluation_stage='development', identity=derived_identity, entries=[], errors=[],
                    started_at=now(), frozen_verification=index['frozen_verification'],
                    feature_condition='official_openai_clip_vit_b32' if official_clip_condition(identity) else 'validation_only_unverified_features')
    source = FeatureCache(paths['feature_cache_root'], identity)
    # Check raw identities before producing a new successful feature bank; no decoding.
    for row in rows:
        video = safe_video_path(paths['dataset_root'], row['relative_path'])
        if not video.is_file() or video.stat().st_size != row['bytes'] or sha256_file(video) != row['sha256']:
            raise ValueError(f"raw training video identity changed: {row['video_id']}")
    if target.exists():
        previous = read_json(target)
        if previous.get('status') == 'success':
            load_round2_features(target, bundle, output)
            return target
        if (previous.get('status') not in ('failed', 'running') or previous.get('identity') != derived_identity
                or previous.get('evaluation_stage') != 'development'
                or previous.get('frozen_verification') != index['frozen_verification']):
            raise ValueError('cannot resume a mismatched or invalid derived index')
        entries = previous.get('entries', [])
        expected = [expected_entry(row) for row in rows]
        if entries != expected[:len(entries)]:
            raise ValueError('partial derived index is not a valid prefix')
        for entry in entries:
            cache.load(entry)
        if sha256_file(cache.root / 'parent_feature_index_snapshot.json') != parent_hash:
            raise ValueError('partial derived parent snapshot hash mismatch')
        previous_path = cache.root / f'previous-attempt-{uuid.uuid4().hex}.json'
        previous_path.write_bytes(target.read_bytes())
        prepared['previous_attempt'] = previous_path.name
    cache.root.mkdir(parents=True, exist_ok=True)
    snapshot = cache.root / 'parent_feature_index_snapshot.json'
    if snapshot.exists() and snapshot.read_bytes() != parent_bytes:
        raise ValueError('refusing to overwrite a different parent feature snapshot')
    snapshot.write_bytes(parent_bytes)
    write_json(target, prepared)
    try:
        for row in rows:
            entry = expected_entry(row)
            z, _ = source.load(entry)
            cache.save(entry, z)
            prepared['entries'].append(entry)
        if sha256_file(parent_index_path) != parent_hash or sha256_file(snapshot) != parent_hash:
            raise ValueError('parent feature index changed during import')
        prepared['status'] = 'success'
    except Exception as exc:
        prepared['status'] = 'failed'
        prepared['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        prepared['finished_at'] = now()
        write_json(target, prepared)
    return target


def load_round2_features(index_path: Path, bundle: dict, root: Path) -> tuple[dict, dict]:
    validate_grouped_bundle(bundle, bundle['config'])
    index = read_json(index_path)
    validate_frozen_index(index)
    identity = index['identity']
    parent_snapshot = Path(index_path).parent / 'parent_feature_index_snapshot.json'
    if (index['status'] != 'success' or index.get('errors') or index['evaluation_stage'] != 'development'
            or not index['frozen_verification'].get('unchanged') or identity['schema'] != 'round2_Z_V0_V1_v1'
            or identity['protocol_id'] != bundle['protocol_id'] or identity['representations'] != bundle['config']['representations']
            or identity['feature_code'] != feature_code() or identity['numpy'] != np.__version__
            or sha256_file(parent_snapshot) != identity['parent_index_sha256']):
        raise ValueError('round2 feature stage/freeze/protocol/code/parent identity mismatch')
    parent_index = read_json(parent_snapshot)
    validate_frozen_index(parent_index)
    parent_identity = parent_index['identity']
    if (digest(parent_identity) != identity['parent_feature_identity']
            or parent_identity['protocol_id'] != bundle['parent_manifest']['protocol_id']
            or parent_identity['encoder'] != identity['encoder']
            or parent_index.get('status') != 'success' or parent_index.get('evaluation_stage') != 'development'
            or parent_index['frozen_verification'] != index['frozen_verification']):
        raise ValueError('round2 parent snapshot/freeze identity mismatch')
    if (parent_identity['decoder'] != bundle['parent_manifest']['decoder'] or parent_identity['decoder'] != identity['decoder']
            or parent_identity['video'] != bundle['parent_manifest']['config']['video'] or parent_identity['video'] != identity['video']
            or parent_identity['pooling'] != bundle['parent_manifest']['config']['pooling']):
        raise ValueError('round2 parent encoding/decoding conditions mismatch')
    condition = 'official_openai_clip_vit_b32' if official_clip_condition(identity) else 'validation_only_unverified_features'
    if index['feature_condition'] != condition:
        raise ValueError('round2 feature condition mismatch')
    expected = [expected_entry(row) for row in training_rows(bundle)]
    if index['entries'] != expected or sorted(parent_index['entries'], key=lambda e: e['video_id']) != sorted(expected, key=lambda e: e['video_id']):
        raise ValueError('round2 feature coverage/video identity mismatch')
    cache = Round2Cache(root, identity)
    if Path(index_path).resolve().parent != cache.root.resolve():
        raise ValueError('round2 feature root/index location mismatch')
    vectors = {'V0': {}, 'V1': {}}
    dimension = identity['encoder'].get('dimension')
    if dimension is None or (official_clip_condition(identity) and dimension != 512):
        raise ValueError('round2 frame dimension missing or incorrect')
    for entry in expected:
        z, v0, v1 = cache.load(entry)
        if z.shape[1] != dimension:
            raise ValueError('round2 actual frame dimension differs from encoder')
        vectors['V0'][entry['video_id']], vectors['V1'][entry['video_id']] = v0, v1
    return vectors, identity
