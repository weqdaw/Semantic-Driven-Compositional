"""Identity-bound NPZ arrays with JSON metadata and byte integrity checks."""
from __future__ import annotations

import os
from pathlib import Path
import uuid
import numpy as np

from .clip_features import pool_normalized
from .config import safe_video_path
from .provenance import digest, read_json, sha256_file, write_json


class FeatureCache:
    def __init__(self, root: Path, identity: dict):
        self.identity = identity
        self.identity_hash = digest(identity)
        self.root = Path(root) / self.identity_hash

    def paths(self, expected: dict) -> tuple[Path, Path]:
        safe_video_path(Path("."), expected["video_id"])
        key = digest({"identity": self.identity, "video": expected})
        return self.root / f"{key}.npz", self.root / f"{key}.json"

    def exists(self, expected: dict) -> bool:
        npz, meta = self.paths(expected)
        return npz.exists() or meta.exists()

    @staticmethod
    def validate(z, vector):
        if z.dtype != np.float32 or vector.dtype != np.float32:
            raise ValueError("cache features must be float32")
        if z.ndim != 2 or z.shape[0] != 16 or vector.shape != (2 * z.shape[1],):
            raise ValueError("cache dimension mismatch")
        if not np.isfinite(z).all() or not np.isfinite(vector).all():
            raise ValueError("nonfinite cache features")
        if not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5) or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError("cache normalization mismatch")
        if not np.allclose(pool_normalized(z), vector, atol=2e-5):
            raise ValueError("cache pooling mismatch")

    def save(self, expected: dict, z: np.ndarray, vector: np.ndarray) -> None:
        self.validate(z, vector)
        npz, meta = self.paths(expected)
        if self.exists(expected):
            raise ValueError("refusing to overwrite an existing cache entry")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = npz.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, Z=z, V0=vector)
        os.replace(temporary, npz)
        write_json(meta, {"identity": self.identity, "video": expected, "dimension": z.shape[1],
                          "dtype": "float32", "npz_sha256": sha256_file(npz)})

    def load(self, expected: dict) -> tuple[np.ndarray, np.ndarray]:
        npz, meta = self.paths(expected)
        if not npz.is_file() or not meta.is_file():
            raise FileNotFoundError(f"missing complete feature cache for {expected['video_id']}")
        try:
            metadata = read_json(meta)
            if metadata["identity"] != self.identity or metadata["video"] != expected:
                raise ValueError("cache identity mismatch")
            if metadata["npz_sha256"] != sha256_file(npz):
                raise ValueError("corrupt cache: NPZ hash mismatch")
            with np.load(npz, allow_pickle=False) as content:
                if set(content.files) != {"Z", "V0"}:
                    raise ValueError("corrupt cache array fields")
                z, vector = content["Z"], content["V0"]
            self.validate(z, vector)
            if metadata["dimension"] != z.shape[1] or metadata["dtype"] != "float32":
                raise ValueError("cache metadata shape/dtype mismatch")
            return z, vector
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt/invalid cache for {expected['video_id']}: {exc}") from exc
