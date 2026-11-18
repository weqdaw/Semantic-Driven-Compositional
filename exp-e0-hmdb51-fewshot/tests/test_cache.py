import numpy as np
import pytest

from hmdb51_e0.clip_features import pool_features
from hmdb51_e0.feature_cache import FeatureCache


def test_cache_rejects_identity_change_and_corrupt_array(tmp_path):
    identity = {"checkpoint_sha256": "a" * 64, "decoder": "test", "protocol_id": "p"}
    cache = FeatureCache(tmp_path, identity)
    z, v = pool_features(np.tile([[1.0, 2, 3]], (16, 1)))
    expected = {"video_id": "clap/a.avi", "video_sha256": "b" * 64, "frame_indices": list(range(16))}
    cache.save(expected, z, v)
    read_z, read_v = cache.load(expected)
    np.testing.assert_array_equal(read_z, z)
    np.testing.assert_array_equal(read_v, v)
    with pytest.raises((ValueError, FileNotFoundError)):
        FeatureCache(tmp_path, {**identity, "checkpoint_sha256": "c" * 64}).load(expected)
    with pytest.raises((ValueError, FileNotFoundError)):
        cache.load({**expected, "video_sha256": "d" * 64})
    npz = next(tmp_path.rglob("*.npz"))
    npz.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt|hash"):
        cache.load(expected)


def test_native_preprocessing_identity_survives_process_restart():
    import os
    from pathlib import Path
    import subprocess
    import sys
    code = "import clip.clip as c; from hmdb51_e0.clip_features import preprocessing_identity; import json; print(json.dumps(preprocessing_identity(c._transform(224)),sort_keys=True))"
    env = {key: os.environ[key] for key in ("SystemRoot", "PATH", "TEMP", "TMP") if key in os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    first = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    second = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    assert first == second
