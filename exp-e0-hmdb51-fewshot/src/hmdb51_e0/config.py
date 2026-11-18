"""Strict E0 protocol and paths resolved independently of the shell's cwd."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import yaml

CLASSES = ("clap", "jump", "run", "walk", "wave")
ROOT = Path(__file__).resolve().parents[2]


def validate_config(config: dict) -> None:
    expected = {
        "experiment_id": "e0_hmdb51_v0", "protocol_version": "0.1", "classes": list(CLASSES),
        "official_split": 1, "official_tags": {"train": 1, "test": 2, "unused": 0},
        "development": {"videos_per_class": 10, "seed": 20261116, "source": "official_train_only"},
        "sampling": {"shots": [1, 3, 5], "support_seeds": list(range(10)), "nested_support_sets": True,
                     "unit": "video", "rng": "numpy_generator_pcg64"},
        "video": {"frame_count": 16, "frame_sampling": "stratified_midpoint",
                  "short_video_policy": "exclude_if_fewer_than_16_decodable_frames", "view": "full_frame", "audio_used": False},
        "encoder": {"family": "clip", "architecture": "ViT-B/32", "checkpoint_identity": None,
                    "frozen": True, "evaluation_mode": True, "preprocessing": "checkpoint_native"},
        "pooling": {"statistics": ["mean", "population_std"], "frame_l2_normalize": True,
                    "block_l2_normalize": True, "final_l2_normalize": True, "epsilon": 1e-6},
        "classifier": {"family": "multioutput_ridge", "target_encoding": "one_hot",
                       "objective_reduction": "mean_over_samples_sum_over_classes", "lambda": 1.0,
                       "fit_intercept": True, "penalize_intercept": False, "decision": "argmax"},
        "evaluation": {"current_target": "development", "final_test_enabled": False,
                       "metrics": ["accuracy", "macro_f1", "per_class_recall", "confusion_matrix"]},
    }
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported E0 protocol setting: {key}; expected {value!r}")
    if set(config) - set(expected) - {"status"}:
        raise ValueError("unknown protocol keys")


def load_config(path: Path | str = ROOT / "configs/e0.v0.example.yaml") -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def load_paths(path: Path | str, root: Path = ROOT) -> dict[str, Path]:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"dataset_root", "annotations_root", "weights_file", "feature_cache_root", "run_root", "report_root"}
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError(f"path configuration must contain {sorted(required)}")
    result = {}
    for key, value in values.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid path: {key}")
        p = Path(value).expanduser()
        result[key] = (p if p.is_absolute() else Path(root) / p).resolve()
    # Never allow generated outputs in the external dataset.
    dataset = result.get("dataset_root")
    if dataset is None:
        raise ValueError("dataset_root is required")
    for key in ("feature_cache_root", "run_root", "report_root"):
        if key not in result or result[key].is_relative_to(dataset):
            raise ValueError(f"unsafe output path: {key}")
    return result


def safe_video_path(root: Path, video_id: str) -> Path:
    if not isinstance(video_id, str) or "\\" in video_id or ":" in video_id:
        raise ValueError(f"unsafe video identity: {video_id!r}")
    parts = video_id.split("/")
    if len(parts) != 2 or parts[0] not in CLASSES or parts[1] in ("", ".", "..") or not parts[1].endswith(".avi"):
        raise ValueError(f"unsafe video identity: {video_id!r}")
    if PurePosixPath(video_id).as_posix() != video_id:
        raise ValueError(f"noncanonical video identity: {video_id!r}")
    root = Path(root).resolve()
    result = (root / video_id).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"video path escapes dataset root: {video_id}")
    return result
