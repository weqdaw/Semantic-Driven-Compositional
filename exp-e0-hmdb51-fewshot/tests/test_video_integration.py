from pathlib import Path
import cv2
import numpy as np

from hmdb51_e0.config import CLASSES
from hmdb51_e0.inventory import audit_dataset
from hmdb51_e0.pipeline import extract_features, freeze_manifest, load_bundle, run_development, write_supports
from hmdb51_e0.provenance import read_json
from hmdb51_e0.video_io import audit_video, sampled_frames


def make_video(path, n, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for frame in range(n):
        image = np.full((32, 32, 3), value, dtype=np.uint8)
        image[:, :16, 0] = frame * 7
        writer.write(image)
    writer.release()


def test_real_decoder_uses_actual_frames_and_rgb(tmp_path):
    video = tmp_path / "clip.avi"
    make_video(video, 17, 40)
    record = audit_video(video)
    assert record["frame_count"] == 17 and record["decode_status"] == "ok"
    indices, frames = sampled_frames(video, 17)
    assert indices.tolist() == list(range(8)) + list(range(9, 17))
    assert len(frames) == 16 and frames[0].shape == (32, 32, 3)
    # Generator changes blue, so RGB conversion puts that change in channel 2.
    assert frames[-1][:, :10, 2].mean() > frames[0][:, :10, 2].mean() + 90
    make_video(tmp_path / "short.avi", 8, 50)
    assert audit_video(tmp_path / "short.avi")["decode_status"] == "too_short"


def test_video_to_smoke_pipeline_never_decodes_or_extracts_sealed_test(tmp_path, config):
    root, annotations = tmp_path / "videos", tmp_path / "annotations"
    annotations.mkdir()
    for class_id, name in enumerate(CLASSES):
        lines = []
        for i in range(12):
            make_video(root / name / f"{i:02}.avi", 17, 10 + class_id * 35 + i * 2)
            lines.append(f"{i:02}.avi 1")
        # Deliberately undecodable final-test bytes: metadata/hash audit is allowed,
        # but any accidental test-content decode would fail this complete pipeline.
        (root / name / "test.avi").write_bytes(f"sealed-{name}".encode())
        lines.append("test.avi 2")
        (annotations / f"{name}_test_split1.txt").write_text("\n".join(lines))
    audit = audit_dataset(root, annotations)
    assert audit["status"] == "success"
    bundle = freeze_manifest(audit, config, tmp_path / "manifests")
    supports = write_supports(load_bundle(bundle, config), config, tmp_path / "supports")
    class SyntheticEncoder:
        identity = {"synthetic_test_fixture": True, "dimension": 4}
        def encode(self, frames):
            means = np.array([f.mean(axis=(0, 1)) for f in frames])
            return np.column_stack([means, np.ones(16)])
        def verify_frozen(self):
            return {"unchanged": True, "synthetic_test_fixture": True}
    paths = {"dataset_root": root, "feature_cache_root": tmp_path / "features", "run_root": tmp_path / "runs"}
    index = extract_features(bundle, config, paths, SyntheticEncoder(), stage="smoke")
    entries = read_json(index)["entries"]
    assert len(entries) == 10
    assert not any(e["video_id"].endswith("test.avi") for e in entries)
    batch = run_development(bundle, supports, index, config, paths, stage="smoke")
    assert read_json(batch / "batch.json")["status"] == "success"
