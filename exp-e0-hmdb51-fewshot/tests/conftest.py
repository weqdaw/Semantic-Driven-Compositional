from pathlib import Path
import copy

import pytest
import yaml


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return copy.deepcopy(yaml.safe_load((root / "configs/e0.v0.example.yaml").read_text(encoding="utf-8")))


@pytest.fixture
def eligible_rows():
    from hmdb51_e0.config import CLASSES
    rows = []
    for class_id, name in enumerate(CLASSES):
        for i in range(18):
            vid = f"{name}/{i:03}.avi"
            tag = 1 if i < 16 else (2 if i == 16 else 0)
            rows.append(dict(video_id=vid, relative_path=vid, class_name=name,
                             class_id=class_id, official_split=1, official_tag=tag,
                             bytes=20, sha256=f"{class_id * 100 + i:064x}",
                             frame_count=32 if tag == 1 else None,
                             duration_seconds=1.0 if tag == 1 else None,
                             decode_status="ok" if tag == 1 else "not_checked",
                             exclusion_reason=""))
    return rows
