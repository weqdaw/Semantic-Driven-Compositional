"""Fixed E2.0 audit identities and strict human annotation contracts."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import yaml

from .provenance import digest

CONCEPTS = ('pose01', 'pose02', 'pose03', 'pose10', 'pose11', 'pose12')
HUMAN_FIELDS = ('judgment', 'reason_code', 'reviewer', 'reviewed_at', 'evidence', 'elapsed_seconds')
REASONS = ('clear', 'crop_missing', 'occlusion', 'blur', 'ambiguous_person', 'ground_unknown', 'other')
VIEWS = ('crop', 'original', 'review-crop', 'review-original', 'adjudication')
DEFAULT_CONFIG = dict(
    experiment_id='e20_concept_grounding', version=1, evaluation_stage='development', final_test_enabled=False,
    concept_ids=list(CONCEPTS), wrong_concept_ids=[*CONCEPTS[1:], CONCEPTS[0]],
    n_videos=10, frame_positions=[2, 7, 13], review_videos=6, seed=20261124,
    bootstrap=dict(repetitions=1000, min_sources=5, min_valid=800),
    continuation=dict(min_present=5, min_absent=5, min_label_videos=3,
                      min_determinable=.7, min_auroc=.7, min_auroc_delta=.05, min_usable_concepts=3),
)
_CONFIG = json.dumps(DEFAULT_CONFIG, sort_keys=True, allow_nan=False)


def validate_config(config):
    if json.dumps(config, sort_keys=True, allow_nan=False) != _CONFIG:
        raise ValueError('E2.0 requires the fixed development audit protocol')


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_config(config)
    return config


def mapping_from_source(source):
    frames = source['frames']
    result = []
    for row in frames:
        vid = row['video_id']
        pos, frame = int(row['frame_position_zero_based']), int(row['frame_index'])
        result.append(dict(
            opaque_id='f_'+digest([source['protocol_id'], vid, frame])[:20],
            anonymous_video_id='v_'+digest([source['protocol_id'], vid])[:20],
            video_id=vid, source_group_id=source['groups'][vid], frame_position_zero_based=str(pos),
            frame_index=str(frame), video_sha256=row['video_sha256'],
            original_image=row['original_image'], clip_image=row['clip_image'],
            original_sha256=source['audit_hashes'][row['original_image']],
            clip_sha256=source['audit_hashes'][row['clip_image']],
        ))
    result.sort(key=lambda row: row['opaque_id'])
    if (len(result) != 30 or len({r['opaque_id'] for r in result}) != 30
            or len({r['video_id'] for r in result}) != 10
            or any({int(r['frame_position_zero_based']) for r in result if r['video_id'] == v} != {2, 7, 13}
                   for v in {r['video_id'] for r in result})):
        raise ValueError('E2.0 requires10videos and30unique frozen frames')
    return result


def review_frame_ids(mapping):
    videos = sorted({r['anonymous_video_id'] for r in mapping})
    if len(videos) != 10:
        raise ValueError('E2.0 review sampling requires10videos')
    rng = np.random.Generator(np.random.PCG64(DEFAULT_CONFIG['seed']))
    selected = []
    for video in rng.choice(videos, size=6, replace=False):
        frames = sorted(r['opaque_id'] for r in mapping if r['anonymous_video_id'] == video)
        selected.append(str(rng.choice(frames)))
    return sorted(selected)


def form_template(mapping, concepts, view, selected=None):
    kind = 'crop' if view in ('crop', 'review-crop') else 'original'
    definitions = {r['concept_id']: r for r in concepts}
    return [dict(opaque_id=frame['opaque_id'], concept_id=cid, view=kind,
                 image=f"images/{frame['opaque_id']}.png", concept_label=definitions[cid]['concept_label'],
                 description_en=definitions[cid]['description_en'], **{k: '' for k in HUMAN_FIELDS})
            for frame in mapping if selected is None or frame['opaque_id'] in selected for cid in CONCEPTS]


def validate_judgments(rows, template, *, complete=False):
    if len(rows) != len(template):
        raise ValueError('incomplete/duplicate E2.0 judgment coverage')
    for row, original in zip(rows, template):
        if set(row) != set(original) or any(row[k] != v for k, v in original.items() if k not in HUMAN_FIELDS):
            raise ValueError('E2.0 judgment identity/template changed')
        fields = {key: str(row[key]).strip() for key in HUMAN_FIELDS}
        label = fields['judgment']
        if not label:
            if complete:
                raise ValueError('incomplete E2.0 judgments; fill every row before freezing')
            if any(fields.values()):
                raise ValueError('partial human metadata without a judgment')
            continue
        if label not in ('present', 'absent', 'uncertain') or fields['reason_code'] not in REASONS:
            raise ValueError('invalid E2.0 judgment or reason code')
        if not all(fields[k] for k in HUMAN_FIELDS):
            raise ValueError('E2.0 human judgment needs all reviewer/time/evidence fields')
        try:
            reviewed = datetime.fromisoformat(fields['reviewed_at'].replace('Z', '+00:00'))
            elapsed = float(fields['elapsed_seconds'])
        except (ValueError, TypeError) as exc:
            raise ValueError('invalid review time/elapsed_seconds') from exc
        if reviewed.tzinfo is None or reviewed.utcoffset() is None or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('review requires timezone and finite nonnegative elapsed_seconds')
        if label == 'uncertain' and fields['reason_code'] == 'clear':
            raise ValueError('uncertain needs a specific missing/ambiguous evidence reason')
        if label != 'uncertain' and fields['reason_code'] != 'clear':
            raise ValueError('present/absent requires clear visible evidence')
    return rows
