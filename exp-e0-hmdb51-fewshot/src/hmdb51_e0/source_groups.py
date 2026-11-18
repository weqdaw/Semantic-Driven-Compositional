"""Training-only source candidates and explicit, traceable human review."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
from pathlib import PurePosixPath, Path
import re

from .provenance import digest, new_directory, write_csv, write_json

REVIEW_FIELDS = ['video_id', 'class_name', 'video_sha256', 'original_role', 'candidate_source_id',
                 'source_group_id', 'verification_status', 'evidence', 'reviewer', 'reviewed_at']


def training_rows(parent: dict) -> list[dict]:
    return [r for r in parent['rows'] if r['official_tag'] == 1
            and r['experimental_role'] in ('support_pool', 'development')]


def candidate_source(video_id: str) -> str:
    stem = PurePosixPath(video_id).stem
    # The official training lists also contain *_jump_l_* and *_run_l_*.
    match = re.search(r'_(?:clap|jump|run|walk|wave)_[fuhl]_', stem, flags=re.IGNORECASE)
    return stem[:match.start()].lstrip('#').casefold() if match else ''


def candidate_records(parent: dict) -> list[dict]:
    records = []
    for row in sorted(training_rows(parent), key=lambda r: r['video_id']):
        candidate = candidate_source(row['video_id'])
        records.append(dict(video_id=row['video_id'], class_name=row['class_name'], video_sha256=row['sha256'],
                            original_role=row['experimental_role'], candidate_source_id=candidate,
                            source_group_id='src_' + digest(candidate)[:20] if candidate else '',
                            verification_status='unresolved', evidence='', reviewer='', reviewed_at=''))
    return records


def read_review(path: Path) -> list[dict]:
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(REVIEW_FIELDS).issubset(fields):
            raise ValueError('source review has missing/duplicate CSV columns')
        rows = list(reader)
        if any(None in row or any(row.get(key) is None for key in REVIEW_FIELDS) for row in rows):
            raise ValueError('malformed source review CSV row')
        return [{key: row[key] for key in REVIEW_FIELDS} for row in rows]


def validate_review(records: list[dict], parent: dict) -> list[dict]:
    expected = {r['video_id']: r for r in candidate_records(parent)}
    if len(records) != len(expected) or len({r.get('video_id') for r in records}) != len(records):
        raise ValueError('source review must cover each eligible training video exactly once')
    candidates = defaultdict(set)
    clean = []
    for record in records:
        if set(REVIEW_FIELDS) - set(record):
            raise ValueError('source review fields missing')
        row = {key: record[key] for key in REVIEW_FIELDS}
        if any(not isinstance(v, str) for v in row.values()):
            raise ValueError('source review fields must be strings')
        original = expected.get(row['video_id'])
        if original is None:
            raise ValueError('sealed or unknown video in source review')
        for key in ('video_id', 'class_name', 'video_sha256', 'original_role', 'candidate_source_id'):
            if row[key] != original[key]:
                raise ValueError(f'source review/video identity mismatch: {key}')
        if row['verification_status'] not in ('confirmed', 'conservative'):
            raise ValueError(f"unresolved source review: {row['video_id']}")
        if re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,95}', row['source_group_id']) is None:
            raise ValueError('source_group_id must be a canonical lowercase ID, up to 96 characters')
        if row['source_group_id'] in {'unknown', 'unresolved', 'tbd', 'todo', 'none', 'null', 'na', 'n_a', 'pending', 'unassigned'}:
            raise ValueError('unassigned placeholder is not a reviewed source_group_id')
        if not row['evidence'].strip() or not row['reviewer'].strip():
            raise ValueError('source review requires evidence and reviewer')
        try:
            date = datetime.fromisoformat(row['reviewed_at'])
            if date.tzinfo is None or date.utcoffset() is None:
                raise ValueError('timezone required')
        except ValueError as exc:
            raise ValueError('source reviewed_at must be an ISO timestamp with timezone') from exc
        if not row['candidate_source_id'] and row['verification_status'] != 'confirmed':
            raise ValueError('unknown filename source needs confirmed manual evidence; do not invent an independent singleton')
        if row['candidate_source_id']:
            candidates[row['candidate_source_id']].add(row['source_group_id'])
        clean.append(row)
    if any(len(groups) != 1 for groups in candidates.values()):
        raise ValueError('same candidate source cannot be split across source groups in this protocol')
    return sorted(clean, key=lambda r: r['video_id'])


def audit_sources(parent: dict, output: Path) -> Path:
    records = candidate_records(parent)
    pools = {r['candidate_source_id'] for r in records if r['original_role'] == 'support_pool' and r['candidate_source_id']}
    overlap = [r['video_id'] for r in records if r['original_role'] == 'development'
               and r['candidate_source_id'] and r['candidate_source_id'] in pools]
    groups = defaultdict(list)
    for row in records:
        groups[row['candidate_source_id'] or '<unresolved>'].append(row['video_id'])
    directory = new_directory(output, 'source-audit')
    write_csv(directory / 'source_groups.csv', records, REVIEW_FIELDS)
    write_json(directory / 'candidate_audit.json', dict(
        parent_protocol_id=parent['protocol_id'], n_training_videos=len(records),
        status='needs_human_review', status_counts=dict(Counter(r['verification_status'] for r in records)),
        candidate_groups=dict(sorted(groups.items())), candidate_overlap_development_ids=overlap,
        n_candidate_overlap_development=len(overlap),
        interpretation='Filename candidates only; neither verified leakage nor verified source independence.'))
    return directory / 'source_groups.csv'
