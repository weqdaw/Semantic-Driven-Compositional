"""Immutable E1 audit inputs and staged, human-only E2.0 review packets."""
from pathlib import Path
import html
import shutil

import numpy as np

from .config import ROOT
from .e1_experiment import OFFICIAL
from .e11_workflow import tree_hashes, copy_evidence, code_hashes
from .e12_audit import analyze_audit
from .e20_protocol import (
    DEFAULT_CONFIG, CONCEPTS, VIEWS, HUMAN_FIELDS, validate_config, mapping_from_source,
    review_frame_ids, form_template, validate_judgments,
)
from .provenance import digest, environment, new_directory, now, read_json, sha256_file, write_csv, write_json
from .reporting import _read_csv
from .round2_io import safe_output


def source_context(e11_batch, audit_directory):
    e11_batch, audit_directory = Path(e11_batch).resolve(), Path(audit_directory).resolve()
    batch_before, audit_before = tree_hashes(e11_batch), tree_hashes(audit_directory)
    checked = analyze_audit(audit_directory, e11_batch)
    batch = read_json(e11_batch/'batch.json')
    if batch['evaluation_stage'] != 'development':
        raise ValueError('E2.0 development only; final test sealed')
    bundle = read_json(e11_batch/'parent/reference/manifest_snapshot.json')
    frames = _read_csv(audit_directory/'frames.csv')
    by_id = {r['video_id']: r for r in bundle['rows']}
    if any(by_id[r['video_id']]['experimental_role'] != 'development' for r in frames):
        raise ValueError('E2.0 audit includes non-development/final-test images')
    if tree_hashes(e11_batch) != batch_before or tree_hashes(audit_directory) != audit_before:
        raise ValueError('E2.0 source changed during validation')
    return dict(protocol_id=checked['protocol_id'], feature_condition=batch['feature_condition'],
        e11_batch=e11_batch, audit_directory=audit_directory, batch_hashes=batch_before, audit_hashes=audit_before,
        frames=frames, groups={v: by_id[v]['source_group_id'] for v in {r['video_id'] for r in frames}},
        template=_read_csv(audit_directory/'judgments.template.csv'), scores=_read_csv(audit_directory/'model_scores.after_review.csv'))


def _definitions(source):
    result = []
    for cid in CONCEPTS:
        part = [r for r in source['template'] if r['concept_id'] == cid]
        definitions = {(r['concept_label'], r['description_en']) for r in part}
        if len(definitions) != 1:
            raise ValueError('mixed E2.0 concept definitions')
        label, text = definitions.pop()
        result.append(dict(concept_id=cid, concept_label=label, description_en=text))
    return result


def _scores(source, mapping):
    original = {(r['video_id'], int(r['frame_position_zero_based']), r['concept_id']): r['cosine_score'] for r in source['scores']}
    return [dict(opaque_id=r['opaque_id'], concept_id=c, cosine_score=str(original[r['video_id'], int(r['frame_position_zero_based']), c]))
            for r in mapping for c in CONCEPTS]


def check_inputs(e11_batch, audit_directory, config=DEFAULT_CONFIG, *, require_official=False):
    validate_config(config)
    source = source_context(e11_batch, audit_directory)
    if require_official and source['feature_condition'] != OFFICIAL:
        raise ValueError('production E2.0 requires official validated CLIP concept features')
    mapping = mapping_from_source(source)
    return dict(status='verified', evaluation_stage='development', final_test_enabled=False,
        protocol_id=source['protocol_id'], feature_condition=source['feature_condition'],
        n_videos=10, n_source_groups=len(set(source['groups'].values())), n_frames=30, n_existing_images=60,
        n_frame_concept_pairs=180, n_primary_judgments=360, n_secondary_judgments=72,
        selected_review_frames=review_frame_ids(mapping), image_encoding_executed=False, text_encoding_executed=False,
        video_decoding_executed=False, gradient_fits_executed=0)


def _adjudication_template(forms):
    primary = {(r['opaque_id'], r['concept_id'], r['view']): r for view in ('crop', 'original') for r in forms[view]}
    result = []
    for view in ('review-crop', 'review-original'):
        for second in forms[view]:
            first = primary[second['opaque_id'], second['concept_id'], second['view']]
            if first['judgment'].strip() == second['judgment'].strip():
                continue
            result.append(dict(
                **{k: first[k] for k in first if k not in HUMAN_FIELDS and k != 'image'},
                image=f"../{first['view']}/images/{first['opaque_id']}.png",
                first_judgment=first['judgment'].strip(), second_judgment=second['judgment'].strip(),
                first_reason=first['reason_code'].strip(), second_reason=second['reason_code'].strip(),
                **{key: '' for key in HUMAN_FIELDS},
            ))
    return result


def expected_template(view, mapping, definitions, forms):
    if view == 'adjudication':
        return _adjudication_template(forms)
    selected = review_frame_ids(mapping) if view.startswith('review-') else None
    return form_template(mapping, definitions, view, selected)


def _html(view, template):
    lines = ['<!doctype html><html lang="zh"><meta charset="utf-8">',
        '<title>E2.0 概念审核</title><style>body{font-family:system-ui;margin:24px}img{max-width:640px;max-height:400px}td{padding:10px}</style>',
        f'<h1>E2.0 {html.escape(view)} 审核</h1>',
        '<p>在 judgments.csv 填写 present / absent / uncertain 和全部人工字段。看不清、部位缺失不等于 absent。仅依据当前图像判断，不推断动作类别。</p>',
        '<p>只向审核者提供当前文件夹；private 为分析者身份与分数材料，不应提前查看。不能以此流程证明完全双盲。</p>', '<table>']
    seen = set()
    for row in template:
        if row['opaque_id'] not in seen:
            lines.append(f'<tr><td>{html.escape(row["opaque_id"])}</td><td><img src="{html.escape(row["image"], quote=True)}"></td></tr>')
            seen.add(row['opaque_id'])
    lines += ['</table><h2>判断概念</h2><ul>']
    definitions = {(r['concept_id'], r['concept_label'], r['description_en']) for r in template}
    for cid, label, description in sorted(definitions):
        lines.append('<li>'+html.escape(f'{cid}: {label} | {description}')+'</li>')
    return '\n'.join(lines+['</ul></html>'])


def _open_form(packet, manifest, view, template, mapping):
    folder = packet/view
    folder.mkdir(exist_ok=False)
    fields = list(template[0]) if template else ['opaque_id', 'concept_id', 'view', 'image', 'concept_label', 'description_en',
                                               'first_judgment', 'second_judgment', 'first_reason', 'second_reason', *HUMAN_FIELDS]
    write_csv(folder/'judgments.csv', template, fields)
    (folder/'index.html').write_text(_html(view, template), encoding='utf-8')
    manifest['immutable'][f'{view}/index.html'] = sha256_file(folder/'index.html')
    if view != 'adjudication':
        by_id = {r['opaque_id']: r for r in mapping}
        kind = 'clip_image' if view in ('crop', 'review-crop') else 'original_image'
        for frame in sorted({r['opaque_id'] for r in template}):
            target = folder/'images'/f'{frame}.png'
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(packet/'private/source_audit'/by_id[frame][kind], target)
            manifest['immutable'][target.relative_to(packet).as_posix()] = sha256_file(target)
    manifest['forms'][view] = f'{view}/judgments.csv'


def prepare_e20(e11_batch, audit_directory, config, paths, output, *, require_official=False):
    validate_config(config)
    source = source_context(e11_batch, audit_directory)
    if require_official and source['feature_condition'] != OFFICIAL:
        raise ValueError('production E2.0 requires official validated CLIP concept features')
    mapping, definitions = mapping_from_source(source), _definitions(source)
    root = safe_output(output, paths, protected=(source['e11_batch'], source['audit_directory']))
    packet = new_directory(root, 'e20-audit')
    (packet/'private').mkdir()
    manifest = dict(status='running', errors=[], schema='e20_concept_grounding_v1', config=config,
        evaluation_stage='development', final_test_enabled=False, protocol_id=source['protocol_id'],
        feature_condition=source['feature_condition'], e11_batch=str(source['e11_batch']), audit_directory=str(source['audit_directory']),
        batch_hashes=source['batch_hashes'], audit_hashes=source['audit_hashes'], source_code=code_hashes(),
        prepared_at=now(), immutable={}, forms={}, locks={})
    write_json(packet/'input_manifest.json', manifest)
    try:
        copy_evidence(source['audit_directory'], packet/'private/source_audit', source['audit_hashes'])
        copy_evidence(ROOT, packet/'private/code_snapshot', manifest['source_code'])
        write_json(packet/'private/environment.json', environment())
        write_json(packet/'private/concepts.json', definitions)
        write_csv(packet/'private/opaque_mapping.private.csv', mapping, list(mapping[0]))
        score_rows = _scores(source, mapping)
        write_csv(packet/'private/model_scores.csv', score_rows, list(score_rows[0]))
        (packet/'preregistration.md').write_text(
            '# E2.0 固定审核协议\n\n180帧—概念对；主审核双视图360判断，第二人72判断。'
            '先冻结crop，再看original；第二人重复此顺序。真实标签不由程序或模型填写。'
            '概念分数仅对应crop，original只作可见性／裁剪保留诊断。'
            '来源聚类bootstrap1000次；6概念正确／固定错配AUROC比较。'
            '可判定率>=.70、5正5负且各3视频、AUC>=.70、差>=.05；至少3概念且完成独立审核／裁决才达到探索门槛。'
            '无阈值搜索、训练、编码、解码或最终测试。\n\n精确配置见input_manifest.json。\n', encoding='utf-8')
        manifest['immutable'] = tree_hashes(packet/'private')
        manifest['immutable'] = {'private/'+k: v for k, v in manifest['immutable'].items()}
        manifest['immutable']['preregistration.md'] = sha256_file(packet/'preregistration.md')
        _open_form(packet, manifest, 'crop', form_template(mapping, definitions, 'crop'), mapping)
        if (tree_hashes(source['audit_directory']) != source['audit_hashes']
                or tree_hashes(source['e11_batch']) != source['batch_hashes'] or code_hashes() != manifest['source_code']):
            raise ValueError('E2.0 inputs/code changed during preparation')
        manifest['status'] = 'success'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(packet/'input_manifest.json', manifest)
    return packet


def load_packet(packet, *, require_official=False):
    packet = Path(packet).resolve()
    manifest = read_json(packet/'input_manifest.json')
    validate_config(manifest['config'])
    if (manifest['status'] != 'success' or manifest['errors'] or manifest['schema'] != 'e20_concept_grounding_v1'
            or manifest['evaluation_stage'] != 'development' or manifest['final_test_enabled'] is not False):
        raise ValueError('incomplete/non-development E2.0 packet; final test sealed')
    source = source_context(manifest['e11_batch'], manifest['audit_directory'])
    if require_official and source['feature_condition'] != OFFICIAL:
        raise ValueError('production E2.0 requires official validated CLIP concept features')
    if (source['audit_hashes'] != manifest['audit_hashes'] or source['batch_hashes'] != manifest['batch_hashes']
            or source['protocol_id'] != manifest['protocol_id'] or source['feature_condition'] != manifest['feature_condition']
            or code_hashes() != manifest['source_code']):
        raise ValueError('E2.0 original source/code identity changed; prepare a new packet')
    for name, expected in manifest['immutable'].items():
        path = (packet/name).resolve()
        if not path.is_relative_to(packet) or sha256_file(path) != expected:
            raise ValueError('unsafe/corrupt E2.0 immutable evidence')
    if tree_hashes(packet/'private/source_audit') != source['audit_hashes']:
        raise ValueError('E2.0 source snapshot reconstruction mismatch')
    if tree_hashes(packet/'private/code_snapshot') != manifest['source_code']:
        raise ValueError('E2.0 source code snapshot mismatch')
    mapping, definitions = mapping_from_source(source), _definitions(source)
    if _read_csv(packet/'private/opaque_mapping.private.csv') != mapping:
        raise ValueError('E2.0 opaque mapping differs from source reconstruction')
    if read_json(packet/'private/concepts.json') != definitions:
        raise ValueError('E2.0 concept definitions differ from original source')
    scores = _scores(source, mapping)
    if _read_csv(packet/'private/model_scores.csv') != scores:
        raise ValueError('E2.0 score reconstruction mismatch')
    nlocks = len(manifest['locks'])
    if set(manifest['locks']) != set(VIEWS[:nlocks]):
        raise ValueError('invalid E2.0 freeze order')
    available = VIEWS[:min(nlocks+1, len(VIEWS))]
    if set(manifest['forms']) != set(available):
        raise ValueError('invalid E2.0 form opening order')
    forms, form_hashes = {}, {}
    for view in available:
        if manifest['forms'][view] != f'{view}/judgments.csv':
            raise ValueError('unsafe E2.0 form path')
        template = expected_template(view, mapping, definitions, forms)
        form_hashes[view] = sha256_file(packet/manifest['forms'][view])
        forms[view] = validate_judgments(_read_csv(packet/manifest['forms'][view]), template, complete=view in manifest['locks'])
        if sha256_file(packet/manifest['forms'][view]) != form_hashes[view]:
            raise ValueError('E2.0 judgment CSV changed during validation')
        if (packet/view/'index.html').read_text(encoding='utf-8') != _html(view, template):
            raise ValueError('E2.0 reviewer HTML differs from anonymous reconstruction')
        if view != 'adjudication':
            kind = 'clip_sha256' if view in ('crop', 'review-crop') else 'original_sha256'
            for row in mapping:
                if row['opaque_id'] in {r['opaque_id'] for r in template}:
                    path = packet/view/'images'/f"{row['opaque_id']}.png"
                    if sha256_file(path) != row[kind]:
                        raise ValueError('E2.0 image differs from original pixels')
        if view in manifest['locks']:
            lock = manifest['locks'][view]
            frozen = packet/'private/frozen'/f'{view}.csv'
            if (lock['view'] != view or lock['form_sha256'] != sha256_file(packet/manifest['forms'][view])
                    or sha256_file(frozen) != lock['form_sha256'] or not lock.get('frozen_at')):
                raise ValueError('E2.0 frozen judgment/lock changed')
    if 'review-crop' in forms:
        primary_names = {r['reviewer'].strip().casefold() for v in ('crop', 'original') for r in forms[v]}
        for view in ('review-crop', 'review-original'):
            if view in forms and any(r['judgment'].strip() and r['reviewer'].strip().casefold() in primary_names for r in forms[view]):
                raise ValueError('independent reviewer must differ from primary reviewer')
    allowed = set(manifest['immutable']) | set(manifest['forms'].values()) | {'input_manifest.json'}
    if set(tree_hashes(packet)) != allowed:
        raise ValueError('unregistered/missing E2.0 evidence file')
    return dict(packet=packet, manifest=manifest, mapping=mapping, definitions=definitions, scores=scores,
                forms=forms, form_hashes=form_hashes)


def freeze_view(packet, view):
    if view not in VIEWS:
        raise ValueError('unknown E2.0 view')
    context = load_packet(packet)
    packet, manifest = context['packet'], context['manifest']
    n = len(manifest['locks'])
    if n >= len(VIEWS) or VIEWS[n] != view:
        raise ValueError('E2.0 freeze order: complete the available view first; frozen views cannot be overwritten')
    template = expected_template(view, context['mapping'], context['definitions'], context['forms'])
    validate_judgments(context['forms'][view], template, complete=True)
    expected_form_hash = context['form_hashes'][view]
    if sha256_file(packet/manifest['forms'][view]) != expected_form_hash:
        raise ValueError('E2.0 judgment CSV changed concurrently before freeze')
    before = digest(manifest)
    frozen = packet/'private/frozen'/f'{view}.csv'
    frozen.parent.mkdir(exist_ok=True)
    if frozen.exists():
        raise ValueError('refusing to overwrite frozen judgments')
    shutil.copyfile(packet/manifest['forms'][view], frozen)
    if sha256_file(frozen) != expected_form_hash:
        manifest['status'] = 'failed'
        manifest['errors'].append('judgment changed during freeze snapshot')
        write_json(packet/'input_manifest.json', manifest)
        raise ValueError('E2.0 judgment changed during freeze snapshot')
    manifest['immutable'][f'private/frozen/{view}.csv'] = sha256_file(frozen)
    manifest['locks'][view] = dict(view=view, form_sha256=sha256_file(frozen), frozen_at=now())
    try:
        if n+1 < len(VIEWS):
            next_view = VIEWS[n+1]
            template = expected_template(next_view, context['mapping'], context['definitions'], context['forms'])
            _open_form(packet, manifest, next_view, template, context['mapping'])
        if (digest(read_json(packet/'input_manifest.json')) != before
                or sha256_file(packet/manifest['forms'][view]) != expected_form_hash):
            raise ValueError('E2.0 packet changed concurrently during freezing')
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['errors'].append(str(exc))
        write_json(packet/'input_manifest.json', manifest)
        raise
    write_json(packet/'input_manifest.json', manifest)
    return packet
