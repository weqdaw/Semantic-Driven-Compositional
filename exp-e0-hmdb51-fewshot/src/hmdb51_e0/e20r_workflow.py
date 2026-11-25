"""Source-isolated repair cohorts, paired images, frozen encoding and staged reviews."""
from pathlib import Path
import shutil
import html
import numpy as np
from PIL import Image

from .config import ROOT, safe_video_path
from .e1_experiment import OFFICIAL
from .e11_workflow import tree_hashes, code_hashes, copy_evidence
from .e11_audit import native_crop
from .e20_workflow import source_context as original_context
from .e20r_protocol import (DEFAULT_CONFIG, CONCEPTS, VIEWS, HUMAN_FIELDS, validate_config,
    select_representative, select_supplement, make_template, validate_form)
from .e20r_media import letterbox, PairedCLIP
from .video_io import decoder_identity, decode_video
from .provenance import read_json, write_json, write_csv, sha256_file, digest, new_directory, now, environment
from .reporting import _read_csv
from .round2_io import safe_output


def source_context(e11_batch, audit_directory):
    source = original_context(e11_batch, audit_directory)
    parent = source['e11_batch']/'parent'
    bundle = read_json(parent/'reference/manifest_snapshot.json')
    index = read_json(parent/'feature_index_snapshot.json')
    catalog = read_json(parent/'concept_catalog.json')['concepts']
    positions = [next(i for i, c in enumerate(catalog) if c['id'] == cid) for cid in CONCEPTS]
    with np.load(parent/'feature_bank.npz', allow_pickle=False) as bank:
        text = bank['E'][positions].copy()
    definitions = [dict(concept_id=catalog[i]['id'], concept_label=catalog[i]['label_zh'],
                        description_en=catalog[i]['description_en']) for i in positions]
    return dict(e11_batch=source['e11_batch'], audit_directory=source['audit_directory'],
        batch_hashes=source['batch_hashes'], audit_hashes=source['audit_hashes'],
        protocol_id=source['protocol_id'], feature_condition=source['feature_condition'],
        rows=bundle['rows'], entries={r['video_id']: r for r in index['entries']},
        excluded_ids=sorted(source['groups']),
        excluded_groups=sorted({r['source_group_id'] for r in bundle['rows']
                                if r['experimental_role'] in ('development', 'final_test') and r.get('source_group_id')}),
        encoder=index['identity']['encoder'], decoder=bundle['parent_manifest']['decoder'],
        concepts=definitions, text_embeddings=text)


def _official(source, require):
    if require and source['feature_condition'] != OFFICIAL:
        raise ValueError('production E2.0-R requires official validated CLIP features')


def _signature(source):
    keys = ('protocol_id', 'feature_condition', 'batch_hashes', 'audit_hashes', 'encoder', 'decoder', 'concepts')
    return digest(dict(**{k: source[k] for k in keys}, text=source['text_embeddings'].tolist()))


def _selected(source, selection=None, base=None):
    excluded_groups = set(source['excluded_groups'])
    excluded_ids = set(source['excluded_ids'])
    if base is None:
        rows = select_representative(source['rows'], excluded_groups, excluded_ids)
    else:
        excluded_groups.update(r['source_group_id'] for r in base['selected_videos'])
        excluded_ids.update(r['video_id'] for r in base['selected_videos'])
        rows = select_supplement(source['rows'], selection, excluded_groups, excluded_ids)
    for row in rows:
        entry = source['entries'][row['video_id']]
        if len(entry['frame_indices']) != 16:
            raise ValueError('repair selection requires original16-frame source indices')
        row['frame_indices'] = [entry['frame_indices'][i] for i in DEFAULT_CONFIG['frame_positions']]
    return rows


def check_inputs(e11_batch, audit_directory, config=DEFAULT_CONFIG, *, require_official=False):
    validate_config(config)
    source = source_context(e11_batch, audit_directory)
    _official(source, require_official)
    rows = _selected(source)
    return dict(status='verified', feature_condition=source['feature_condition'], evaluation_stage='development',
        final_test_enabled=False, n_videos=len(rows), n_source_groups=len({r['source_group_id'] for r in rows}),
        n_frames=60, primary_judgments=1080, secondary_judgments=108,
        source_isolation_scope='audited support_pool versus development; final-test source groups remain unaudited',
        video_decoding_executed=False, image_encoding_executed=False, gradient_fits_executed=0)


def plan_e20r(e11_batch, audit_directory, config, paths, output, *, selection_path=None, base_plan=None, require_official=False):
    validate_config(config)
    source = source_context(e11_batch, audit_directory)
    _official(source, require_official)
    if bool(selection_path) != bool(base_plan):
        raise ValueError('supplement requires both selection CSV and representative base plan')
    base = load_plan(base_plan, require_official=require_official)[0] if base_plan else None
    if base and (base['cohort'] != 'representative' or base['source_signature'] != _signature(source)):
        raise ValueError('supplement requires same-source representative base plan')
    selection = _read_csv(selection_path) if selection_path else None
    rows = _selected(source, selection, base)
    root = safe_output(output, paths, protected=(source['e11_batch'], source['audit_directory']))
    folder = new_directory(root, 'e20r-plan')
    plan = dict(schema='e20r_plan_v1', config=config, created_at=now(),
        cohort='coverage_supplement' if base else 'representative',
        e11_batch=str(source['e11_batch']), audit_directory=str(source['audit_directory']),
        dataset_root=str(Path(paths['dataset_root']).resolve()), source_signature=_signature(source),
        source_code=code_hashes(), selected_videos=rows, selection=selection,
        base_plan=str(Path(base_plan).resolve()) if base_plan else None,
        base_plan_sha256=sha256_file(base_plan) if base_plan else None)
    write_json(folder/'plan.json', plan)
    return folder/'plan.json'


def load_plan(path, *, require_official=False):
    plan = read_json(path)
    validate_config(plan['config'])
    if plan['schema'] != 'e20r_plan_v1' or plan['source_code'] != code_hashes():
        raise ValueError('repair plan source code changed; make a new plan')
    source = source_context(plan['e11_batch'], plan['audit_directory'])
    _official(source, require_official)
    if plan['source_signature'] != _signature(source):
        raise ValueError('repair source identity changed')
    base = None
    if plan['cohort'] == 'coverage_supplement':
        if sha256_file(plan['base_plan']) != plan['base_plan_sha256']:
            raise ValueError('representative base plan changed')
        base = read_json(plan['base_plan'])
        if base['cohort'] != 'representative':
            raise ValueError('supplement base must be representative')
        base, _ = load_plan(plan['base_plan'], require_official=require_official)
    elif plan['cohort'] != 'representative' or plan['selection'] is not None or plan['base_plan'] is not None:
        raise ValueError('invalid repair cohort')
    if plan['selected_videos'] != _selected(source, plan['selection'], base):
        raise ValueError('repair selection reconstruction mismatch')
    return plan, source


def _mapping(plan):
    result = []
    for i, row in enumerate(plan['selected_videos']):
        for position, index in zip(DEFAULT_CONFIG['frame_positions'], row['frame_indices']):
            result.append(dict(opaque_id=f'F{i*3+len(result)%3+1:04d}', anonymous_video_id=f'V{i+1:03d}',
                video_id=row['video_id'], source_group_id=row['source_group_id'],
                frame_position_zero_based=position, decoded_frame_index=index))
    # Anonymize ordering without consulting labels or model scores.
    rng = np.random.Generator(np.random.PCG64(DEFAULT_CONFIG['seed']))
    order = rng.permutation(len(result))
    for i, j in enumerate(order):
        result[j]['opaque_id'] = f'F{i+1:04d}'
    return sorted(result, key=lambda r: r['opaque_id'])


def _html(view, template):
    lines = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>E2.0-R</title>',
        '<style>body{font-family:system-ui;margin:24px}img{max-width:720px;max-height:480px}</style>',
        f'<h1>E2.0-R {html.escape(view)}</h1>',
        '<p>仅依据当前图像判断：present / absent / uncertain。部位不可见不等于absent；主角不明确记uncertain。',
        '填写judgments.csv的全部审核字段，reviewer_origin必须为human或ai。原生裁剪→加边→原图按顺序冻结。',
        '只向审核者提供当前视图文件夹；private含来源及模型分数，不应提前查看。名称不同不能证明实际人员独立。</p>']
    for row in {(r['opaque_id'], r['view']): r for r in template}.values():
        lines.append(f'<p>{html.escape(row["opaque_id"]+" / "+row["view"])}</p><img src="{html.escape(row["image"], quote=True)}">')
    for row in {r['concept_id']: r for r in template}.values():
        lines.append('<p>'+html.escape(f"{row['concept_id']} {row['description_en']} | {row['operational_rule']}")+'</p>')
    return '\n'.join(lines+['</html>'])


def expected_template(view, mapping, concepts, forms):
    if view != 'adjudication':
        return make_template(mapping, concepts, view)
    primary = {(r['opaque_id'], r['concept_id'], r['view']): r for v in VIEWS[:3] for r in forms[v]}
    result = []
    for v in VIEWS[3:6]:
        for second in forms[v]:
            first = primary[second['opaque_id'], second['concept_id'], second['view']]
            if first['judgment'].strip() == second['judgment'].strip():
                continue
            result.append(dict(**{k: value for k, value in first.items() if k not in HUMAN_FIELDS and k != 'image'},
                image=f"../{first['view']}/images/{first['opaque_id']}.png",
                first_judgment=first['judgment'].strip(), second_judgment=second['judgment'].strip(),
                first_reason=first['reason_code'], second_reason=second['reason_code'],
                first_origin=first['reviewer_origin'], second_origin=second['reviewer_origin'],
                **{k: '' for k in HUMAN_FIELDS}))
    return result


def _register(packet, manifest):
    found = {k: v for k, v in tree_hashes(packet).items()
        if k not in set(manifest['forms'].values()) | {'input_manifest.json'}}
    if any(found.get(k) != v for k, v in manifest['immutable'].items()):
        raise ValueError('immutable repair evidence changed during operation')
    manifest['immutable'] = found


def _open_form(packet, manifest, view, template):
    folder = packet/view
    folder.mkdir()
    fields = list(template[0]) if template else ['opaque_id', 'concept_id', 'view', 'image', 'concept_label',
        'description_en', 'operational_rule', 'first_judgment', 'second_judgment', 'first_reason', 'second_reason',
        'first_origin', 'second_origin', *HUMAN_FIELDS]
    write_csv(folder/'judgments.csv', template, fields)
    (folder/'index.html').write_text(_html(view, template), encoding='utf-8')
    if view != 'adjudication':
        for frame in sorted({r['opaque_id'] for r in template}):
            target = folder/'images'/f'{frame}.png'
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(packet/'private/images'/view.removeprefix('review-')/f'{frame}.png', target)
    manifest['forms'][view] = f'{view}/judgments.csv'


def prepare_e20r(plan_path, paths, output, *, require_official=False):
    plan, source = load_plan(plan_path, require_official=require_official)
    if str(Path(paths['dataset_root']).resolve()) != plan['dataset_root'] or decoder_identity() != source['decoder']:
        raise ValueError('dataset root or decoder differs from original source')
    root = safe_output(output, paths, protected=(source['e11_batch'], source['audit_directory'], Path(plan_path).parent))
    packet = new_directory(root, 'e20r-audit')
    (packet/'private').mkdir()
    manifest = dict(schema='e20r_packet_v1', status='running', errors=[], plan_path=str(Path(plan_path).resolve()),
        plan_sha256=sha256_file(plan_path), feature_condition=source['feature_condition'], cohort=plan['cohort'],
        config=plan['config'], source_code=code_hashes(), forms={}, locks={}, immutable={}, encoding=None, created_at=now())
    write_json(packet/'input_manifest.json', manifest)
    try:
        shutil.copyfile(plan_path, packet/'private/plan.json')
        copy_evidence(ROOT, packet/'private/code_snapshot', manifest['source_code'])
        write_json(packet/'private/environment.json', environment())
        write_json(packet/'private/concepts.json', source['concepts'])
        mapping = _mapping(plan)
        write_json(packet/'private/mapping.json', mapping)
        np.savez_compressed(packet/'private/text_embeddings.npz', E=source['text_embeddings'])
        for row in plan['selected_videos']:
            path = safe_video_path(paths['dataset_root'], row['relative_path'])
            if sha256_file(path) != row['sha256']:
                raise ValueError('raw video hash differs from source manifest')
            metadata, frames = decode_video(path, row['frame_indices'])
            if metadata['frame_count'] != row['frame_count'] or len(frames) != 3 or sha256_file(path) != row['sha256']:
                raise ValueError('raw video changed or decode/frame count mismatch')
            for index, rgb in zip(row['frame_indices'], frames):
                opaque = next(r['opaque_id'] for r in mapping if r['video_id'] == row['video_id'] and r['decoded_frame_index'] == index)
                original = Image.fromarray(rgb).convert('RGB')
                for view, image in [('original', original), ('native', native_crop(rgb, source['encoder'])), ('letterbox', letterbox(original))]:
                    target = packet/'private/images'/view/f'{opaque}.png'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.save(target)
        _open_form(packet, manifest, 'native', make_template(mapping, source['concepts'], 'native'))
        load_plan(plan_path, require_official=require_official)
        _register(packet, manifest)
        manifest['status'] = 'success'
    except Exception as exc:
        manifest['status'] = 'failed'; manifest['errors'].append(str(exc)); raise
    finally:
        write_json(packet/'input_manifest.json', manifest)
    return packet


def _verify_encoder(identity, expected):
    keys = ('implementation', 'architecture', 'dimension', 'checkpoint_sha256', 'parameter_sha256_before',
            'clip_source_sha256', 'preprocess', 'color', 'dtype', 'torch', 'torchvision', 'pillow')
    if expected.get('synthetic_test_fixture'):
        if identity != expected:
            raise ValueError('synthetic encoder mismatch')
    elif any(identity.get(k) != expected.get(k) for k in keys):
        raise ValueError('encoder/checkpoint/native preprocessing differs from source CLIP')


def _score_rows(mapping, q, text):
    values = q.astype(np.float64) @ text.astype(np.float64).T
    return [dict(opaque_id=r['opaque_id'], view=view, concept_id=cid, cosine_score=format(values[v, i, c], '.17g'))
        for v, view in enumerate(('native', 'letterbox')) for i, r in enumerate(mapping) for c, cid in enumerate(CONCEPTS)]


def _validate_encoding(packet, manifest, mapping, source):
    if manifest['encoding'] is None:
        return []
    record = read_json(packet/'private/encoding/encoder.json')
    if record != manifest['encoding']:
        raise ValueError('encoder identity mismatch')
    if record['input_image_sha256'] != _input_hashes(packet, mapping):
        raise ValueError('encoding input image provenance mismatch')
    _verify_encoder(record['identity'], source['encoder'])
    frozen = record['frozen']
    if (frozen.get('unchanged') is not True or frozen.get('before') != frozen.get('after')
            or (not source['encoder'].get('synthetic_test_fixture') and frozen['before'] != source['encoder']['parameter_sha256_before'])):
        raise ValueError('CLIP frozen parameter proof failed')
    with np.load(packet/'private/encoding/features.npz', allow_pickle=False) as bank:
        q, text = bank['Q'], bank['E']
    if (q.dtype != np.float32 or q.shape != (2, len(mapping), source['text_embeddings'].shape[1])
            or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=2e-6)
            or not np.array_equal(text, source['text_embeddings'])):
        raise ValueError('invalid paired feature/text provenance')
    scores = _score_rows(mapping, q, text)
    if _read_csv(packet/'private/encoding/model_scores.csv') != scores:
        raise ValueError('paired score forward reconstruction mismatch')
    return scores


def _input_hashes(packet, mapping):
    return {f'{view}/{r["opaque_id"]}': sha256_file(packet/'private/images'/view/f'{r["opaque_id"]}.png')
            for view in ('native', 'letterbox') for r in mapping}


def load_packet(packet, *, require_official=False):
    packet = Path(packet).resolve(); manifest = read_json(packet/'input_manifest.json')
    validate_config(manifest['config'])
    if manifest['schema'] != 'e20r_packet_v1' or manifest['status'] != 'success' or manifest['errors']:
        raise ValueError('incomplete/failed repair packet')
    if sha256_file(manifest['plan_path']) != manifest['plan_sha256'] or sha256_file(packet/'private/plan.json') != manifest['plan_sha256']:
        raise ValueError('repair plan snapshot changed')
    plan, source = load_plan(manifest['plan_path'], require_official=require_official)
    if (manifest['source_code'] != code_hashes() or tree_hashes(packet/'private/code_snapshot') != manifest['source_code']
            or manifest['config'] != plan['config'] or manifest['feature_condition'] != source['feature_condition'] or manifest['cohort'] != plan['cohort']):
        raise ValueError('repair packet source/config identity mismatch')
    for name, expected in manifest['immutable'].items():
        path = (packet/name).resolve()
        if not path.is_relative_to(packet) or sha256_file(path) != expected:
            raise ValueError('corrupt/unsafe repair immutable evidence')
    mapping = _mapping(plan)
    if read_json(packet/'private/mapping.json') != mapping or read_json(packet/'private/concepts.json') != source['concepts']:
        raise ValueError('mapping/concept reconstruction mismatch')
    with np.load(packet/'private/text_embeddings.npz', allow_pickle=False) as bank:
        if not np.array_equal(bank['E'], source['text_embeddings']):
            raise ValueError('source text embeddings changed')
    for row in mapping:
        with Image.open(packet/'private/images/original'/f"{row['opaque_id']}.png") as image:
            original = image.convert('RGB')
        for view, expected in [('native', native_crop(np.array(original), source['encoder'])), ('letterbox', letterbox(original))]:
            with Image.open(packet/'private/images'/view/f"{row['opaque_id']}.png") as image:
                if not np.array_equal(np.array(image.convert('RGB')), np.array(expected)):
                    raise ValueError('paired input pixel reconstruction mismatch')
    scores = _validate_encoding(packet, manifest, mapping, source)
    nlocks = len(manifest['locks'])
    available = VIEWS[:min(nlocks+1, len(VIEWS))]
    if set(manifest['locks']) != set(VIEWS[:nlocks]) or set(manifest['forms']) != set(available):
        raise ValueError('invalid repair review freeze/opening order')
    forms, hashes = {}, {}
    for view in available:
        if manifest['forms'][view] != f'{view}/judgments.csv':
            raise ValueError('unsafe form path')
        template = expected_template(view, mapping, source['concepts'], forms)
        path = packet/manifest['forms'][view]; hashes[view] = sha256_file(path)
        forms[view] = validate_form(_read_csv(path), template, complete=view in manifest['locks'])
        if sha256_file(path) != hashes[view] or (packet/view/'index.html').read_text(encoding='utf-8') != _html(view, template):
            raise ValueError('judgment changed concurrently or reviewer HTML changed')
        if view != 'adjudication':
            for opaque in {r['opaque_id'] for r in template}:
                if sha256_file(packet/view/'images'/f'{opaque}.png') != sha256_file(packet/'private/images'/view.removeprefix('review-')/f'{opaque}.png'):
                    raise ValueError('review image differs from paired input')
        if view in manifest['locks']:
            lock = manifest['locks'][view]
            if (lock['form_sha256'] != hashes[view] or not lock['frozen_at']
                    or sha256_file(packet/'private/frozen'/f'{view}.csv') != hashes[view]):
                raise ValueError('frozen repair judgments changed')
    primary = {r['reviewer'].strip().casefold() for v in VIEWS[:3] if v in forms for r in forms[v] if r['judgment'].strip()}
    identities = {}
    for view, rows in forms.items():
        for r in rows:
            if not r['judgment'].strip():
                continue
            name = r['reviewer'].strip().casefold()
            if view.startswith('review-') and name in primary:
                raise ValueError('independent reviewer must differ from primary reviewer')
            if name in identities and identities[name] != r['reviewer_origin']:
                raise ValueError('reviewer origin inconsistent across records')
            identities[name] = r['reviewer_origin']
    allowed = set(manifest['immutable']) | set(manifest['forms'].values()) | {'input_manifest.json'}
    if set(tree_hashes(packet)) != allowed:
        raise ValueError('unregistered/missing repair evidence file')
    return dict(packet=packet, manifest=manifest, plan=plan, source=source, mapping=mapping,
                concepts=source['concepts'], forms=forms, form_hashes=hashes, scores=scores)


def encode_e20r(packet, paths, *, device='cuda', encoder=None, require_official=False):
    context = load_packet(packet, require_official=require_official)
    packet, manifest, source = context['packet'], context['manifest'], context['source']
    if manifest['encoding'] is not None or (packet/'private/encoding').exists():
        raise ValueError('refusing to overwrite repair encoding')
    encoder = encoder or PairedCLIP(paths['weights_file'], device=device)
    _verify_encoder(encoder.identity, source['encoder'])
    images = []
    for view in ('native', 'letterbox'):
        for r in context['mapping']:
            with Image.open(packet/'private/images'/view/f"{r['opaque_id']}.png") as image:
                images.append(image.convert('RGB'))
    before = tree_hashes(packet)
    q = encoder.encode_images(images).reshape(2, len(context['mapping']), -1)
    record = dict(identity=encoder.identity, frozen=encoder.verify_frozen(), encoded_at=now(),
                  input_image_sha256=_input_hashes(packet, context['mapping']),
                  image_encoding_executed=True, text_encoding_executed=False, gradient_fits_executed=0)
    if tree_hashes(packet) != before:
        raise ValueError('repair packet changed during encoding')
    folder = packet/'private/encoding'; folder.mkdir()
    try:
        np.savez_compressed(folder/'features.npz', Q=q, E=source['text_embeddings'])
        write_json(folder/'encoder.json', record)
        scores = _score_rows(context['mapping'], q, source['text_embeddings'])
        write_csv(folder/'model_scores.csv', scores, list(scores[0]))
        manifest['encoding'] = record
        _validate_encoding(packet, manifest, context['mapping'], source)
        _register(packet, manifest)
    except Exception as exc:
        manifest['status'] = 'failed'; manifest['errors'].append(str(exc)); raise
    finally:
        write_json(packet/'input_manifest.json', manifest)
    return packet


def freeze_view(packet, view):
    context = load_packet(packet)
    packet, manifest = context['packet'], context['manifest']; n = len(manifest['locks'])
    if n >= len(VIEWS) or VIEWS[n] != view:
        raise ValueError('freeze the currently available view; frozen forms cannot be overwritten')
    template = expected_template(view, context['mapping'], context['concepts'], context['forms'])
    validate_form(context['forms'][view], template, complete=True)
    before = digest(manifest); expected = context['form_hashes'][view]
    frozen = packet/'private/frozen'/f'{view}.csv'; frozen.parent.mkdir(exist_ok=True)
    if frozen.exists():
        raise ValueError('refusing to overwrite frozen repair judgments')
    try:
        shutil.copyfile(packet/manifest['forms'][view], frozen)
        if sha256_file(frozen) != expected:
            raise ValueError('judgment changed during freezing')
        manifest['locks'][view] = dict(form_sha256=expected, frozen_at=now())
        if n+1 < len(VIEWS):
            next_view = VIEWS[n+1]
            _open_form(packet, manifest, next_view,
                expected_template(next_view, context['mapping'], context['concepts'], context['forms']))
        if digest(read_json(packet/'input_manifest.json')) != before or sha256_file(packet/manifest['forms'][view]) != expected:
            raise ValueError('repair packet changed concurrently during freeze')
        _register(packet, manifest)
    except Exception as exc:
        manifest['status'] = 'failed'; manifest['errors'].append(str(exc)); raise
    finally:
        write_json(packet/'input_manifest.json', manifest)
    return packet
