"""Immutable fixed-person annotations and three-view frozen-CLIP experiments."""
from pathlib import Path
import shutil,html
import numpy as np
from PIL import Image
from .config import ROOT
from .e1_experiment import OFFICIAL
from .e11_workflow import tree_hashes,code_hashes,copy_evidence
from .e11_audit import native_crop
from .e20r_workflow import _register,_verify_encoder
from .e20r_media import letterbox,PairedCLIP
from .e20r_protocol import HUMAN_FIELDS,review_frames
from .e20r2_source import parent_context,verify_parent
from .e20r2_protocol import (DEFAULT_CONFIG,VIEWS,CONCEPTS,validate_config,target_template,validate_targets,
    concept_template,validate_concepts,roi_image,guide_image,eligible,calibration_ids,target_key)
from .provenance import read_json,write_json,write_csv,sha256_file,new_directory,now,digest,environment
from .reporting import _read_csv
from .round2_io import safe_output

STAGES=('targets',*VIEWS,'review-native','review-letterbox','review-roi','adjudication')

def expected_template(view,source,targets,forms):
    if view=='adjudication':
        primary={(r['opaque_id'],r['concept_id'],r['view']):r for v in VIEWS for r in forms[v]}
        result=[]
        for v in ('review-native','review-letterbox','review-roi'):
            for second in forms[v]:
                first=primary[second['opaque_id'],second['concept_id'],second['view']]
                if first['judgment'].strip()==second['judgment'].strip():continue
                result.append(dict(**{k:v for k,v in first.items() if k not in HUMAN_FIELDS and k not in ('image','guide')},
                    image=f'../{first["view"]}/images/{first["opaque_id"]}.png' if first['image'] else '',
                    guide=f'../{first["view"]}/guides/{first["opaque_id"]}.png' if first['guide'] else '',
                    first_judgment=first['judgment'],second_judgment=second['judgment'],
                    first_origin=first['reviewer_origin'],second_origin=second['reviewer_origin'],**{k:'' for k in HUMAN_FIELDS}))
        return result
    result=concept_template(source['mapping'],source['concepts'],targets,view.removeprefix('review-'))
    if view.startswith('review-'):
        selected=set(review_frames(source['mapping']));result=[r for r in result if r['opaque_id'] in selected]
    return result

def _official(source,required):
    if required and (source['feature_condition']!=OFFICIAL or len(source['mapping'])!=60 or len(source['selected_videos'])!=20):
        raise ValueError('production E20R2 requires official fixed60-frame/20-video E20R main cohort')

def check_inputs(parent,config=DEFAULT_CONFIG,*,require_official=True):
    validate_config(config);source=parent_context(parent);_official(source,require_official)
    return dict(status='verified',n_frames=len(source['mapping']),n_videos=len(source['selected_videos']),
        feature_condition=source['feature_condition'],calibration_ids=calibration_ids(source['mapping']),
        image_encoding_executed=False,gradient_fits_executed=0,final_test_enabled=False)

def _sizes(packet,mapping):
    result={}
    for r in mapping:
        with Image.open(packet/'private/original'/f'{r["opaque_id"]}.png') as image:result[r['opaque_id']]=image.size
    return result

def _html(view,rows):
    lines=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>E20R2</title>',
        '<style>body{font-family:system-ui;margin:24px}img{max-width:640px;max-height:360px;margin:8px}</style>',
        f'<h1>E20R2 {html.escape(view)}</h1>',
        '<p>只判断固定target_id及其身份说明。绿色框仅为审核参考，不进入模型输入。',
        '目标不明、人体/必要部位不可见不能自动当作负例。记录human/ai来源和图像证据；private含模型分数。',
        '原生→加边→ROI顺序冻结；同人重复查看不等于独立审核。</p>']
    for row in {(r['opaque_id'],r.get('view',view)):r for r in rows}.values():
        lines.append('<p>'+html.escape(row['opaque_id']+' '+row['target_id']+' '+row.get('target_description',''))+'</p>')
        if row.get('image'):
            lines.append(f'<img src="{html.escape(row["image"],quote=True)}">')
            if row.get('guide'):lines.append(f'<img src="{html.escape(row["guide"],quote=True)}">')
        else:lines.append('<p>无可用目标ROI；该行保留，不能填写确定姿态。</p>')
    if view=='targets':
        lines.append('<p>框仅涵盖当前可见人体，原图像素整数坐标[x1,y1,x2,y2)，不补画外肢体。',
                     )
        lines.append('每帧原图按可见人体面积最大者（并列最左）预先固定目标，跨三输入不切换目标；不同帧不声称已完成身份追踪。12帧校准字段已固定。</p>')
    else:
        for row in {r['concept_id']:r for r in rows}.values():
            lines.append('<p>'+html.escape(row['concept_id']+' '+row['description_en']+' '+row['operational_rule'])+'</p>')
    return '\n'.join(lines+['</html>'])

def _open_form(packet,manifest,view,rows):
    folder=packet/view;folder.mkdir()
    filename='targets.csv' if view=='targets' else 'judgments.csv'
    fields=list(rows[0]) if rows else ['opaque_id','target_id','view','concept_id','image','guide','target_status','target_in_view',
        'target_description','concept_label','description_en','operational_rule','first_judgment','second_judgment','first_origin','second_origin',*HUMAN_FIELDS]
    write_csv(folder/filename,rows,fields);(folder/'index.html').write_text(_html(view,rows),encoding='utf-8')
    if view=='targets':
        for r in rows:
            target=folder/r['image'];target.parent.mkdir(exist_ok=True)
            shutil.copyfile(packet/'private/original'/f'{r["opaque_id"]}.png',target)
    elif view!='adjudication':
        for r in {r['opaque_id']:r for r in rows}.values():
            for key,kind in (('image','images'),('guide','guides')):
                if r[key]:
                    target=folder/r[key];target.parent.mkdir(exist_ok=True)
                    shutil.copyfile(packet/'private/views'/view.removeprefix('review-')/kind/f'{r["opaque_id"]}.png',target)
    manifest['forms'][view]=f'{view}/{filename}'

def prepare(parent,paths,output,config=DEFAULT_CONFIG,*,require_official=True):
    validate_config(config);source=parent_context(parent);_official(source,require_official)
    root=safe_output(output,paths,protected=(parent,source['e11_batch'],source['audit_directory']))
    packet=new_directory(root,'e20r2-audit');(packet/'private').mkdir()
    manifest=dict(schema='e20r2_person_roi_v1',status='running',errors=[],config=config,source_code=code_hashes(),
        feature_condition=source['feature_condition'],forms={},locks={},immutable={},encoding=None,created_at=now())
    write_json(packet/'input_manifest.json',manifest)
    try:
        write_json(packet/'private/parent_context.json',source)
        copy_evidence(ROOT,packet/'private/code_snapshot',manifest['source_code'])
        write_json(packet/'private/environment.json',environment())
        for r in source['mapping']:
            target=packet/'private/original'/f'{r["opaque_id"]}.png';target.parent.mkdir(exist_ok=True)
            shutil.copyfile(Path(parent)/'private/images/original'/target.name,target)
        rows=target_template(source['mapping'],_sizes(packet,source['mapping']))
        _open_form(packet,manifest,'targets',rows)
        (packet/'preregistration.md').write_text(
            '# E20R2 固定帧人物ROI协议\n\n原20视频/60帧，不替换样本。三输入native/letterbox/roi；'
            'ROI四周各扩10%后裁至原图，加边224。目标与可见性审核先冻结，三视图概念标签重新审核，不导入旧判断。'
            '目标框为新增监督；同时改变背景和人物尺度。无ROI保留空分数与NaN特征，无效行不能计作确定姿态。'
            '12帧校准子集不代表人工审核已完成。AI标签只能探索，人工复核未完成不能推进。'
            '全部60帧覆盖与适用子集指标分别报告；不训练、不重编码文本、final_test封存。\n',encoding='utf-8')
        verify_parent(source);_register(packet,manifest);manifest['status']='success'
    except Exception as exc:manifest['status']='failed';manifest['errors'].append(str(exc));raise
    finally:write_json(packet/'input_manifest.json',manifest)
    return packet

def _build_views(packet,source,targets):
    for row in targets:
        with Image.open(packet/'private/original'/f'{row["opaque_id"]}.png') as image:original=image.convert('RGB')
        images={'native':native_crop(np.array(original),source['encoder']),'letterbox':letterbox(original),'roi':roi_image(original,row)}
        for view,image in images.items():
            if image is None:continue
            for kind,value in [('images',image),('guides',guide_image(image,row,view))]:
                target=packet/'private/views'/view/kind/f'{row["opaque_id"]}.png';target.parent.mkdir(parents=True,exist_ok=True);value.save(target)

def _expected_images(packet,source,targets):
    for row in targets:
        with Image.open(packet/'private/original'/f'{row["opaque_id"]}.png') as image:original=image.convert('RGB')
        expected={'native':native_crop(np.array(original),source['encoder']),'letterbox':letterbox(original),'roi':roi_image(original,row)}
        for view,image in expected.items():
            path=packet/'private/views'/view/'images'/f'{row["opaque_id"]}.png'
            if image is None:
                if path.exists():raise ValueError('fabricated ROI for absent/ambiguous target')
                continue
            for kind,value in [('images',image),('guides',guide_image(image,row,view))]:
                with Image.open(packet/'private/views'/view/kind/f'{row["opaque_id"]}.png') as saved:
                    if not np.array_equal(np.array(saved.convert('RGB')),np.array(value)):raise ValueError('ROI/input pixel reconstruction mismatch')

def _input_hashes(packet,targets):
    return {f'{v}/{r["opaque_id"]}':sha256_file(packet/'private/views'/v/'images'/f'{r["opaque_id"]}.png')
            for v in VIEWS for r in targets if v!='roi' or r['target_status']=='selected'}

def _mask(targets):
    return np.array([[v!='roi' or r['target_status']=='selected' for r in targets] for v in VIEWS],dtype=bool)

def _score_rows(mapping,q,text,mask):
    values=q.astype('float64')@text.astype('float64').T
    return [dict(view=v,opaque_id=r['opaque_id'],target_id=target_key(r),concept_id=cid,
        status='encoded' if mask[i,j] else 'no_roi',cosine_score=format(values[i,j,c],'.17g') if mask[i,j] else '')
        for i,v in enumerate(VIEWS) for j,r in enumerate(mapping) for c,cid in enumerate(CONCEPTS)]

def _validate_encoding(packet,manifest,source,targets):
    if manifest['encoding'] is None:return []
    record=read_json(packet/'private/encoding/encoder.json')
    if record!=manifest['encoding'] or record['input_image_sha256']!=_input_hashes(packet,targets):raise ValueError('encoding input/identity changed')
    _verify_encoder(record['identity'],source['encoder']);frozen=record['frozen']
    if not frozen.get('unchanged') or frozen['before']!=frozen['after']:raise ValueError('CLIP parameters changed')
    if not source['encoder'].get('synthetic_test_fixture') and frozen['before']!=source['encoder']['parameter_sha256_before']:
        raise ValueError('CLIP parameters differ from parent')
    with np.load(packet/'private/encoding/features.npz',allow_pickle=False) as bank:q,text,mask=bank['Q'],bank['E'],bank['valid']
    original_text=np.asarray(source['text_embeddings'],dtype='float32')
    if (q.dtype!=np.float32 or q.shape!=(3,len(targets),original_text.shape[1]) or mask.dtype!=np.bool_
        or not np.array_equal(mask,_mask(targets)) or not np.array_equal(text,original_text)
        or not np.isfinite(q[mask]).all() or not np.isnan(q[~mask]).all()
        or not np.allclose(np.linalg.norm(q[mask],axis=-1),1,atol=2e-6)):
        raise ValueError('invalid masked features/text provenance')
    scores=_score_rows(source['mapping'],q,text,mask)
    if _read_csv(packet/'private/encoding/model_scores.csv')!=scores:raise ValueError('score forward reconstruction mismatch')
    return scores

def load_packet(packet,*,require_official=False):
    packet=Path(packet).resolve();manifest=read_json(packet/'input_manifest.json');validate_config(manifest['config'])
    if manifest['schema']!='e20r2_person_roi_v1' or manifest['status']!='success' or manifest['errors']:raise ValueError('incomplete/failed E20R2 packet')
    if manifest['source_code']!=code_hashes() or tree_hashes(packet/'private/code_snapshot')!=manifest['source_code']:raise ValueError('E20R2 source code changed')
    for name,h in manifest['immutable'].items():
        path=(packet/name).resolve()
        if not path.is_relative_to(packet) or sha256_file(path)!=h:raise ValueError('corrupt/unsafe E20R2 evidence')
    source=read_json(packet/'private/parent_context.json');verify_parent(source);_official(source,require_official)
    if manifest['feature_condition']!=source['feature_condition']:raise ValueError('feature condition changed')
    mapping=source['mapping'];sizes=_sizes(packet,mapping)
    for r in mapping:
        if sha256_file(packet/'private/original'/f'{r["opaque_id"]}.png')!=sha256_file(Path(source['parent'])/'private/images/original'/f'{r["opaque_id"]}.png'):
            raise ValueError('source original image changed')
    n=len(manifest['locks']);available=STAGES[:min(n+1,len(STAGES))]
    if set(manifest['locks'])!=set(STAGES[:n]) or set(manifest['forms'])!=set(available):raise ValueError('invalid target/review freeze order')
    forms={};hashes={}
    for view in available:
        filename='targets.csv' if view=='targets' else 'judgments.csv'
        if manifest['forms'][view]!=f'{view}/{filename}':raise ValueError('unsafe form path')
        template=target_template(mapping,sizes) if view=='targets' else expected_template(view,source,forms['targets'],forms)
        path=packet/manifest['forms'][view];hashes[view]=sha256_file(path);rows=_read_csv(path)
        forms[view]=(validate_targets if view=='targets' else validate_concepts)(rows,template,complete=view in manifest['locks'])
        if sha256_file(path)!=hashes[view] or (packet/view/'index.html').read_text(encoding='utf-8')!=_html(view,template):raise ValueError('form changed concurrently/HTML changed')
        if view in manifest['locks']:
            lock=manifest['locks'][view]
            if lock['sha256']!=hashes[view] or sha256_file(packet/'private/frozen'/f'{view}.csv')!=hashes[view] or not lock['frozen_at']:
                raise ValueError('frozen target/judgment changed')
        for r in {r['opaque_id']:r for r in template}.values():
            for key,kind in [('image','images'),('guide','guides')]:
                if r.get(key):
                    expected=packet/'private/original'/f'{r["opaque_id"]}.png' if view=='targets' else packet/'private/views'/r['view']/kind/f'{r["opaque_id"]}.png'
                    if sha256_file(packet/view/r[key])!=sha256_file(expected):raise ValueError('review image differs from actual input')
    primary_names={r['reviewer'].strip().casefold() for v in ('targets',*VIEWS) if v in forms for r in forms[v] if r['reviewer'].strip()}
    origins={}
    for view,rows in forms.items():
        for r in rows:
            name=r['reviewer'].strip().casefold()
            if not name:continue
            if view.startswith('review-') and name in primary_names:raise ValueError('independent reviewer must differ from primary/target annotator')
            if name in origins and origins[name]!=r['reviewer_origin']:raise ValueError('reviewer origin inconsistent')
            origins[name]=r['reviewer_origin']
    if 'targets' in manifest['locks']:_expected_images(packet,source,forms['targets'])
    elif manifest['encoding'] is not None:raise ValueError('cannot encode before target freeze')
    scores=_validate_encoding(packet,manifest,source,forms['targets']) if 'targets' in manifest['locks'] else []
    if set(tree_hashes(packet))!=set(manifest['immutable'])|set(manifest['forms'].values())|{'input_manifest.json'}:raise ValueError('unregistered/missing E20R2 evidence')
    return dict(packet=packet,manifest=manifest,source=source,mapping=mapping,targets=forms['targets'],forms=forms,form_hashes=hashes,scores=scores)

def freeze(packet,view):
    context=load_packet(packet);packet=context['packet'];manifest=context['manifest'];n=len(manifest['locks'])
    if n>=len(STAGES) or STAGES[n]!=view:raise ValueError('freeze currently available stage; frozen forms cannot be overwritten')
    template=target_template(context['mapping'],_sizes(packet,context['mapping'])) if view=='targets' else expected_template(view,context['source'],context['targets'],context['forms'])
    (validate_targets if view=='targets' else validate_concepts)(context['forms'][view],template,complete=True)
    before=digest(manifest);expected=context['form_hashes'][view];frozen=packet/'private/frozen'/f'{view}.csv';frozen.parent.mkdir(exist_ok=True)
    if frozen.exists():raise ValueError('refusing frozen evidence overwrite')
    try:
        shutil.copyfile(packet/manifest['forms'][view],frozen)
        if sha256_file(frozen)!=expected:raise ValueError('form changed during freeze')
        manifest['locks'][view]=dict(sha256=expected,frozen_at=now())
        if view=='targets':_build_views(packet,context['source'],context['targets'])
        if n+1<len(STAGES):
            following=STAGES[n+1]
            _open_form(packet,manifest,following,expected_template(following,context['source'],context['targets'],context['forms']))
        if digest(read_json(packet/'input_manifest.json'))!=before or sha256_file(packet/manifest['forms'][view])!=expected:raise ValueError('packet changed during freeze')
        _register(packet,manifest)
    except Exception as exc:manifest['status']='failed';manifest['errors'].append(str(exc));raise
    finally:write_json(packet/'input_manifest.json',manifest)
    return packet

def encode(packet,paths,*,device='cuda',encoder=None,require_official=True):
    c=load_packet(packet,require_official=require_official);packet=c['packet'];manifest=c['manifest'];source=c['source']
    if 'targets' not in manifest['locks']:raise ValueError('freeze target audit before encoding')
    if manifest['encoding'] is not None or (packet/'private/encoding').exists():raise ValueError('refusing encoding overwrite')
    before=tree_hashes(packet);mask=_mask(c['targets']);images=[]
    for i,v in enumerate(VIEWS):
        for j,r in enumerate(c['mapping']):
            if mask[i,j]:
                with Image.open(packet/'private/views'/v/'images'/f'{r["opaque_id"]}.png') as image:images.append(image.convert('RGB'))
    encoder=encoder or PairedCLIP(paths['weights_file'],device=device);_verify_encoder(encoder.identity,source['encoder'])
    q=np.full((3,len(c['mapping']),len(source['text_embeddings'][0])),np.nan,dtype='float32');q[mask]=encoder.encode_images(images)
    record=dict(identity=encoder.identity,frozen=encoder.verify_frozen(),input_image_sha256=_input_hashes(packet,c['targets']),
        encoded_at=now(),n_images=int(mask.sum()),text_encoding_executed=False,gradient_fits_executed=0)
    if tree_hashes(packet)!=before:raise ValueError('packet changed during encoding')
    folder=packet/'private/encoding';folder.mkdir()
    try:
        text=np.asarray(source['text_embeddings'],dtype='float32');np.savez_compressed(folder/'features.npz',Q=q,E=text,valid=mask)
        scores=_score_rows(c['mapping'],q,text,mask);write_csv(folder/'model_scores.csv',scores,list(scores[0]));write_json(folder/'encoder.json',record)
        manifest['encoding']=record;_validate_encoding(packet,manifest,source,c['targets']);_register(packet,manifest)
    except Exception as exc:manifest['status']='failed';manifest['errors'].append(str(exc));raise
    finally:write_json(packet/'input_manifest.json',manifest)
    return packet
