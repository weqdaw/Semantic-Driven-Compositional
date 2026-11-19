"""Read-only analysis of verified E1.1 human audit packets; no annotation or fitting."""
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np

from .e11_audit import select_videos
from .e11_features import DEFAULT_CONFIG as C11
from .e11_workflow import tree_hashes
from .e12_workflow import parent_context
from .provenance import digest,read_json,sha256_file,write_csv,write_json
from .reporting import _read_csv

HUMAN_FIELDS=('judgment','reviewer','reviewed_at','evidence')


def analyze_audit(directory: Path,e11_batch: Path):
    directory,e11_batch=Path(directory),Path(e11_batch)
    context=parent_context(e11_batch); manifest=read_json(directory/'audit_manifest.json')
    selected=select_videos(context['batch']['evaluation_ids'],C11)
    if (manifest['protocol_id']!=context['batch']['protocol_id'] or manifest['config']!=C11 or
            manifest['selected_video_ids']!=selected or manifest['n_judgments']!=180 or
            manifest['parent_evidence_sha256']!=digest(tree_hashes(e11_batch/'parent'))):
        raise ValueError('audit packet differs from original E1/development evidence')
    frozen=manifest['files']
    required={'judgments.csv','judgments.template.csv','model_scores.after_review.csv','frames.csv'}
    if not required.issubset(frozen): raise ValueError('incomplete audit packet')
    if frozen['judgments.csv']!=frozen['judgments.template.csv']:
        raise ValueError('initial audit judgments must match the frozen blank template')
    for name,expected in frozen.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()): raise ValueError('unsafe audit packet path')
        if name!='judgments.csv' and sha256_file(path)!=expected: raise ValueError('changed frozen audit material')
    judgments_hash=sha256_file(directory/'judgments.csv'); manifest_hash=sha256_file(directory/'audit_manifest.json')
    template=_read_csv(directory/'judgments.template.csv'); rows=_read_csv(directory/'judgments.csv')
    scores=_read_csv(directory/'model_scores.after_review.csv')
    if len(template)!=180 or len(rows)!=180 or len(scores)!=180: raise ValueError('incomplete audit rows')
    if any(any(r.get(k)!='' for k in HUMAN_FIELDS) for r in template):
        raise ValueError('frozen audit template must have blank human review fields')
    entries={e['video_id']:e for e in context['index']['entries']}
    positions={e['video_id']:j for j,e in enumerate(context['index']['entries'])}
    frame_rows=_read_csv(directory/'frames.csv'); frame_map={}
    for frame in frame_rows:
        vid,pos=frame['video_id'],int(frame['frame_position_zero_based'])
        key=(vid,pos)
        if (vid not in selected or pos not in C11['audit']['frame_positions_zero_based'] or key in frame_map or
                int(frame['frame_index'])!=entries[vid]['frame_indices'][pos] or frame['video_sha256']!=entries[vid]['video_sha256']):
            raise ValueError('audit frame map differs from frozen video/frame identity')
        image_id=digest([vid,int(frame['frame_index'])])[:20]
        if (frame['original_image']!=f'images/{image_id}-original.png' or frame['clip_image']!=f'images/{image_id}-clip.png' or
                any(frame[k] not in frozen for k in ('original_image','clip_image'))):
            raise ValueError('audit image is missing from the frozen packet manifest')
        frame_map[key]=frame
    if set(frame_map)!={(v,p) for v in selected for p in C11['audit']['frame_positions_zero_based']}:
        raise ValueError('incomplete audit frame/image coverage')
    catalog=read_json(e11_batch/'parent/concept_catalog.json')['concepts']
    concepts={c['id']:j for j,c in enumerate(catalog)}
    with np.load(e11_batch/'parent/feature_bank.npz',allow_pickle=False) as bank: frame_scores=bank['C']
    expected_keys={(v,p,c) for v in selected for p in C11['audit']['frame_positions_zero_based'] for c in C11['audit']['concept_ids']}
    seen=set(); judged=[]
    for original,row,score in zip(template,rows,scores):
        if set(row)!=set(original) or any(row[k]!=v for k,v in original.items() if k not in HUMAN_FIELDS):
            raise ValueError('audit row differs from frozen template identity')
        key=(row['video_id'],int(row['frame_position_zero_based']),row['concept_id'])
        if key not in expected_keys or key in seen: raise ValueError('changed/repeated audit row identity')
        seen.add(key); vid,pos,cid=key
        if int(row['frame_index'])!=entries[vid]['frame_indices'][pos]: raise ValueError('wrong audit frame index')
        if any(row[k]!=frame_map[vid,pos][k] for k in ('original_image','clip_image')):
            raise ValueError('audit row image differs from frozen frame map')
        for field in ('video_id','frame_position_zero_based','frame_index','concept_id'):
            if score[field]!=row[field]: raise ValueError('audit score row identity mismatch')
        value=float(score['cosine_score'])
        if not np.isfinite(value) or value!=float(frame_scores[positions[vid],pos,concepts[cid]]):
            raise ValueError('audit model score differs from frozen frame bank')
        judgment=row['judgment'].strip()
        if judgment not in ('','present','absent','uncertain'): raise ValueError('unsupported human judgment')
        if judgment:
            if not row['reviewer'].strip() or not row['evidence'].strip(): raise ValueError('human review needs reviewer/evidence')
            try: reviewed=datetime.fromisoformat(row['reviewed_at'].replace('Z','+00:00'))
            except ValueError as exc: raise ValueError('reviewed_at must be timezone-aware ISO date') from exc
            if reviewed.tzinfo is None or reviewed.utcoffset() is None: raise ValueError('reviewed_at requires timezone')
        judged.append(dict(video_id=vid,concept_id=cid,judgment=judgment,score=value))
    if seen!=expected_keys: raise ValueError('audit frame/concept coverage mismatch')
    counts={j:sum(r['judgment']==j for r in judged) for j in ('','present','absent','uncertain')}
    complete=counts['']==0; output=[]
    for cid in C11['audit']['concept_ids']:
        part=[r for r in judged if r['concept_id']==cid]
        positive=[r['score'] for r in part if r['judgment']=='present']; negative=[r['score'] for r in part if r['judgment']=='absent']
        auc=None
        if complete and positive and negative:
            difference=np.asarray(positive)[:,None]-np.asarray(negative)[None,:]
            auc=float(np.mean((difference>0)+.5*(difference==0)))
        output.append(dict(concept_id=cid,n_rows=len(part),n_present=len(positive),n_absent=len(negative),
                           n_uncertain=sum(r['judgment']=='uncertain' for r in part),n_blank=sum(r['judgment']=='' for r in part),
                           determinable_fraction=(len(positive)+len(negative))/len(part),
                           n_present_videos=len({r['video_id'] for r in part if r['judgment']=='present'}),
                           n_absent_videos=len({r['video_id'] for r in part if r['judgment']=='absent'}),
                           present_score_median=float(np.median(positive)) if complete and positive else None,
                           absent_score_median=float(np.median(negative)) if complete and negative else None,auroc=auc))
    if sha256_file(directory/'judgments.csv')!=judgments_hash or sha256_file(directory/'audit_manifest.json')!=manifest_hash:
        raise ValueError('audit input changed during analysis')
    return dict(protocol_id=manifest['protocol_id'],audit_directory=str(directory.resolve()),human_review_completed=complete,
                n_judgments=180,n_blank=counts[''],n_present=counts['present'],n_absent=counts['absent'],n_uncertain=counts['uncertain'],
                judgments_sha256=judgments_hash,manifest_sha256=manifest_hash,
                template_sha256=sha256_file(directory/'judgments.template.csv'),concepts=output)


def write_audit_report(result: dict,output: Path):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('refusing to overwrite audit report')
    output.mkdir(parents=True,exist_ok=True)
    source=Path(result['audit_directory'])
    for name,key in (('judgments.csv','judgments_sha256'),('judgments.template.csv','template_sha256'),('audit_manifest.json','manifest_sha256')):
        if sha256_file(source/name)!=result[key]: raise ValueError('audit changed during report snapshot')
        shutil.copyfile(source/name,output/name)
    write_json(output/'audit-summary.json',result)
    write_csv(output/'audit-concepts.csv',result['concepts'],list(result['concepts'][0]))
    lines=['# E1.2 概念人工审核汇总','', '人工审核已完成' if result['human_review_completed'] else '人工审核未完成：不生成分数分布/AUROC结论', '',
           f"总行数180；空白{result['n_blank']}；present {result['n_present']}；absent {result['n_absent']}；uncertain {result['n_uncertain']}。", '',
           '| 概念 | present | absent | uncertain | blank | 可判定率 | AUROC |','|---|---:|---:|---:|---:|---:|---:|']
    for r in result['concepts']:
        auc='不计算' if r['auroc'] is None else f"{r['auroc']:.3f}"
        lines.append(f"| {r['concept_id']} | {r['n_present']} | {r['n_absent']} | {r['n_uncertain']} | {r['n_blank']} | {r['determinable_fraction']:.1%} | {auc} |")
    lines+=['','AUROC仅描述可判定行的分数排序，不调阈值、不作为概率。180行来自10个视频，具有聚类相关性。',
            '文件路径可能暴露动作类别，不宣称严格盲审。人工标签只用于本诊断，不进入分类器拟合或概念选择。','']
    (output/'audit-summary.md').write_text('\n'.join(lines),encoding='utf-8')
