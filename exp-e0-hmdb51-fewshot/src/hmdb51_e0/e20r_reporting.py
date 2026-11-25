"""Separate-cohort coverage and paired-input concept diagnostics."""
import numpy as np
from pathlib import Path
import shutil
from .e20_metrics import ranking_metrics,paired_cluster_interval
from .e20r_protocol import CONCEPTS, VIEWS, fully_human, SELECTION_FIELDS
from .provenance import write_json,write_csv,sha256_file,now
from .e11_workflow import tree_hashes
from .e1_experiment import OFFICIAL

def paired_input_metrics(rows):
    known=lambda x:x in ('present','absent')
    common=[r for r in rows if known(r['native_label']) and r['native_label']==r['letterbox_label']]
    conflicts=sum(known(r['native_label']) and known(r['letterbox_label']) and r['native_label']!=r['letterbox_label'] for r in rows)
    y=[int(r['native_label']=='present') for r in common]
    a=[r['native_score'] for r in common];b=[r['letterbox_score'] for r in common]
    aa,bb=ranking_metrics(y,a)['auroc'],ranking_metrics(y,b)['auroc']
    result=dict(n_pairs=len(rows),n_common_consistent=len(common),n_conflicting_labels=conflicts,
        n_letterbox_newly_determinable=sum(not known(r['native_label']) and known(r['letterbox_label']) for r in rows),
        n_native_only_determinable=sum(known(r['native_label']) and not known(r['letterbox_label']) for r in rows),
        native_auroc_common=aa,letterbox_auroc_common=bb,letterbox_minus_native_common=None if aa is None else bb-aa)
    interval=paired_cluster_interval(y,b,a,[r['source_group_id'] for r in common])
    result.update({('paired_'+k):v for k,v in interval.items()})
    return result


def usable_concept(row):
    return bool(row['determinable_fraction'] >= .70 and row['n_present'] >= 5 and row['n_absent'] >= 5
        and row['n_present_videos'] >= 3 and row['n_absent_videos'] >= 3
        and row['correct_auroc'] is not None and row['correct_auroc'] >= .70
        and row['auroc_delta'] is not None and row['auroc_delta'] >= .05-1e-12)


def analyze(context):
    forms, manifest = context['forms'], context['manifest']
    mapping = {r['opaque_id']: r for r in context['mapping']}
    primary = {(r['opaque_id'],r['concept_id'],r['view']):r for v in VIEWS[:3] for r in forms[v]}
    effective = dict(primary)
    if 'adjudication' in manifest['locks']:
        for r in forms['adjudication']:
            effective[r['opaque_id'],r['concept_id'],r['view']] = r
    score = {(r['opaque_id'],r['concept_id'],r['view']):float(r['cosine_score']) for r in context['scores']}
    coverage,alignment,retention,paired,matched,agreements = [],[],[],[],[],[]
    for c,cid in enumerate(CONCEPTS):
        for view in VIEWS[:3]:
            rows = [effective[f,cid,view] for f in mapping]
            counts = {label:sum(r['judgment'].strip()==label for r in rows) for label in ('present','absent','uncertain')}
            coverage.append(dict(concept_id=cid,view=view,n_rows=len(rows),
                **{'n_'+k:v for k,v in counts.items()},determinable_fraction=(counts['present']+counts['absent'])/len(rows),
                **{'uncertain_'+reason:sum(r['judgment'].strip()=='uncertain' and r['reason_code'].strip()==reason for r in rows)
                    for reason in ('crop_missing','occlusion','blur','ambiguous_person','ground_unknown','other')}))
            if view=='original':
                continue
            known = [r for r in rows if r['judgment'].strip() in ('present','absent')]
            positive = [r for r in known if r['judgment'].strip()=='present']
            negative = [r for r in known if r['judgment'].strip()=='absent']
            wrong = CONCEPTS[(c+1)%len(CONCEPTS)]
            y = [int(r['judgment'].strip()=='present') for r in known]
            a = [score[r['opaque_id'],cid,view] for r in known]
            b = [score[r['opaque_id'],wrong,view] for r in known]
            aa,bb = ranking_metrics(y,a),ranking_metrics(y,b)
            row = dict(concept_id=cid,view=view,wrong_concept_id=wrong,n_present=len(positive),n_absent=len(negative),
                n_uncertain=len(rows)-len(known),determinable_fraction=len(known)/len(rows),
                n_present_videos=len({mapping[r['opaque_id']]['video_id'] for r in positive}),
                n_absent_videos=len({mapping[r['opaque_id']]['video_id'] for r in negative}),
                correct_auroc=aa['auroc'],wrong_auroc=bb['auroc'],
                auroc_delta=None if aa['auroc'] is None else aa['auroc']-bb['auroc'],
                correct_average_precision=aa['average_precision'],prevalence=aa['prevalence'])
            row.update(paired_cluster_interval(y,a,b,[mapping[r['opaque_id']]['source_group_id'] for r in known]))
            row['initially_usable'] = usable_concept(row); alignment.append(row)
            for r in rows:
                frame=r['opaque_id']
                matched.append(dict(opaque_id=frame,concept_id=cid,view=view,judgment=r['judgment'].strip(),
                    source_group_id=mapping[frame]['source_group_id'],video_id=mapping[frame]['video_id'],
                    correct_score=score[frame,cid,view],wrong_score=score[frame,wrong,view]))
            pairs=[(effective[f,cid,'original']['judgment'].strip(),effective[f,cid,view]['judgment'].strip()) for f in mapping]
            n=sum(a=='present' for a,b in pairs); retained=sum(a==b=='present' for a,b in pairs)
            retention.append(dict(concept_id=cid,view=view,n_original_present=n,n_retained_present=retained,
                retention_rate=retained/n if n else None,
                original_present_input_uncertain=sum(a=='present' and b=='uncertain' for a,b in pairs),
                original_determinable_input_uncertain=sum(a in ('present','absent') and b=='uncertain' for a,b in pairs),
                **{a+'_to_'+b:sum(x==a and y==b for x,y in pairs)
                    for a in ('present','absent','uncertain') for b in ('present','absent','uncertain')}))
        input_rows=[dict(native_label=effective[f,cid,'native']['judgment'].strip(),
            letterbox_label=effective[f,cid,'letterbox']['judgment'].strip(),
            native_score=score[f,cid,'native'],letterbox_score=score[f,cid,'letterbox'],
            source_group_id=mapping[f]['source_group_id']) for f in mapping]
        paired.append(dict(concept_id=cid,**paired_input_metrics(input_rows)))
    for v in VIEWS[3:6]:
        if v in manifest['locks']:
            for second in forms[v]:
                first=primary[second['opaque_id'],second['concept_id'],second['view']]
                agreements.append(dict(opaque_id=second['opaque_id'],concept_id=second['concept_id'],view=second['view'],
                    first_judgment=first['judgment'],second_judgment=second['judgment'],
                    agreed=first['judgment'].strip()==second['judgment'].strip()))
    return dict(concept_coverage=coverage,concept_alignment=alignment,input_retention=retention,paired_inputs=paired,
                matched_scores=matched,agreement_rows=agreements)


def summarize(packet, output):
    from .e20r_workflow import load_packet
    context=load_packet(packet);manifest=context['manifest'];packet=context['packet']
    before=tree_hashes(packet);output=Path(output).resolve()
    for protected in (packet,Path(manifest['plan_path']).parent,context['source']['e11_batch'],context['source']['audit_directory'],
                      Path(context['plan']['dataset_root'])):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('report output overlaps protected repair evidence')
    if output.exists() and any(output.iterdir()):raise ValueError('refusing to overwrite repair report')
    forms,locks=context['forms'],manifest['locks']
    ready=all(v in locks for v in VIEWS[:3]) and manifest['encoding'] is not None
    reviewed=all(v in locks for v in VIEWS)
    human=reviewed and fully_human([forms[v] for v in VIEWS[:6]]) and all(r['reviewer_origin']=='human' for r in forms['adjudication'])
    result=dict(cohort=manifest['cohort'],packet=str(packet),evaluation_stage='development',final_test_enabled=False,
        feature_condition=manifest['feature_condition'],analysis_ready=ready,review_completed=reviewed,
        independent_human_review_completed=bool(human),exploratory_continuation_gate=False,
        n_videos=len(context['plan']['selected_videos']),n_frames=len(context['mapping']),
        n_source_groups=len({r['source_group_id'] for r in context['mapping']}),created_at=now(),
        concept_coverage=[],concept_alignment=[],input_retention=[],paired_inputs=[],matched_scores=[],agreement_rows=[],
        usable_concepts_by_view={},gate_by_view={},progress=[])
    result['progress']=[dict(view=v,n_rows=len(rows),n_judged=sum(bool(r['judgment'].strip()) for r in rows),
        n_ai=sum(r['reviewer_origin']=='ai' for r in rows),frozen=v in locks,
        human_time_seconds=sum(float(r['elapsed_seconds']) for r in rows if r['judgment'].strip() and r['reviewer_origin']=='human'))
        for v,rows in forms.items()]
    if ready:
        result.update(analyze(context))
        for view in ('native','letterbox'):
            usable=[r['concept_id'] for r in result['concept_alignment'] if r['view']==view and r['initially_usable']]
            result['usable_concepts_by_view'][view]=usable
            result['gate_by_view'][view]=bool(len(usable)>=3 and human and manifest['cohort']=='representative'
                and manifest['feature_condition']==OFFICIAL)
        result['exploratory_continuation_gate']=any(result['gate_by_view'].values())
    output.mkdir(parents=True,exist_ok=True)
    for view,path in manifest['forms'].items():
        target=output/'judgment_snapshots'/f'{view}.csv';target.parent.mkdir(exist_ok=True)
        shutil.copyfile(packet/path,target)
    for name in ('concept_coverage','concept_alignment','input_retention','paired_inputs','matched_scores','agreement_rows','progress'):
        if result[name]:write_csv(output/(name+'.csv'),result[name],list(result[name][0]))
    gaps=[]
    for r in result['concept_alignment']:
        gaps.append(dict(concept_id=r['concept_id'],view=r['view'],missing_present=max(0,5-r['n_present']),
            missing_absent=max(0,5-r['n_absent']),missing_present_videos=max(0,3-r['n_present_videos']),
            missing_absent_videos=max(0,3-r['n_absent_videos']),determinable_fraction=r['determinable_fraction']))
    if gaps:write_csv(output/'coverage_gaps.csv',gaps,list(gaps[0]))
    write_csv(output/'supplement-selection.template.csv',[],list(SELECTION_FIELDS))
    shutil.copyfile(packet/'input_manifest.json',output/'input_manifest.snapshot.json')
    write_json(output/'provenance.json',dict(packet_manifest_sha256=sha256_file(packet/'input_manifest.json'),
        packet_hashes=before,source_code=manifest['source_code'],config=manifest['config'],
        report_image_encoding_executed=False,report_text_encoding_executed=False,gradient_fits_executed=0))
    if tree_hashes(packet)!=before:raise ValueError('repair evidence changed during reporting')
    write_json(output/'summary.json',result)
    _write_report(output,result)
    return result


def _write_report(output,r):
    fmt=lambda v:'NA' if v is None else f'{v:.3f}'
    lines=['# E2.0-R：概念适用性与样本覆盖修复','',
        '正式冻结CLIP development诊断' if r['feature_condition']==OFFICIAL else '仅程序验证：合成输入不构成研究结果','',
        f"队列：{r['cohort']}；{r['n_videos']}视频、{r['n_frames']}帧、{r['n_source_groups']}来源组。",
        f"分析就绪：{r['analysis_ready']}；全部复核与裁决冻结：{r['review_completed']}；人工审核链完整：{r['independent_human_review_completed']}。",'',
        '|阶段|已判断/总量|AI记录|冻结|','|---|---:|---:|---|']
    lines += [f"|{x['view']}|{x['n_judged']}/{x['n_rows']}|{x['n_ai']}|{x['frozen']}|" for x in r['progress']]
    if r['analysis_ready']:
        lines += ['','## 概念分数与覆盖','',
            '|概念|输入|正/负/不确定|可判定率|AUROC|错配AUROC|差|AP|初步可用|',
            '|---|---|---:|---:|---:|---:|---:|---:|---|']
        for x in r['concept_alignment']:
            lines.append(f"|{x['concept_id']}|{x['view']}|{x['n_present']}/{x['n_absent']}/{x['n_uncertain']}|{fmt(x['determinable_fraction'])}|{fmt(x['correct_auroc'])}|{fmt(x['wrong_auroc'])}|{fmt(x['auroc_delta'])}|{fmt(x['correct_average_precision'])}|{x['initially_usable']}|")
        lines += ['','## 同帧输入对照','',
            '|概念|共同可判定且一致|标签冲突|加边新增可判定|原生AUROC|加边AUROC|差|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for x in r['paired_inputs']:
            lines.append(f"|{x['concept_id']}|{x['n_common_consistent']}|{x['n_conflicting_labels']}|{x['n_letterbox_newly_determinable']}|{fmt(x['native_auroc_common'])}|{fmt(x['letterbox_auroc_common'])}|{fmt(x['letterbox_minus_native_common'])}|")
    lines += ['','## 解释边界','',
        f"探索性继续门槛：{r['exploratory_continuation_gate']}；分输入门槛：{r['gate_by_view']}。",
        'AI标签只能支持探索性诊断；不同审核者名称不能证明实际独立。未完成审核、编码或冻结时只报告进度。',
        '定向补样单独统计，不能与自然抽样合并估计出现率或满足代表性队列推进门槛。',
        '来源隔离仅针对已有保守来源分组的训练池与开发集；最终测试尚无来源审核，不能声称已证明与其来源独立。',
        '不确定标签保留并分原因统计；单类AUROC/AP为NA。输入效果只在同帧、可判定且标签一致的共同子集比较；冲突另列。',
        '来源聚类bootstrap固定1000次、最少5来源且800次有效，区间为描述性不确定性；相邻帧不作为独立视频。',
        '门槛：同一输入至少3概念，可判定率≥.70、各5正5负且各3视频、AUROC≥.70、较固定错配高≥.05，并完成人工复核裁决。',
        '原图仅作可见性参照，没有原图模型分数。本实验不训练分类器，不验证时序、组合泛化或真实溺水检测；final_test封存。','']
    (output/'e20r-results.md').write_text('\n'.join(lines),encoding='utf-8')
