"""Full-denominator applicability, target-ROI effects and source-paired uncertainty."""
from pathlib import Path
from itertools import combinations
import shutil
import numpy as np
from .e20r2_protocol import CONCEPTS,VIEWS,eligible,calibration_ids
from .e20r2_workflow import load_packet,STAGES
from .e20_metrics import ranking_metrics,paired_cluster_interval
from .e20r_reporting import usable_concept
from .e11_workflow import tree_hashes
from .provenance import now,write_csv,write_json,sha256_file
from .e1_experiment import OFFICIAL

def _interval(y,a,b,groups):
    if len(set(y))<2:return dict(n_sources=len(set(groups)),n_valid=0,correct_ci_low=None,correct_ci_high=None,delta_ci_low=None,delta_ci_high=None)
    return paired_cluster_interval(y,a,b,groups)

def _alignment(cid,view,frames,labels,scores,mapping,wrong):
    part=[labels[f,cid,view] for f in frames];known=[r for r in part if r['judgment'].strip() in ('present','absent') and scores[view,r['opaque_id'],cid] is not None]
    y=[int(r['judgment'].strip()=='present') for r in known]
    a=[scores[view,r['opaque_id'],cid] for r in known];b=[scores[view,r['opaque_id'],wrong] for r in known]
    aa,bb=ranking_metrics(y,a),ranking_metrics(y,b)
    pos=[r for r in known if r['judgment'].strip()=='present'];neg=[r for r in known if r['judgment'].strip()=='absent']
    result=dict(concept_id=cid,view=view,n_rows=len(frames),n_present=len(pos),n_absent=len(neg),n_uncertain=len(frames)-len(known),
        determinable_fraction=len(known)/len(frames) if frames else 0.,
        n_present_videos=len({mapping[r['opaque_id']]['video_id'] for r in pos}),n_absent_videos=len({mapping[r['opaque_id']]['video_id'] for r in neg}),
        wrong_concept_id=wrong,correct_auroc=aa['auroc'],wrong_auroc=bb['auroc'],
        auroc_delta=aa['auroc']-bb['auroc'] if aa['auroc'] is not None else None,
        correct_average_precision=aa['average_precision'],prevalence=aa['prevalence'])
    result.update(_interval(y,a,b,[mapping[r['opaque_id']]['source_group_id'] for r in known]));result['initially_usable']=usable_concept(result)
    return result

def analyze(c):
    mapping={r['opaque_id']:r for r in c['mapping']};frames=list(mapping)
    labels={(r['opaque_id'],r['concept_id'],r['view']):r for v in VIEWS for r in c['forms'][v]}
    if 'adjudication' in c['manifest']['locks']:
        for r in c['forms']['adjudication']:labels[r['opaque_id'],r['concept_id'],r['view']]=r
    scores={(r['view'],r['opaque_id'],r['concept_id']):float(r['cosine_score']) if r['cosine_score'] else None for r in c['scores']}
    applicable=[r['opaque_id'] for r in c['targets'] if eligible(r)]
    alignment=[];coverage=[];subset=[];paired=[];matched=[]
    for i,cid in enumerate(CONCEPTS):
        wrong=CONCEPTS[(i+1)%6]
        for view in VIEWS:
            a=_alignment(cid,view,frames,labels,scores,mapping,wrong);alignment.append(a)
            coverage.append(dict(concept_id=cid,view=view,n_rows=len(frames),n_present=a['n_present'],n_absent=a['n_absent'],
                n_uncertain=a['n_uncertain'],determinable_fraction=a['determinable_fraction'],
                n_encoded=sum(scores[view,f,cid] is not None for f in frames),
                **{'uncertain_'+reason:sum(labels[f,cid,view]['judgment'].strip()=='uncertain' and labels[f,cid,view]['reason_code']==reason for f in frames)
                    for reason in ('crop_missing','blur','occlusion','ambiguous_person','ground_unknown','other')}))
            subset.append(_alignment(cid,view,applicable,labels,scores,mapping,wrong))
            for f in frames:
                matched.append(dict(view=view,opaque_id=f,target_id=labels[f,cid,view]['target_id'],video_id=mapping[f]['video_id'],
                    source_group_id=mapping[f]['source_group_id'],concept_id=cid,judgment=labels[f,cid,view]['judgment'].strip(),
                    score=scores[view,f,cid],wrong_score=scores[view,f,wrong]))
        for first,second in combinations(VIEWS,2):
            known=lambda f,v:labels[f,cid,v]['judgment'].strip() in ('present','absent') and scores[v,f,cid] is not None
            common=[f for f in frames if known(f,first) and known(f,second) and labels[f,cid,first]['judgment'].strip()==labels[f,cid,second]['judgment'].strip()]
            y=[int(labels[f,cid,first]['judgment'].strip()=='present') for f in common]
            a=[scores[first,f,cid] for f in common];b=[scores[second,f,cid] for f in common]
            aa,bb=ranking_metrics(y,a)['auroc'],ranking_metrics(y,b)['auroc']
            entry=dict(concept_id=cid,first_view=first,second_view=second,n_rows=len(frames),n_common_consistent=len(common),
                n_conflicting_labels=sum(known(f,first) and known(f,second) and labels[f,cid,first]['judgment'].strip()!=labels[f,cid,second]['judgment'].strip() for f in frames),
                n_second_newly_determinable=sum(not known(f,first) and known(f,second) for f in frames),
                n_first_only_determinable=sum(known(f,first) and not known(f,second) for f in frames),
                first_auroc_common=aa,second_auroc_common=bb,second_minus_first_common=bb-aa if aa is not None else None)
            entry.update({'paired_'+k:v for k,v in _interval(y,b,a,[mapping[f]['source_group_id'] for f in common]).items()});paired.append(entry)
    agreement=[]
    for view in ('review-native','review-letterbox','review-roi'):
        if view in c['manifest']['locks']:
            for r in c['forms'][view]:
                first=next(p for p in c['forms'][r['view']] if p['opaque_id']==r['opaque_id'] and p['concept_id']==r['concept_id'])
                agreement.append(dict(view=r['view'],opaque_id=r['opaque_id'],concept_id=r['concept_id'],first_judgment=first['judgment'],
                    second_judgment=r['judgment'],agreed=first['judgment'].strip()==r['judgment'].strip()))
    return dict(concept_alignment=alignment,concept_coverage=coverage,eligible_alignment=subset,paired_inputs=paired,matched_scores=matched,agreement_rows=agreement)

def summarize(packet,output):
    c=load_packet(packet);p=c['packet'];m=c['manifest'];before=tree_hashes(p);output=Path(output).resolve()
    for protected in (p,Path(c['source']['parent']),Path(c['source']['e11_batch']),Path(c['source']['audit_directory'])):
        if output.is_relative_to(protected) or protected.is_relative_to(output):raise ValueError('report overlaps protected evidence')
    if c['source'].get('dataset_root'):
        raw=Path(c['source']['dataset_root']).resolve()
        if output.is_relative_to(raw) or raw.is_relative_to(output):raise ValueError('report overlaps raw dataset')
    if output.exists() and any(output.iterdir()):raise ValueError('refusing E20R2 report overwrite')
    frozen_targets='targets' in m['locks'];targets=c['targets'];ready=all(v in m['locks'] for v in VIEWS) and m['encoding'] is not None
    complete=all(v in m['locks'] for v in STAGES)
    human=complete and all(r['reviewer_origin']=='human' for rows in c['forms'].values() for r in rows)
    n=len(c['mapping']);roi=sum(r['target_status']=='selected' for r in targets) if frozen_targets else 0
    applicable=sum(eligible(r) for r in targets) if frozen_targets else 0
    result=dict(packet=str(p),created_at=now(),evaluation_stage='development',final_test_enabled=False,feature_condition=m['feature_condition'],
        n_frames=n,n_videos=len(c['source']['selected_videos']),n_source_groups=len({r['source_group_id'] for r in c['mapping']}),
        n_roi_available=roi,roi_availability_fraction=roi/n,n_eligible_body_frames=applicable,eligible_fraction=applicable/n,
        calibration_ids=calibration_ids(c['mapping']),calibration_human_completed=frozen_targets and all(r['reviewer_origin']=='human' for r in targets if r['calibration']=='yes'),
        analysis_ready=ready,review_completed=complete,independent_human_review_completed=bool(human),exploratory_continuation_gate=False,
        concept_alignment=[],concept_coverage=[],eligible_alignment=[],paired_inputs=[],matched_scores=[],agreement_rows=[],
        usable_concepts_by_view={},gate_by_view={},progress=[])
    for view,rows in c['forms'].items():
        key='target_status' if view=='targets' else 'judgment'
        result['progress'].append(dict(view=view,n_rows=len(rows),n_completed=sum(bool(r[key]) for r in rows),frozen=view in m['locks'],
            n_ai=sum(r['reviewer_origin']=='ai' for r in rows),human_time_seconds=sum(float(r['elapsed_seconds']) for r in rows if r['reviewer_origin']=='human' and r[key])))
    if ready:
        result.update(analyze(c))
        for v in VIEWS:
            result['usable_concepts_by_view'][v]=[r['concept_id'] for r in result['concept_alignment'] if r['view']==v and r['initially_usable']]
            result['gate_by_view'][v]=bool(len(result['usable_concepts_by_view'][v])>=3 and human and m['feature_condition']==OFFICIAL)
        result['exploratory_continuation_gate']=any(result['gate_by_view'].values())
    output.mkdir(parents=True,exist_ok=True)
    for name in ('concept_alignment','concept_coverage','eligible_alignment','paired_inputs','matched_scores','agreement_rows','progress'):
        if result[name]:write_csv(output/(name+'.csv'),result[name],list(result[name][0]))
    for view,path in m['forms'].items():
        target=output/'judgment_snapshots'/f'{view}.csv';target.parent.mkdir(exist_ok=True);shutil.copyfile(p/path,target)
    if frozen_targets:
        quality=[dict(**r,roi_available=r['target_status']=='selected',body_eligible=eligible(r)) for r in targets]
        write_csv(output/'target_quality.csv',quality,list(quality[0]))
    shutil.copyfile(p/'input_manifest.json',output/'input_manifest.snapshot.json')
    write_json(output/'provenance.json',dict(packet_manifest_sha256=sha256_file(p/'input_manifest.json'),packet_hashes=before,
        source_code=m['source_code'],config=m['config'],image_encoding_executed=False,gradient_fits_executed=0))
    if tree_hashes(p)!=before:raise ValueError('evidence changed during report')
    write_json(output/'summary.json',result);_report(output,result);return result

def _report(output,r):
    fmt=lambda v:'NA' if v is None else f'{v:.3f}'
    lines=['# E2.0-R2 固定人体目标与ROI对照','',
        '真实冻结CLIP输入；审核来源与完成状态以下表为准。' if r['feature_condition']==OFFICIAL else '仅程序验证，合成输入不构成研究成绩。','',
        f"{r['n_videos']}视频、{r['n_frames']}帧、{r['n_source_groups']}来源组；ROI可用{r['n_roi_available']}帧；有效人体条件满足{r['n_eligible_body_frames']}帧。",
        f"分析就绪：{r['analysis_ready']}；独立人工复核完成：{r['independent_human_review_completed']}；12帧人工校准完成：{r['calibration_human_completed']}。",'',
        '|阶段|完成/总项|AI项|冻结|','|---|---:|---:|---|']
    lines += [f"|{x['view']}|{x['n_completed']}/{x['n_rows']}|{x['n_ai']}|{x['frozen']}|" for x in r['progress']]
    if r['analysis_ready']:
        lines+=['','## 全部固定帧的覆盖与分数','',
            '|概念|输入|正/负/不确定|可判定率|AUROC|错配AUROC|差|初步可用|','|---|---|---:|---:|---:|---:|---:|---|']
        for x in r['concept_alignment']:
            lines.append(f"|{x['concept_id']}|{x['view']}|{x['n_present']}/{x['n_absent']}/{x['n_uncertain']}|{fmt(x['determinable_fraction'])}|{fmt(x['correct_auroc'])}|{fmt(x['wrong_auroc'])}|{fmt(x['auroc_delta'])}|{x['initially_usable']}|")
        lines+=['','## 同帧同目标配对','',
            '|概念|对照|共同一致帧|冲突|新增可判定|第一AUROC|第二AUROC|第二减第一|','|---|---|---:|---:|---:|---:|---:|---:|']
        for x in r['paired_inputs']:
            lines.append(f"|{x['concept_id']}|{x['first_view']}→{x['second_view']}|{x['n_common_consistent']}|{x['n_conflicting_labels']}|{x['n_second_newly_determinable']}|{fmt(x['first_auroc_common'])}|{fmt(x['second_auroc_common'])}|{fmt(x['second_minus_first_common'])}|")
    lines+=['','## 解释范围','',
        f"完整推进门槛：{r['exploratory_continuation_gate']}。",
        '目标框和审核标签属于新增监督。AI框与AI标签只支持探索，不能冒作人工校准或独立复核。',
        '原图上固定目标ID，审核参考框不进入模型输入。ROI同时改变人物尺度和背景，不能排他归因去背景或语义能力增强。',
        '无ROI行保留在全帧分母，特征NaN、分数空白，不伪装为零分数或负例。可判定子集和有效人体子集分别报告，条件AUROC不能外推总体。',
        '同帧三输入的配对只用均可判定且标签一致的共同子集。人物ID字段一致不证明真实视觉身份无误，需人工核验。',
        '单类AUROC/AP为NA；来源组bootstrap1000次、至少5来源/800有效，属于描述性区间。',
        '同一输入至少3概念需满足70%可判定、5正5负各3视频、AUROC≥.70、正确减错配≥.05，并完成独立人工链。',
        '固定帧不被替换，无有效人体的视频仍保留。视频内重采样是独立后续条件。',
        '没有新分类训练、文本编码、时序/组合泛化或溺水检测成绩，final_test继续封存。','']
    (output/'e20r2-results.md').write_text('\n'.join(lines),encoding='utf-8')
