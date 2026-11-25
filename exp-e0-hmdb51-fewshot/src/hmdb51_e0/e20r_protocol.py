"""Prespecified visibility repair, grouped cohorts and annotation provenance."""
import json
from pathlib import Path
import numpy as np
import yaml
from .e20_protocol import HUMAN_FIELDS as BASE_HUMAN, validate_judgments

CONCEPTS=('pose01','pose02','pose03','pose04','pose06','pose14')
VIEWS=('native','letterbox','original','review-native','review-letterbox','review-original','adjudication')
HUMAN_FIELDS=(*BASE_HUMAN,'reviewer_origin')
SELECTION_FIELDS=('video_id','target_concept_id','target_label','selection_reason','selector','selection_origin')
DEFAULT_CONFIG=dict(experiment_id='e20r_coverage_repair',version=1,evaluation_stage='development',final_test_enabled=False,
    concept_ids=list(CONCEPTS),wrong_concept_ids=[*CONCEPTS[1:],CONCEPTS[0]],seed=20261124,
    videos_per_class=4,source_cap=2,min_sources=10,frame_positions=[2,7,13],review_videos=6,
    supplement_max_videos=12,letterbox_size=224,letterbox_rgb=[123,117,104])
_SIGNATURE=json.dumps(DEFAULT_CONFIG,sort_keys=True,allow_nan=False)
RULES={
 'pose01':'两只手及胸前位置均可辨认；明确在胸前靠近记present，位置不清或接近边界记uncertain；不从动作名称推断。',
 'pose02':'双手均可见、在躯干前方且明显分开才记present；只看到侧展而前后平面不明记uncertain。',
 'pose03':'比较手掌中心与同侧肩：至少一手明显高于肩记present；判absent需两手均可判断；高度接近、肩遮挡记uncertain。',
 'pose04':'至少一臂可见从肩向侧方伸展记present；视角或遮挡使侧方方向不明确记uncertain。',
 'pose06':'两臂及肩均可见且两臂均低于肩记present；一臂明显高于肩记absent；缺一臂或高度不清记uncertain。',
 'pose14':'需躯干和腰部可见，按身体朝向判断前倾；镜头倾斜、仅头部或方向不明记uncertain。',
}

def validate_config(config):
    if json.dumps(config,sort_keys=True,allow_nan=False)!=_SIGNATURE:
        raise ValueError('E2.0-R requires the fixed visibility-repair protocol')

def load_config(path):
    value=yaml.safe_load(Path(path).read_text(encoding='utf-8'));validate_config(value);return value

def select_representative(rows,excluded_groups,excluded_ids=()):
    eligible=[r for r in rows if r['experimental_role']=='support_pool' and r['source_group_id'] not in excluded_groups and r['video_id'] not in excluded_ids]
    if len({r['video_id'] for r in eligible})!=len(eligible):raise ValueError('duplicate source video identity')
    rng=np.random.Generator(np.random.PCG64(DEFAULT_CONFIG['seed']))
    pools={}
    for c in range(5):
        part=sorted([r for r in eligible if r['class_id']==c],key=lambda r:r['video_id'])
        pools[c]=[part[i] for i in rng.permutation(len(part))]
    counts={};selected=[]
    for _ in range(4):
        for c in range(5):
            options=[r for r in pools[c] if counts.get(r['source_group_id'],0)<2]
            if not options:raise ValueError('insufficient source-isolated videos for balanced20-video plan')
            chosen=min(options,key=lambda r:counts.get(r['source_group_id'],0))
            selected.append(chosen.copy());pools[c].remove(chosen)
            group=chosen['source_group_id'];counts[group]=counts.get(group,0)+1
    if len(counts)<10:raise ValueError('insufficient distinct sources')
    return selected

def select_supplement(rows,selection,excluded_groups,excluded_ids):
    by_id={r['video_id']:r for r in rows};chosen={}
    if not selection or any(set(r)!=set(SELECTION_FIELDS) for r in selection):raise ValueError('invalid supplement selection columns')
    keys=set()
    for r in selection:
        key=(r['video_id'],r['target_concept_id'],r['target_label'])
        if (key in keys or r['target_concept_id'] not in CONCEPTS or r['target_label'] not in ('present','absent')
            or r['selection_origin'] not in ('human','ai') or any(not r[k].strip() for k in SELECTION_FIELDS)):
            raise ValueError('invalid/duplicate supplement target metadata')
        keys.add(key);v=by_id.get(r['video_id'])
        if (v is None or v['experimental_role']!='support_pool' or v['source_group_id'] in excluded_groups or v['video_id'] in excluded_ids):
            raise ValueError('supplement overlaps excluded/test/base sources')
        chosen[v['video_id']]=v.copy()
    if not 1<=len(chosen)<=12:raise ValueError('supplement requires1..12unique videos')
    values=[chosen[k] for k in sorted(chosen)]
    if any(sum(v['source_group_id']==r['source_group_id'] for v in values)>2 for r in values):raise ValueError('supplement source cap exceeded')
    return values

def review_frames(mapping):
    rng=np.random.Generator(np.random.PCG64(DEFAULT_CONFIG['seed']))
    videos=sorted({r['anonymous_video_id'] for r in mapping});n=min(6,len(videos))
    result=[]
    for v in rng.choice(videos,n,replace=False):
        options=sorted(r['opaque_id'] for r in mapping if r['anonymous_video_id']==v)
        result.append(str(rng.choice(options)))
    return sorted(result)

def make_template(mapping,concepts,view):
    if view=='adjudication':raise ValueError('adjudication requires both review records')
    pixel_view=view.removeprefix('review-');selected=set(review_frames(mapping)) if view.startswith('review-') else None
    return [dict(opaque_id=r['opaque_id'],concept_id=c['concept_id'],view=pixel_view,image=f"images/{r['opaque_id']}.png",
        concept_label=c['concept_label'],description_en=c['description_en'],operational_rule=RULES[c['concept_id']],
        **{k:'' for k in HUMAN_FIELDS}) for r in mapping if selected is None or r['opaque_id'] in selected for c in concepts]

def validate_form(rows,template,*,complete=False):
    clean=lambda seq:[{k:v for k,v in r.items() if k!='reviewer_origin'} for r in seq]
    validate_judgments(clean(rows),clean(template),complete=complete)
    for r in rows:
        if (r['judgment'].strip() and r['reviewer_origin'] not in ('human','ai')) or (not r['judgment'].strip() and r['reviewer_origin']):
            raise ValueError('every judgment requires explicit human/ai reviewer_origin')
    return rows

def fully_human(forms):
    return bool(forms and all(rows and all(r['judgment'].strip() and r['reviewer_origin']=='human' for r in rows) for rows in forms))
