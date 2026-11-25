"""Fixed-frame person-conditioned ROI protocol, with explicit localization cost."""
from pathlib import Path
from datetime import datetime
import json,math
import numpy as np
import yaml
from PIL import Image,ImageDraw
from .e20r_protocol import CONCEPTS,HUMAN_FIELDS,RULES,validate_form
from .e20r_media import letterbox

VIEWS=('native','letterbox','roi')
DEFAULT_CONFIG=dict(experiment_id='e20r2_person_roi',version=1,evaluation_stage='development',final_test_enabled=False,
    concept_ids=list(CONCEPTS),views=list(VIEWS),roi_padding_fraction=.10,min_target_height_px=80,
    calibration_videos=12,seed=20261124,target_rule='per_frame_largest_visible_person_then_leftmost_fixed_across_views',frame_selection='fixed_parent60')
TARGET_FIELDS=('person_status','target_status','x1','y1','x2','y2','shoulders_visible','hands_visible','waist_visible',
               'target_description','reviewer','reviewer_origin','reviewed_at','evidence','elapsed_seconds')

def validate_config(value):
    if json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(DEFAULT_CONFIG,sort_keys=True):
        raise ValueError('E20R2 fixed development/person-ROI protocol required; final test sealed')

def load_config(path):
    value=yaml.safe_load(Path(path).read_text(encoding='utf-8'));validate_config(value);return value

def calibration_ids(mapping):
    videos=sorted({r['anonymous_video_id'] for r in mapping})
    chosen=np.random.Generator(np.random.PCG64(DEFAULT_CONFIG['seed'])).choice(videos,min(12,len(videos)),replace=False)
    return sorted(min((r for r in mapping if r['anonymous_video_id']==v),key=lambda r:r['decoded_frame_index'])['opaque_id'] for v in chosen)

def target_key(row):
    return row['anonymous_video_id']+'-'+row['opaque_id']

def target_template(mapping,sizes):
    return [dict(opaque_id=r['opaque_id'],target_id=target_key(r),image=f'images/{r["opaque_id"]}.png',
        width=str(sizes[r['opaque_id']][0]),height=str(sizes[r['opaque_id']][1]),
        calibration='yes' if r['opaque_id'] in calibration_ids(mapping) else 'no',**{k:'' for k in TARGET_FIELDS}) for r in mapping]

def validate_targets(rows,template,complete=False):
    if len(rows)!=len(template):raise ValueError('incomplete/duplicate target coverage')
    descriptions={}
    for r,t in zip(rows,template):
        if set(r)!=set(t) or any(r[k]!=v for k,v in t.items() if k not in TARGET_FIELDS):raise ValueError('target identity/template changed')
        if not r['target_status']:
            if complete or any(r[k] for k in TARGET_FIELDS):raise ValueError('incomplete target audit')
            continue
        if r['person_status'] not in ('absent','single','multiple','uncertain') or r['target_status'] not in ('selected','absent','ambiguous'):
            raise ValueError('invalid person/target status')
        if r['person_status']=='absent' and r['target_status']!='absent':raise ValueError('person/target absence inconsistent')
        if r['person_status']=='uncertain' and r['target_status']!='ambiguous':raise ValueError('uncertain person requires ambiguous target')
        if any(r[k] not in ('yes','no','uncertain') for k in ('shoulders_visible','hands_visible','waist_visible')):raise ValueError('explicit visibility audit required')
        if r['reviewer_origin'] not in ('human','ai') or not all(r[k].strip() for k in ('reviewer','reviewed_at','evidence','elapsed_seconds')):
            raise ValueError('target audit requires explicit reviewer provenance')
        try:
            time=datetime.fromisoformat(r['reviewed_at'].replace('Z','+00:00'));elapsed=float(r['elapsed_seconds'])
        except ValueError as exc:raise ValueError('invalid target audit time') from exc
        if time.tzinfo is None or not math.isfinite(elapsed) or elapsed<0:raise ValueError('timezone and finite nonnegative elapsed required')
        coords=[r[k] for k in ('x1','y1','x2','y2')]
        if r['target_status']=='selected':
            if not r['target_description'].strip():raise ValueError('selected target needs fixed identity description')
            if r['target_id'] in descriptions and descriptions[r['target_id']]!=r['target_description']:raise ValueError('target identity switched within video')
            descriptions[r['target_id']]=r['target_description']
            try:box=[int(x) for x in coords]
            except ValueError as exc:raise ValueError('integer target box required') from exc
            x1,y1,x2,y2=box
            if not (0<=x1<x2<=int(r['width']) and 0<=y1<y2<=int(r['height'])):raise ValueError('target box outside image/degenerate')
        elif any(coords):raise ValueError('absent/ambiguous target must not have fabricated ROI')
    return rows

def roi_bounds(row):
    if row['target_status']!='selected':return None
    x1,y1,x2,y2=[int(row[k]) for k in ('x1','y1','x2','y2')];px=.1*(x2-x1);py=.1*(y2-y1)
    return (max(0,math.floor(x1-px)),max(0,math.floor(y1-py)),min(int(row['width']),math.ceil(x2+px)),min(int(row['height']),math.ceil(y2+py)))

def eligible(row):
    return bool(row['target_status']=='selected' and int(row['y2'])-int(row['y1'])>=80
        and all(row[k]=='yes' for k in ('shoulders_visible','hands_visible','waist_visible')))

def roi_image(image,row):
    box=roi_bounds(row);return None if box is None else letterbox(image.crop(box))

def transformed_box(row,view):
    if row['target_status']!='selected':return None
    x1,y1,x2,y2=[int(row[k]) for k in ('x1','y1','x2','y2')];w,h=int(row['width']),int(row['height'])
    if view=='roi':
        a,b,c,d=roi_bounds(row);w,h=c-a,d-b;x1-=a;x2-=a;y1-=b;y2-=b
    if view=='native':
        nw,nh=(int(224*w/h),224) if w>=h else (224,int(224*h/w))
        ox,oy=int(round((nw-224)/2)),int(round((nh-224)/2))
        result=(x1*nw/w-ox,y1*nh/h-oy,x2*nw/w-ox,y2*nh/h-oy)
    else:
        scale=224/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));ox,oy=(224-nw)//2,(224-nh)//2
        result=(x1*nw/w+ox,y1*nh/h+oy,x2*nw/w+ox,y2*nh/h+oy)
    a,b,c,d=result
    if c<=0 or d<=0 or a>=224 or b>=224:return None
    return tuple(max(0,min(223,x)) for x in result)

def guide_image(image,row,view):
    result=image.copy();box=transformed_box(row,view)
    if box:ImageDraw.Draw(result).rectangle(box,outline=(0,255,0),width=2)
    return result

def concept_template(mapping,concepts,targets,view):
    target={r['opaque_id']:r for r in targets}
    return [dict(opaque_id=r['opaque_id'],target_id=target_key(r),view=view,concept_id=c['concept_id'],
        image=f'images/{r["opaque_id"]}.png' if view!='roi' or target[r['opaque_id']]['target_status']=='selected' else '',
        guide=f'guides/{r["opaque_id"]}.png' if view!='roi' or target[r['opaque_id']]['target_status']=='selected' else '',
        target_status=target[r['opaque_id']]['target_status'],
        target_in_view='yes' if transformed_box(target[r['opaque_id']],view) is not None else 'no',
        target_description=target[r['opaque_id']]['target_description'],
        concept_label=c['concept_label'],description_en=c['description_en'],operational_rule=RULES[c['concept_id']],
        **{k:'' for k in HUMAN_FIELDS}) for r in mapping for c in concepts]

def validate_concepts(rows,template,complete=False):
    validate_form(rows,template,complete=complete)
    for r in rows:
        if (r['target_status']!='selected' or r['target_in_view']=='no') and r['judgment'] and r['judgment'].strip()!='uncertain':
            raise ValueError('no fixed target/ROI cannot become a known pose label')
    return rows
