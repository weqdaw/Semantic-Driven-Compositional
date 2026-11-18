"""Explicit frame audit export; no automatic concept-truth annotation."""
from __future__ import annotations

from pathlib import Path
import html
import importlib.metadata
import shutil
import numpy as np
from PIL import Image

from .config import safe_video_path
from .e11_features import validate_config
from .e11_workflow import parent_context, tree_hashes
from .provenance import digest, new_directory, now, read_json, sha256_file, write_csv, write_json
from .round2_io import safe_output
from .video_io import decoder_identity, decode_video, midpoint_indices


def select_videos(ids: list[str], config: dict) -> list[str]:
    validate_config(config)
    ordered=sorted(ids); audit=config['audit']
    if len(set(ordered))!=len(ordered) or len(ordered)<audit['videos']:
        raise ValueError('audit requires at least 10 unique development videos')
    selected=np.random.Generator(np.random.PCG64(audit['seed'])).choice(len(ordered),audit['videos'],replace=False)
    return [ordered[i] for i in sorted(selected)]


def native_crop(rgb: np.ndarray, encoder: dict) -> Image.Image:
    from torchvision.transforms import Resize, CenterCrop, InterpolationMode
    prefix=[{'antialias':True,'interpolation':'InterpolationMode.BICUBIC','max_size':None,'size':224,
             'type':'torchvision.transforms.transforms.Resize'},
            {'size':[224,224],'type':'torchvision.transforms.transforms.CenterCrop'},
            {'function':'clip.clip._convert_image_to_rgb'}]
    if encoder.get('preprocess',[])[:3]!=prefix:
        raise ValueError('unsupported native preprocessing/crop identity')
    for name in ('pillow','torchvision'):
        if encoder.get(name)!=importlib.metadata.version(name):
            raise ValueError(f'{name} version changed since frozen image preprocessing')
    return CenterCrop(224)(Resize(224,interpolation=InterpolationMode.BICUBIC)(Image.fromarray(rgb))).convert('RGB')


def export_audit(parent: Path, config: dict, paths: dict, output: Path) -> Path:
    validate_config(config); parent=Path(parent)
    before=tree_hashes(parent)
    batch,_,_,index,bank,bundle,_=parent_context(parent)
    if bundle['parent_manifest']['decoder']!=decoder_identity():
        raise ValueError('decoder changed since frozen audit')
    selected=select_videos(batch['evaluation_ids'],config)
    by_id={r['video_id']:r for r in bundle['rows']}
    entries={r['video_id']:r for r in index['entries']}
    positions={r['video_id']:i for i,r in enumerate(index['entries'])}
    catalog=read_json(parent/'concept_catalog.json')
    concepts={r['id']:r for r in catalog['concepts']}
    concept_positions={r['id']:i for i,r in enumerate(catalog['concepts'])}
    # Verify all sources before decoding or creating an output directory.
    for vid in selected:
        row,entry=by_id[vid],entries[vid]
        path=safe_video_path(paths['dataset_root'],row['relative_path'])
        if sha256_file(path)!=row['sha256'] or row['sha256']!=entry['video_sha256']:
            raise ValueError('audit video hash changed or source alias differs')
        if entry['frame_indices']!=midpoint_indices(row['frame_count']).tolist():
            raise ValueError('audit frame indices differ from frozen midpoint sampling')
    # Validate preprocessing before any potentially large export.
    native_crop(np.zeros((224,224,3),dtype=np.uint8),index['identity']['encoder'])
    output=safe_output(output,paths,protected=(parent,))
    folder=new_directory(output,'e11-frame-audit'); (folder/'images').mkdir()
    judgments=[]; scores=[]; image_rows=[]
    for vid in selected:
        row,entry=by_id[vid],entries[vid]
        offsets=config['audit']['frame_positions_zero_based']
        wanted=[entry['frame_indices'][p] for p in offsets]
        path=safe_video_path(paths['dataset_root'],row['relative_path'])
        metadata,frames=decode_video(path,wanted)
        if metadata['frame_count']!=row['frame_count'] or sha256_file(path)!=entry['video_sha256']:
            raise ValueError('audit video changed during decoding')
        for offset,frame_index,rgb in zip(offsets,wanted,frames):
            image_id=digest([vid,frame_index])[:20]
            original=f'images/{image_id}-original.png'; cropped=f'images/{image_id}-clip.png'
            Image.fromarray(rgb).save(folder/original)
            native_crop(rgb,index['identity']['encoder']).save(folder/cropped)
            image_rows.append(dict(video_id=vid,video_sha256=entry['video_sha256'],frame_position_zero_based=offset,
                                   frame_index=frame_index,original_image=original,clip_image=cropped))
            for cid in config['audit']['concept_ids']:
                concept=concepts[cid]
                item=dict(video_id=vid,frame_position_zero_based=offset,frame_index=frame_index,concept_id=cid,
                          concept_label=concept['label_zh'],description_en=concept['description_en'],
                          visibility_note=concept.get('visibility_note',''),original_image=original,clip_image=cropped)
                judgments.append(dict(**item,judgment='',reviewer='',reviewed_at='',evidence=''))
                scores.append(dict(video_id=vid,frame_position_zero_based=offset,frame_index=frame_index,concept_id=cid,
                                   cosine_score=float(bank['C'][positions[vid],offset,concept_positions[cid]])))
    if tree_hashes(parent)!=before: raise ValueError('E1 evidence changed during audit export')
    write_csv(folder/'judgments.csv',judgments,list(judgments[0]))
    shutil.copyfile(folder/'judgments.csv',folder/'judgments.template.csv')
    write_csv(folder/'model_scores.after_review.csv',scores,list(scores[0]))
    write_csv(folder/'frames.csv',image_rows,list(image_rows[0]))
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>E1.1 Frame Audit</title>',
          '<style>body{font-family:system-ui;margin:24px}td{padding:12px;border-bottom:1px solid #ddd}img{max-width:320px;max-height:224px}</style>',
          '<h1>E1.1 概念画面审核</h1><p>先在 judgments.csv 填写人工判断，再查看 model_scores.after_review.csv。</p>',
          '<p>以 CLIP 裁剪图为判断依据，原图作上下文参考；文件名可能暴露类别，因此不是严格盲审。</p>',
          '<table><tr><th>视频与帧</th><th>原图</th><th>CLIP 裁剪图</th></tr>']
    for r in image_rows:
        body.append(f'<tr><td>{html.escape(r["video_id"])}<br>缓存位置 {r["frame_position_zero_based"]}; 原视频帧 {r["frame_index"]}</td>'
                    f'<td><img src="{r["original_image"]}"></td><td><img src="{r["clip_image"]}"></td></tr>')
    (folder/'index.html').write_text('\n'.join(body+['</table></html>']),encoding='utf-8')
    (folder/'README.md').write_text(
        '# E1.1 人工审核\n\n在 judgments.csv 中将 judgment 填为 present / absent / uncertain，填写 reviewer、reviewed_at、evidence。'
        '判断对象是 CLIP 裁剪图中可见的姿态，不用动作类别推断姿态。遮挡、模糊、身份混杂、裁剪缺失均可记 uncertain 并说明。'
        '不确定行单独统计，不能默认视作 absent。\n\n打开 index.html 逐帧核对。10 段 development 视频 × 3 帧 × 6 概念 = 180 行。'
        '本抽样不保证类别均衡。不要先查看 model_scores.after_review.csv；文件名仍可能暴露类别，因此不宣称严格盲审。'
        'judgments.template.csv 是空表快照，勿覆盖。建议审核后另存版本并计算校验和。'
        '本工具只导出材料，不宣称审核已完成，也不将余弦分数解释为概率。\n',encoding='utf-8')
    write_json(folder/'audit_manifest.json',dict(created_at=now(),protocol_id=batch['protocol_id'],
               feature_condition=batch['feature_condition'],config=config,selected_video_ids=selected,
               parent_evidence_sha256=digest(before),human_review_completed=False,n_judgments=len(judgments),
               files=tree_hashes(folder)))
    return folder
