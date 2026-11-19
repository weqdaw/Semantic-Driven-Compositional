import csv
import copy
import numpy as np
import pytest
from hmdb51_e0.provenance import digest,write_json,write_csv,sha256_file
from hmdb51_e0.e11_features import DEFAULT_CONFIG as C11
from hmdb51_e0.e11_workflow import tree_hashes


def test_audit_keeps_blank_reviews_incomplete_and_scores_only_verified_human_rows(tmp_path,monkeypatch):
    from hmdb51_e0 import e12_audit as audit
    parent=tmp_path/'e11'; (parent/'parent').mkdir(parents=True)
    ids=[f'walk/{j}.avi' for j in range(10)]; concepts=C11['audit']['concept_ids']
    write_json(parent/'parent/concept_catalog.json',{'concepts':[{'id':c} for c in concepts]})
    c=np.zeros((10,16,6),dtype='f4'); c[:,2]=.8; c[:,7]=.2; c[:,13]=.5
    np.savez_compressed(parent/'parent/feature_bank.npz',C=c)
    context={'batch':{'evaluation_ids':ids,'protocol_id':'fixture'},'index':{'entries':[{'video_id':v,'video_sha256':digest(v),'frame_indices':list(range(16))} for v in ids]}}
    monkeypatch.setattr(audit,'parent_context',lambda p:context)
    packet=tmp_path/'audit'; packet.mkdir()
    rows=[];scores=[];frames=[]; (packet/'images').mkdir()
    for j,vid in enumerate(ids):
        for pos in (2,7,13):
            image_id=digest([vid,pos])[:20]
            original=f'images/{image_id}-original.png'; crop=f'images/{image_id}-clip.png'
            (packet/original).write_bytes(b'original_fixture'); (packet/crop).write_bytes(b'crop_fixture')
            frames.append(dict(video_id=vid,video_sha256=digest(vid),frame_position_zero_based=pos,frame_index=pos,original_image=original,clip_image=crop))
            for k,cid in enumerate(concepts):
                rows.append(dict(video_id=vid,frame_position_zero_based=pos,frame_index=pos,concept_id=cid,
                                 concept_label=cid,description_en='a pose',visibility_note='',original_image=original,clip_image=crop,
                                 judgment='',reviewer='',reviewed_at='',evidence=''))
                scores.append(dict(video_id=vid,frame_position_zero_based=pos,frame_index=pos,concept_id=cid,cosine_score=float(c[j,pos,k])))
    write_csv(packet/'judgments.csv',rows,list(rows[0])); write_csv(packet/'judgments.template.csv',rows,list(rows[0]))
    write_csv(packet/'model_scores.after_review.csv',scores,list(scores[0])); write_csv(packet/'frames.csv',frames,list(frames[0]))
    write_json(packet/'audit_manifest.json',dict(protocol_id='fixture',config=C11,selected_video_ids=ids,
               parent_evidence_sha256=digest(tree_hashes(parent/'parent')),human_review_completed=False,n_judgments=180,files=tree_hashes(packet)))
    result=audit.analyze_audit(packet,parent)
    assert result['human_review_completed'] is False and result['n_blank']==180
    assert all(r['auroc'] is None for r in result['concepts'])
    for r in rows:
        r.update(judgment={2:'present',7:'absent',13:'uncertain'}[r['frame_position_zero_based']],reviewer='human_fixture',
                 reviewed_at='2026-11-18T22:00:00+08:00',evidence='visible pose fixture')
    write_csv(packet/'judgments.csv',rows,list(rows[0]))
    result=audit.analyze_audit(packet,parent)
    assert result['human_review_completed'] is True and result['n_uncertain']==60
    assert all(r['n_present']==10 and r['n_absent']==10 and r['auroc']==1. for r in result['concepts'])
    audit.write_audit_report(result,tmp_path/'report')
    assert (tmp_path/'report/audit-summary.md').is_file()
    # A packet must not turn into an audit without images by omitting hashes.
    from hmdb51_e0.provenance import read_json
    manifest=read_json(packet/'audit_manifest.json'); saved_manifest=copy.deepcopy(manifest)
    image=packet/rows[0]['clip_image']; original_bytes=image.read_bytes()
    del manifest['files'][rows[0]['clip_image']]; image.unlink(); write_json(packet/'audit_manifest.json',manifest)
    with pytest.raises(ValueError,match='image|frame|packet'):
        audit.analyze_audit(packet,parent)
    image.write_bytes(original_bytes); write_json(packet/'audit_manifest.json',saved_manifest)
    # A filled template plus updated hashes cannot stand in for the frozen blank.
    original_template=(packet/'judgments.template.csv').read_bytes()
    write_csv(packet/'judgments.template.csv',rows,list(rows[0]))
    manifest=copy.deepcopy(saved_manifest); manifest['files']['judgments.template.csv']=sha256_file(packet/'judgments.template.csv')
    write_json(packet/'audit_manifest.json',manifest)
    with pytest.raises(ValueError,match='blank|template'):
        audit.analyze_audit(packet,parent)
    manifest['files']['judgments.csv']=manifest['files']['judgments.template.csv']
    write_json(packet/'audit_manifest.json',manifest)
    with pytest.raises(ValueError,match='blank|template'):
        audit.analyze_audit(packet,parent)
    (packet/'judgments.template.csv').write_bytes(original_template); write_json(packet/'audit_manifest.json',saved_manifest)
    rows[0]['frame_index']=9; write_csv(packet/'judgments.csv',rows,list(rows[0]))
    with pytest.raises(ValueError,match='row|template|frame'): audit.analyze_audit(packet,parent)
