import csv
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageOps
from hmdb51_e0.provenance import read_json,write_json,write_csv,sha256_file

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

@pytest.fixture
def source(tmp_path,monkeypatch):
    from hmdb51_e0 import e20r2_workflow as w
    from hmdb51_e0.e11_workflow import tree_hashes
    parent=tmp_path/'parent';(parent/'private/images/original').mkdir(parents=True)
    (parent/'e11').mkdir();(parent/'audit').mkdir()
    mapping=[]
    for i in range(6):
        oid=f'F{i+1:04d}';Image.new('RGB',(400,200),(30+i,60,90)).save(parent/'private/images/original'/f'{oid}.png')
        mapping.append(dict(opaque_id=oid,anonymous_video_id=f'V{i//3+1:03d}',video_id=f'clap/{i//3}.avi',
            source_group_id=f'g{i//3}',decoded_frame_index=i%3,frame_position_zero_based=i%3))
    from hmdb51_e0.e20r2_protocol import CONCEPTS
    data=dict(parent=str(parent),parent_hashes=tree_hashes(parent),mapping=mapping,
        concepts=[dict(concept_id=c,concept_label=c,description_en='test '+c) for c in CONCEPTS],
        text_embeddings=np.eye(6,dtype='float32').tolist(),encoder=dict(synthetic_test_fixture=True,dimension=6),
        selected_videos=[dict(video_id='clap/0.avi'),dict(video_id='clap/1.avi')],
        feature_condition='validation_only_unverified_features',e11_batch=str(parent/'e11'),audit_directory=str(parent/'audit'))
    monkeypatch.setattr(w,'parent_context',lambda p:data)
    def verify(s):
        if tree_hashes(s['parent'])!=s['parent_hashes']:raise ValueError('parent changed')
    monkeypatch.setattr(w,'verify_parent',verify)
    monkeypatch.setattr(w,'native_crop',lambda rgb,identity:ImageOps.fit(Image.fromarray(rgb),(224,224)))
    paths=dict(dataset_root=tmp_path/'raw',feature_cache_root=tmp_path/'cache',run_root=tmp_path/'runs',report_root=tmp_path/'reports')
    return w,parent,paths

class Encoder:
    identity=dict(synthetic_test_fixture=True,dimension=6)
    def encode_images(self,images):return np.tile(np.ones(6,dtype='float32')/np.sqrt(6),(len(images),1)).astype('float32')
    def verify_frozen(self):return dict(unchanged=True,before='fixture',after='fixture')

def fill_targets(packet):
    path=packet/'targets/targets.csv';part=rows(path)
    for i,r in enumerate(part):
        r.update(person_status='single',target_status='selected',x1='100',y1='20',x2='180',y2='180',
            shoulders_visible='yes',hands_visible='yes',waist_visible='yes',target_description='same fixed person',
            reviewer='AI_target',reviewer_origin='ai',reviewed_at='2026-11-24T22:00:00+08:00',evidence='synthetic target fixture',elapsed_seconds='0')
        if i==5:r.update(person_status='absent',target_status='absent',x1='',y1='',x2='',y2='',shoulders_visible='no',hands_visible='no',waist_visible='no')
    write_csv(path,part,list(part[0]))

def fill_concepts(path,reviewer='AI_primary'):
    part=rows(path)
    for r in part:
        known=r['target_in_view']=='yes'
        r.update(judgment='absent' if known else 'uncertain',reason_code='clear' if known else 'other',reviewer=reviewer,
            reviewer_origin='ai',reviewed_at='2026-11-24T22:00:00+08:00',evidence='explicit synthetic visible evidence',elapsed_seconds='0')
    if part:write_csv(path,part,list(part[0]))

def test_end_to_end_keeps_missing_roi_in_denominator_and_masks_features(source,tmp_path):
    w,parent,paths=source;p=w.prepare(parent,paths,tmp_path/'packets',require_official=False)
    with pytest.raises(ValueError,match='incomplete'):w.freeze(p,'targets')
    fill_targets(p);w.freeze(p,'targets')
    assert not (p/'private/views/roi/images/F0006.png').exists()
    w.encode(p,paths,encoder=Encoder(),require_official=False)
    for view in ('native','letterbox','roi'):
        fill_concepts(p/view/'judgments.csv');w.freeze(p,view)
    c=w.load_packet(p);assert len(c['scores'])==108
    assert sum(r['status']=='no_roi' for r in c['scores'])==6
    with np.load(p/'private/encoding/features.npz') as b:
        assert b['valid'].sum()==17 and np.isnan(b['Q'][2,5]).all()
    from hmdb51_e0.e20r2_reporting import summarize
    r=summarize(p,tmp_path/'report')
    assert r['analysis_ready'] and not r['independent_human_review_completed'] and not r['exploratory_continuation_gate']
    assert r['n_frames']==6 and r['n_roi_available']==5
    assert len(r['concept_alignment'])==18 and len(r['paired_inputs'])==18
    assert all(x['n_rows']==6 for x in r['concept_coverage'])
    for view in ('review-native','review-letterbox','review-roi'):
        fill_concepts(p/view/'judgments.csv',reviewer='AI_second')
        if view=='review-native':
            part=rows(p/view/'judgments.csv');known=next(x for x in part if x['judgment']=='absent');known['judgment']='present';write_csv(p/view/'judgments.csv',part,list(part[0]))
        w.freeze(p,view)
    assert len(rows(p/'adjudication/judgments.csv'))==1
    with pytest.raises(ValueError,match='incomplete'):w.freeze(p,'adjudication')
    fill_concepts(p/'adjudication/judgments.csv',reviewer='AI_judge');w.freeze(p,'adjudication')
    final=summarize(p,tmp_path/'report-final');assert final['review_completed'] and not final['independent_human_review_completed']

@pytest.mark.parametrize('kind',['pixels','score','parent','same_reviewer','wrong_order','missing_known'])
def test_roi_integrity_and_protocol_guards(source,tmp_path,kind):
    w,parent,paths=source;p=w.prepare(parent,paths,tmp_path/'packets',require_official=False)
    fill_targets(p);w.freeze(p,'targets')
    if kind=='wrong_order':
        with pytest.raises(ValueError,match='currently'):w.freeze(p,'roi')
    elif kind=='missing_known':
        part=rows(p/'native/judgments.csv');part[-1].update(judgment='absent',reason_code='clear',reviewer='a',reviewer_origin='ai',
            reviewed_at='2026-11-24T22:00:00+08:00',evidence='bad fixture',elapsed_seconds='0');write_csv(p/'native/judgments.csv',part,list(part[0]))
        with pytest.raises(ValueError,match='fixed target'):w.load_packet(p)
    elif kind=='parent':
        (parent/'changed.txt').write_text('changed')
        with pytest.raises(ValueError,match='parent changed'):w.load_packet(p)
    elif kind=='same_reviewer':
        for v in ('native','letterbox','roi'):fill_concepts(p/v/'judgments.csv');w.freeze(p,v)
        fill_concepts(p/'review-native/judgments.csv')
        with pytest.raises(ValueError,match='independent'):w.freeze(p,'review-native')
    else:
        if kind=='pixels':
            path=p/'private/views/roi/images/F0001.png';Image.new('RGB',(224,224),'red').save(path)
        else:
            w.encode(p,paths,encoder=Encoder(),require_official=False)
            path=p/'private/encoding/model_scores.csv';part=rows(path);part[0]['cosine_score']='.123';write_csv(path,part,list(part[0]))
        m=read_json(p/'input_manifest.json');m['immutable'][path.relative_to(p).as_posix()]=sha256_file(path);write_json(p/'input_manifest.json',m)
        with pytest.raises(ValueError,match='reconstruction'):w.load_packet(p)

def test_production_guard_before_creating_output(source,tmp_path):
    w,parent,paths=source
    with pytest.raises(ValueError,match='official'):w.prepare(parent,paths,tmp_path/'packets')
    assert not (tmp_path/'packets').exists()

def test_cli_requires_official_features_and_seals_final_test(source,tmp_path,capsys):
    from hmdb51_e0.e20r2_cli import main,build_parser
    import yaml
    w,parent,paths=source;local=tmp_path/'paths.yaml'
    local.write_text(yaml.safe_dump({k:str(v) for k,v in dict(paths,annotations_root=tmp_path/'ann',weights_file=tmp_path/'weights').items()}))
    assert main(['check','--parent',str(parent),'--paths',str(local)])==2
    assert 'official' in capsys.readouterr().err
    with pytest.raises(SystemExit):build_parser().parse_args(['prepare','--parent','x','--stage','final_test'])
