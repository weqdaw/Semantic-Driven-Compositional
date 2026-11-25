import copy
import csv
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageOps
from hmdb51_e0.provenance import digest,write_json,write_csv,read_json,sha256_file

def readrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

@pytest.fixture
def source(tmp_path,monkeypatch):
    from hmdb51_e0 import e20r_workflow as w
    from hmdb51_e0.e20r_protocol import CONCEPTS
    from hmdb51_e0.e11_workflow import tree_hashes
    from hmdb51_e0.video_io import decoder_identity
    names=('clap','jump','run','walk','wave');root=tmp_path/'videos';parent=tmp_path/'parent';audit=tmp_path/'oldaudit'
    parent.mkdir();audit.mkdir();write_json(parent/'batch.json',{'synthetic':True});write_json(audit/'manifest.json',{'synthetic':True})
    rows=[];entries={}
    for c,name in enumerate(names):
        for i in range(12):
            vid=f'{name}/{i}.avi';path=root/vid;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(vid.encode())
            rows.append(dict(video_id=vid,relative_path=vid,class_id=c,class_name=name,source_group_id=f'g{c}-{i//2}',
                experimental_role='support_pool',official_tag=1,frame_count=16,sha256=sha256_file(path)))
            entries[vid]=dict(video_id=vid,frame_indices=list(range(16)))
    context=dict(e11_batch=parent,audit_directory=audit,protocol_id='fixture',feature_condition='validation_only_unverified_features',
        rows=rows,entries=entries,excluded_ids=[],excluded_groups=[],encoder={'synthetic_test_fixture':True,'dimension':6},decoder=decoder_identity(),
        concepts=[dict(concept_id=c,concept_label='姿态'+c,description_en='a pose '+c) for c in CONCEPTS],text_embeddings=np.eye(6,dtype='f4'))
    def source_context(p,a):
        assert Path(p).resolve()==parent.resolve() and Path(a).resolve()==audit.resolve()
        return dict(context,batch_hashes=tree_hashes(parent),audit_hashes=tree_hashes(audit))
    monkeypatch.setattr(w,'source_context',source_context)
    monkeypatch.setattr(w,'native_crop',lambda rgb,identity:ImageOps.fit(Image.fromarray(rgb),(224,224)))
    def decode(path,indices):
        return {'frame_count':16},[np.full((100,200,3),20+index,dtype='uint8') for index in indices]
    monkeypatch.setattr(w,'decode_video',decode)
    paths=dict(dataset_root=root,feature_cache_root=tmp_path/'oldcache',run_root=tmp_path/'runs',report_root=tmp_path/'reports')
    return context,paths

class Encoder:
    identity={'synthetic_test_fixture':True,'dimension':6}
    def encode_images(self,images):
        q=np.tile(np.array([1.,2.,3.,4.,5.,6.]),(len(images),1));q/=np.linalg.norm(q,axis=1,keepdims=True);return q.astype('f4')
    def verify_frozen(self):return {'unchanged':True,'before':'fixture','after':'fixture'}

def fill(path,reviewer,origin='ai'):
    rows=readrows(path)
    for i,r in enumerate(rows):r.update(judgment='present' if i%2 else 'absent',reason_code='clear',reviewer=reviewer,
        reviewer_origin=origin,reviewed_at='2026-11-24T12:00:00+08:00',evidence='explicit synthetic test label',elapsed_seconds='0')
    if rows:write_csv(path,rows,list(rows[0]))

def test_complete_plan_prepare_encoding_reviews_and_ai_gate(source,tmp_path):
    from hmdb51_e0.e20r_protocol import DEFAULT_CONFIG,VIEWS
    from hmdb51_e0.e20r_workflow import plan_e20r,prepare_e20r,encode_e20r,freeze_view,load_packet
    from hmdb51_e0.e20r_reporting import summarize
    c,paths=source
    plan=plan_e20r(c['e11_batch'],c['audit_directory'],DEFAULT_CONFIG,paths,tmp_path/'plans')
    packet=prepare_e20r(plan,paths,tmp_path/'packets')
    assert len(readrows(packet/'native/judgments.csv'))==360
    public=(packet/'native/judgments.csv').read_text(encoding='utf-8')
    assert '.avi' not in public and 'source_group' not in public and 'score' not in public
    assert not (packet/'letterbox').exists()
    progress=summarize(packet,tmp_path/'progress')
    assert not progress['analysis_ready'] and progress['concept_alignment']==[]
    encode_e20r(packet,paths,encoder=Encoder())
    for view in VIEWS:
        p=packet/view/'judgments.csv'
        if view!='adjudication':
            fill(p,'primary' if not view.startswith('review-') else 'secondary')
            if view=='review-native':
                rows=readrows(p);rows[0]['judgment']='present' if rows[0]['judgment']=='absent' else 'absent'
                write_csv(p,rows,list(rows[0]))
        else:
            rows=readrows(p)
            assert len(rows)==1 and rows[0]['first_origin']==rows[0]['second_origin']=='ai'
            with pytest.raises(ValueError,match='incomplete'):freeze_view(packet,view)
            fill(p,'adjudicator')
        freeze_view(packet,view)
    result=summarize(packet,tmp_path/'report')
    assert result['analysis_ready'] and result['review_completed'] and not result['independent_human_review_completed']
    assert result['exploratory_continuation_gate'] is False
    assert len(result['concept_alignment'])==12 and len(result['paired_inputs'])==6 and len(result['concept_coverage'])==18
    assert result['cohort']=='representative' and (tmp_path/'report/e20r-results.md').exists()
    assert '仅程序验证' in (tmp_path/'report/e20r-results.md').read_text(encoding='utf-8')
    p=packet/'private/encoding/model_scores.csv';rows=readrows(p);rows[0]['cosine_score']='.123456';write_csv(p,rows,list(rows[0]))
    manifest=read_json(packet/'input_manifest.json');manifest['immutable']['private/encoding/model_scores.csv']=sha256_file(p);write_json(packet/'input_manifest.json',manifest)
    with pytest.raises(ValueError,match='score|forward'):load_packet(packet)

def test_supplement_has_separate_cohort_no_imported_target_labels_and_source_exclusions(source,tmp_path):
    from hmdb51_e0.e20r_protocol import DEFAULT_CONFIG,SELECTION_FIELDS
    from hmdb51_e0.e20r_workflow import plan_e20r,prepare_e20r
    c,paths=source;base=plan_e20r(c['e11_batch'],c['audit_directory'],DEFAULT_CONFIG,paths,tmp_path/'plans')
    data=read_json(base);used={r['source_group_id'] for r in data['selected_videos']}
    v=next(r for r in c['rows'] if r['source_group_id'] not in used)
    selection=tmp_path/'select.csv';item=dict(video_id=v['video_id'],target_concept_id='pose03',target_label='present',
        selection_reason='visible evidence fixture; no model scores',selector='fixture',selection_origin='ai')
    write_csv(selection,[item],list(SELECTION_FIELDS))
    plan=plan_e20r(c['e11_batch'],c['audit_directory'],DEFAULT_CONFIG,paths,tmp_path/'plans',selection_path=selection,base_plan=base)
    assert read_json(plan)['cohort']=='coverage_supplement'
    packet=prepare_e20r(plan,paths,tmp_path/'packets');labels=readrows(packet/'native/judgments.csv')
    assert len(labels)==18 and all(not r['judgment'] and not r['reviewer_origin'] for r in labels)
    item['video_id']=data['selected_videos'][0]['video_id'];write_csv(selection,[item],list(SELECTION_FIELDS))
    with pytest.raises(ValueError,match='overlap|excluded'):plan_e20r(c['e11_batch'],c['audit_directory'],DEFAULT_CONFIG,paths,tmp_path/'plans',selection_path=selection,base_plan=base)

def test_production_cli_rejects_synthetic_and_test_stage_before_outputs(source,tmp_path,capsys):
    import yaml
    from hmdb51_e0.e20r_cli import main,build_parser
    c,paths=source;local=tmp_path/'paths.yaml'
    local.write_text(yaml.safe_dump(dict(dataset_root=str(paths['dataset_root']),feature_cache_root=str(paths['feature_cache_root']),
        run_root=str(paths['run_root']),report_root=str(paths['report_root']),annotations_root=None,weights_file=None)))
    assert main(['check','--paths',str(local),'--e11-batch',str(c['e11_batch']),'--audit-directory',str(c['audit_directory'])])==2
    assert 'official' in capsys.readouterr().err
    with pytest.raises(SystemExit):build_parser().parse_args(['prepare','--plan','x','--stage','final_test'])
    assert not paths['report_root'].exists()


@pytest.mark.parametrize('kind',['mapping','pixels','source','incomplete','freeze_order','same_reviewer'])
def test_repair_evidence_and_review_guards(source,tmp_path,kind):
    from hmdb51_e0.e20r_protocol import DEFAULT_CONFIG
    from hmdb51_e0.e20r_workflow import plan_e20r,prepare_e20r,load_packet,freeze_view
    c,paths=source
    plan=plan_e20r(c['e11_batch'],c['audit_directory'],DEFAULT_CONFIG,paths,tmp_path/'plans')
    packet=prepare_e20r(plan,paths,tmp_path/'packets')
    if kind=='source':
        write_json(c['e11_batch']/'batch.json',{'changed':True})
        with pytest.raises(ValueError,match='source identity'):load_packet(packet)
        return
    if kind in ('mapping','pixels'):
        manifest=read_json(packet/'input_manifest.json')
        if kind=='mapping':
            path=packet/'private/mapping.json';rows=read_json(path);rows[0]['video_id']='clap/tampered.avi';write_json(path,rows)
        else:
            path=next((packet/'private/images/letterbox').glob('*.png'));Image.new('RGB',(224,224),(0,0,0)).save(path)
        manifest['immutable'][path.relative_to(packet).as_posix()]=sha256_file(path);write_json(packet/'input_manifest.json',manifest)
        with pytest.raises(ValueError,match='reconstruction'):load_packet(packet)
        return
    if kind=='incomplete':
        with pytest.raises(ValueError,match='incomplete'):freeze_view(packet,'native')
    elif kind=='freeze_order':
        with pytest.raises(ValueError,match='currently available'):freeze_view(packet,'letterbox')
    else:
        for view in ('native','letterbox','original'):
            fill(packet/view/'judgments.csv','one');freeze_view(packet,view)
        fill(packet/'review-native/judgments.csv','one')
        with pytest.raises(ValueError,match='independent reviewer'):freeze_view(packet,'review-native')
