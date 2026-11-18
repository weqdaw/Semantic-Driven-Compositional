import copy
import csv
import numpy as np
import pytest
import yaml

from hmdb51_e0.config import ROOT
from hmdb51_e0.provenance import read_json,write_json,sha256_file
from test_round2_protocol import config2,parent
from test_round2_features import round2_inputs
from test_ablation_experiment import inputs


class SyntheticText:
    identity={'synthetic_test_fixture':True,'dimension':6}
    def encode_text(self,prompts):
        values=np.random.default_rng(928).normal(size=(len(prompts),6))
        return (values/np.linalg.norm(values,axis=1,keepdims=True)).astype(np.float32)
    def verify_frozen(self): return {'unchanged':True}


@pytest.fixture
def e1_data(inputs,tmp_path):
    from hmdb51_e0.e1_features import DEFAULT_CONFIG,load_catalog
    return dict(**inputs,e1_config=copy.deepcopy(DEFAULT_CONFIG),
                catalog_path=ROOT/'configs/e1.concepts.v1.yaml',catalog=load_catalog(ROOT/'configs/e1.concepts.v1.yaml'),
                e1_root=tmp_path/'e1_features')


def prepare(data):
    from hmdb51_e0.e1_experiment import prepare_e1_features
    return prepare_e1_features(data['reference'],data['e1_config'],data['catalog_path'],data['paths'],
                               data['output'],data['e1_root'],text_encoder=SyntheticText())


def test_e1_cache_has_matched_dimensions_and_rejects_rehashed_mismatched_vectors(e1_data):
    from hmdb51_e0.e1_experiment import load_e1_features
    from hmdb51_e0.e1_features import concept_pool,random_directions
    before=sha256_file(e1_data['index'])
    index_path=prepare(e1_data)
    index,bank=load_e1_features(index_path,e1_data['reference'])
    assert index['feature_condition']=='validation_only_unverified_features'
    assert bank['C'].shape==(80,16,16) and bank['E'].shape==(16,6)
    assert bank['C0'].shape==(80,32) and bank['C1'].shape==(80,80)
    assert bank['R0'].shape==(5,80,32) and bank['R1'].shape==(5,80,80)
    assert sha256_file(e1_data['index'])==before
    np.testing.assert_array_equal(bank['C0'][0],concept_pool(bank['C'][0],'C0'))
    for j,seed in enumerate(index['identity']['config']['random_projection']['seeds']):
        np.testing.assert_array_equal(bank['P'][j],random_directions(6,16,seed))
    audit=list(csv.DictReader((index_path.parent/'concept_audit_template.csv').open(encoding='utf-8',newline='')))
    assert len(audit)==min(30,len(read_json(e1_data['reference']/'batch.json')['evaluation_ids']))*16
    assert all(r['judgment']=='' and 'score' not in r and 'true_class_id' not in r for r in audit)
    file=index_path.parent/'features.npz'
    bank['C0'][0]=bank['C0'][1]
    np.savez_compressed(file,**bank)
    index['npz_sha256']=sha256_file(file)
    write_json(index_path,index)
    with pytest.raises(ValueError,match='pool|concept'):
        load_e1_features(index_path,e1_data['reference'])


def test_e1_full_workflow_pairs_360_new_runs_and_two_level_random_controls(e1_data,tmp_path,monkeypatch):
    from hmdb51_e0.e1_experiment import run_e1
    from hmdb51_e0.e1_reporting import summarize_e1,validate_e1_runs
    from matplotlib.axes import Axes
    index=prepare(e1_data)
    folder=run_e1(e1_data['reference'],index,e1_data['paths'])
    batch=read_json(folder/'batch.json')
    assert batch['status']=='success' and len(batch['runs'])==360
    assert sum(r['representation'] in ('C0','C1') for r in batch['runs'])==60
    assert all(r['projection_seed'] is None for r in batch['runs'] if r['representation'].startswith('C'))
    assert all((folder/r['directory']/'provenance.json').is_file() for r in batch['runs'])
    titles=[]; original=Axes.set_title
    def track(self,label,*args,**kwargs):
        if label: titles.append(label)
        return original(self,label,*args,**kwargs)
    monkeypatch.setattr(Axes,'set_title',track)
    result=summarize_e1(folder,tmp_path/'e1_report')
    assert len(result['per_run_metrics'])==420 and len(result['per_support_metrics'])==180
    assert len(result['summary'])==18 and len(result['paired_deltas'])==150
    assert len(result['time_increment_gaps'])==30
    assert len(result['summary_by_projection_seed'])==30
    assert titles and all('PROGRAM VALIDATION' in title for title in titles)
    assert '仅程序验证' in (tmp_path/'e1_report/e1-results.md').read_text(encoding='utf-8')
    assert len({r['macro_f1'] for r in result['per_run_metrics']})>10
    indexed={(r['representation'],r['k'],r['support_seed']):r for r in result['per_support_metrics']}
    for row in result['summary']:
        assert row['n_support_sets']==10
        if row['representation'] not in ('R0','R1'): continue
        values=[np.mean([r['macro_f1'] for r in result['per_run_metrics'] if r['representation']==row['representation']
                        and r['k']==row['k'] and r['support_seed']==seed]) for seed in range(10)]
        assert row['macro_f1_mean']==pytest.approx(np.mean(values))
        assert row['macro_f1_std']==pytest.approx(np.std(values,ddof=1))
    for row in result['paired_deltas']:
        a,b=indexed[row['left'],row['k'],row['support_seed']],indexed[row['right'],row['k'],row['support_seed']]
        assert row['macro_f1_delta']==pytest.approx(a['macro_f1']-b['macro_f1'])
    for row in result['time_increment_gaps']:
        p=lambda name:indexed[name,row['k'],row['support_seed']]['macro_f1']
        assert row['macro_f1_delta']==pytest.approx((p('C1')-p('C0'))-(p('R1')-p('R0')))
    # Each mutation must be rejected without dropping the damaged evidence.
    saved_batch=copy.deepcopy(batch)
    batch['runs'].pop(); write_json(folder/'batch.json',batch)
    with pytest.raises(ValueError): validate_e1_runs(folder)
    write_json(folder/'batch.json',saved_batch)
    for name in ('environment.json','concept_catalog.json'):
        path=folder/name; original_bytes=path.read_bytes(); path.write_text('{}',encoding='utf-8')
        with pytest.raises(ValueError): validate_e1_runs(folder)
        path.write_bytes(original_bytes)
    run=folder/saved_batch['runs'][0]['directory']
    provenance=read_json(run/'provenance.json'); wrong=copy.deepcopy(provenance)
    wrong['support_ids']=wrong['support_ids'][::-1]; write_json(run/'provenance.json',wrong)
    with pytest.raises(ValueError): validate_e1_runs(folder)
    write_json(run/'provenance.json',provenance)
    # Rehashing wrong saved scores must still fail feature-bank/ridge forward verification.
    predictions=run/'predictions.csv'; content=predictions.read_bytes()
    with predictions.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; records=list(reader)
    records[0]['score_clap']=str(float(records[0]['score_clap'])+.001)
    from hmdb51_e0.provenance import write_csv
    write_csv(predictions,records,fields)
    wrong=copy.deepcopy(provenance); wrong['predictions_sha256']=sha256_file(predictions)
    write_json(run/'provenance.json',wrong)
    with pytest.raises(ValueError,match='score|forward|prediction'): validate_e1_runs(folder)
    predictions.write_bytes(content); write_json(run/'provenance.json',provenance)
    validate_e1_runs(folder)


def test_e1_cli_blocks_synthetic_and_final_test_and_has_explicit_steps(e1_data,tmp_path,capsys):
    from hmdb51_e0.e1_cli import build_parser,main
    from hmdb51_e0.e1_experiment import run_e1
    parser=build_parser()
    with pytest.raises(SystemExit): parser.parse_args(['run','--reference-batch','r','--feature-index','f','--stage','final_test'])
    with pytest.raises(SystemExit) as result: parser.parse_args(['prepare','--help'])
    assert result.value.code==0
    config=tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(e1_data['e1_config']),encoding='utf-8')
    paths=tmp_path/'paths.yaml'
    paths.write_text(yaml.safe_dump({**{k:str(v) for k,v in e1_data['paths'].items()},'annotations_root':None,'weights_file':None}),encoding='utf-8')
    assert main(['check','--reference-batch',str(e1_data['reference']),'--config',str(config),'--paths',str(paths),
                 '--concepts',str(e1_data['catalog_path']),'--feature-root',str(e1_data['output'])])==2
    assert 'official' in capsys.readouterr().err
    with pytest.raises(ValueError,match='sealed|development'):
        run_e1(e1_data['reference'],tmp_path/'absent.json',e1_data['paths'],stage='final_test')
    assert not e1_data['e1_root'].exists()


@pytest.mark.parametrize('mutation',['random_controls','concept_scores'])
def test_e1_rehashed_bank_must_still_match_original_frames_and_directions(e1_data,mutation):
    from hmdb51_e0.e1_experiment import load_e1_features
    from hmdb51_e0.e1_features import concept_pool
    index_path=prepare(e1_data); index,bank=load_e1_features(index_path,e1_data['reference'])
    if mutation=='random_controls':
        bank['R1']=np.roll(bank['R1'],1,axis=1)
    else:
        bank['C']=np.roll(bank['C'],1,axis=0)
        bank['C0']=np.stack([concept_pool(c,'C0') for c in bank['C']])
        bank['C1']=np.stack([concept_pool(c,'C1') for c in bank['C']])
    file=index_path.parent/'features.npz'; np.savez_compressed(file,**bank)
    index['npz_sha256']=sha256_file(file); write_json(index_path,index)
    with pytest.raises(ValueError,match='frame|cosine|projection|control|concept'):
        load_e1_features(index_path,e1_data['reference'])
