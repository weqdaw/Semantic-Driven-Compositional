import copy
import csv
import numpy as np
import pytest
from hmdb51_e0.provenance import read_json,write_json,write_csv,sha256_file
from test_round2_protocol import config2,parent
from test_round2_features import round2_inputs
from test_ablation_experiment import inputs
from test_e1_workflow import e1_data,prepare,SyntheticText


def test_e14_workflow_pairs101_evaluations_and_rejects_rehashed_branch_corruption(e1_data,tmp_path):
    from hmdb51_e0.e1_experiment import run_e1
    from hmdb51_e0.e11_workflow import prepare_features,run_e11
    from hmdb51_e0.e11_features import DEFAULT_CONFIG as C11
    from hmdb51_e0.e12_workflow import prepare_e12,run_e12
    from hmdb51_e0.e12_features import DEFAULT_CONFIG as C12
    from hmdb51_e0.e13_workflow import prepare_e13,run_e13
    from hmdb51_e0.e13_features import DEFAULT_CONFIG as C13
    from hmdb51_e0.e14_features import DEFAULT_CONFIG
    from hmdb51_e0.e14_workflow import check_parent,prepare_e14,run_e14
    from hmdb51_e0.e14_reporting import summarize,validate_runs
    e1=run_e1(e1_data['reference'],prepare(e1_data),e1_data['paths'])
    e11=run_e11(e1,prepare_features(e1,C11,e1_data['paths'],tmp_path/'e11cache'),e1_data['paths'])
    e12=run_e12(e11,prepare_e12(e11,C12,e1_data['paths'],tmp_path/'e12cache'),e1_data['paths'])
    e13=run_e13(e12,prepare_e13(e12,C13,e1_data['paths'],tmp_path/'e13cache',text_encoder=SyntheticText()),e1_data['paths'])
    old=sha256_file(e13/'batch.json')
    check=check_parent(e13,DEFAULT_CONFIG)
    assert check['planned_new_evaluations']==80 and check['reused_reference_evaluations']==21
    index=prepare_e14(e13,DEFAULT_CONFIG,e1_data['paths'],tmp_path/'e14cache')
    folder=run_e14(e13,index,e1_data['paths'])
    result=summarize(folder,tmp_path/'e14report')
    assert len(result['per_run_metrics'])==101 and len(result['summary'])==7
    assert len(result['per_support_metrics'])==60 and len(result['paired_deltas'])==100
    assert len(result['prediction_transitions'])==100
    zero=next(r for r in result['summary'] if r['representation']=='T0')
    assert zero['macro_f1_std'] is None and zero['n_support_sets']==0
    assert sum(r['representation']=='T0' for r in result['per_run_metrics'])==1
    for row in result['summary']:
        if row['representation']=='T0':continue
        vals=[np.mean([r['macro_f1'] for r in result['per_run_metrics'] if r['representation']==row['representation'] and r['support_seed']==s]) for s in range(10)]
        assert row['macro_f1_mean']==pytest.approx(np.mean(vals))
        assert row['macro_f1_std']==pytest.approx(np.std(vals,ddof=1))
    assert sha256_file(e13/'batch.json')==old
    assert '仅程序验证' in (tmp_path/'e14report/e14-results.md').read_text(encoding='utf-8')
    batch=read_json(folder/'batch.json');saved=copy.deepcopy(batch);batch['runs'].pop();write_json(folder/'batch.json',batch)
    with pytest.raises(ValueError):validate_runs(folder)
    write_json(folder/'batch.json',saved)
    run=folder/saved['runs'][0]['directory'];path=run/'score_model.npz';original=path.read_bytes()
    with np.load(path,allow_pickle=False) as a:model={k:a[k] for k in a.files}
    model['text_prototypes']=model['text_prototypes'][::-1];np.savez_compressed(path,**model)
    prov=read_json(run/'provenance.json');clean=copy.deepcopy(prov);prov['hashes']['score_model.npz']=sha256_file(path);write_json(run/'provenance.json',prov)
    with pytest.raises(ValueError,match='model|anchor|prototype'):validate_runs(folder)
    path.write_bytes(original);write_json(run/'provenance.json',clean)
    path=run/'predictions.csv'
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    rows[0]['text_score_clap']=str(float(rows[0]['text_score_clap'])+.001);write_csv(path,rows,fields)
    prov=copy.deepcopy(clean);prov['hashes']['predictions.csv']=sha256_file(path);write_json(run/'provenance.json',prov)
    with pytest.raises(ValueError,match='score|branch|anchor'):validate_runs(folder)


def test_e14_cli_rejects_synthetic_and_final_test_before_outputs(tmp_path,capsys):
    import yaml
    from hmdb51_e0.e14_cli import main,build_parser
    parser=build_parser()
    for command in ('check','prepare','run','summarize'):
        with pytest.raises(SystemExit) as exc:parser.parse_args([command,'--help'])
        assert exc.value.code==0
    with pytest.raises(SystemExit):parser.parse_args(['run','--e13-batch','x','--feature-index','x','--stage','final_test'])
    parent=tmp_path/'parent';parent.mkdir();write_json(parent/'feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    batch=tmp_path/'batch';(batch/'parent').mkdir(parents=True);write_json(batch/'parent/feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    paths=tmp_path/'paths.yaml';paths.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path/'data'),annotations_root=None,weights_file=None,
           feature_cache_root=str(tmp_path/'old'),run_root=str(tmp_path/'runs'),report_root=str(tmp_path/'reports'))))
    for command in ('check','prepare','run','summarize'):
        args=[command,'--paths',str(paths)]+(['--batch',str(batch)] if command=='summarize' else ['--e13-batch',str(parent)])
        if command=='run':args+=['--feature-index',str(tmp_path/'absent')]
        assert main(args)==2 and 'official' in capsys.readouterr().err
    assert not (tmp_path/'runs').exists() and not (tmp_path/'reports').exists()
    from hmdb51_e0.e14_workflow import run_e14
    with pytest.raises(ValueError,match='sealed'):run_e14(parent,tmp_path/'absent',{},stage='final_test')
