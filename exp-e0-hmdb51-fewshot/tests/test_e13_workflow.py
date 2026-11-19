import copy
import numpy as np
import pytest
from hmdb51_e0.provenance import read_json,write_json,sha256_file
from test_round2_protocol import config2,parent
from test_round2_features import round2_inputs
from test_ablation_experiment import inputs
from test_e1_workflow import e1_data,prepare,SyntheticText


def test_e13_full_workflow_keeps_single_zero_shot_and_validates_saved_prototypes(e1_data,tmp_path):
    from hmdb51_e0.e1_experiment import run_e1
    from hmdb51_e0.e11_workflow import prepare_features,run_e11
    from hmdb51_e0.e11_features import DEFAULT_CONFIG as C11
    from hmdb51_e0.e12_workflow import prepare_e12,run_e12
    from hmdb51_e0.e12_features import DEFAULT_CONFIG as C12
    from hmdb51_e0.e13_features import DEFAULT_CONFIG
    from hmdb51_e0.e13_workflow import check_parent,prepare_e13,run_e13,load_features
    from hmdb51_e0.e13_reporting import summarize,validate_runs
    e1=run_e1(e1_data['reference'],prepare(e1_data),e1_data['paths'])
    e11=run_e11(e1,prepare_features(e1,C11,e1_data['paths'],tmp_path/'e11cache'),e1_data['paths'])
    e12=run_e12(e11,prepare_e12(e11,C12,e1_data['paths'],tmp_path/'e12cache'),e1_data['paths'])
    old=sha256_file(e12/'batch.json')
    preflight=check_parent(e12,DEFAULT_CONFIG,e1_data['paths'])
    assert preflight['planned_evaluations']==81 and preflight['text_encoding_executed'] is False
    index=prepare_e13(e12,DEFAULT_CONFIG,e1_data['paths'],tmp_path/'e13cache',text_encoder=SyntheticText())
    folder=run_e13(e12,index,e1_data['paths'])
    result=summarize(folder,tmp_path/'e13report')
    assert len(result['per_run_metrics'])==81 and len(result['summary'])==5
    assert len(result['per_support_metrics'])==40 and len(result['paired_deltas'])==60
    assert sum(r['representation']=='T0' for r in result['per_run_metrics'])==1
    zero=next(r for r in result['summary'] if r['representation']=='T0')
    assert zero['macro_f1_std'] is None and zero['n_support_sets']==0
    for row in result['summary']:
        if row['representation']=='T0': continue
        vals=[np.mean([r['macro_f1'] for r in result['per_run_metrics'] if r['representation']==row['representation'] and r['support_seed']==s]) for s in range(10)]
        assert row['macro_f1_mean']==pytest.approx(np.mean(vals))
        assert row['macro_f1_std']==pytest.approx(np.std(vals,ddof=1))
    assert '仅程序验证' in (tmp_path/'e13report/e13-results.md').read_text(encoding='utf-8')
    assert sha256_file(e12/'batch.json')==old
    batch=read_json(folder/'batch.json'); saved=copy.deepcopy(batch)
    batch['runs'].pop(); write_json(folder/'batch.json',batch)
    with pytest.raises(ValueError): validate_runs(folder)
    write_json(folder/'batch.json',saved)
    run=folder/next(r['directory'] for r in saved['runs'] if r['representation']=='P1')
    path=run/'prototypes.npz'
    with np.load(path,allow_pickle=False) as a: model={k:a[k] for k in a.files}
    model['prototypes']=model['prototypes'][::-1]; np.savez_compressed(path,**model)
    prov=read_json(run/'provenance.json');prov['hashes']['prototypes.npz']=sha256_file(path);write_json(run/'provenance.json',prov)
    with pytest.raises(ValueError,match='prototype|model'): validate_runs(folder)
    info,bank,_=load_features(index)
    bank['D'][0]=bank['D'][1]; np.savez_compressed(index.parent/'features.npz',**bank)
    info['npz_sha256']=sha256_file(index.parent/'features.npz'); write_json(index,info)
    with pytest.raises(ValueError,match='bank|derive|direction'): load_features(index)


def test_e13_cli_rejects_synthetic_and_final_before_outputs(tmp_path,capsys):
    import yaml
    from hmdb51_e0.e13_cli import main,build_parser
    parser=build_parser()
    for cmd in ('check','prepare','run','summarize'):
        with pytest.raises(SystemExit) as exc: parser.parse_args([cmd,'--help'])
        assert exc.value.code==0
    with pytest.raises(SystemExit): parser.parse_args(['run','--e12-batch','x','--feature-index','x','--stage','final_test'])
    parent=tmp_path/'parent';(parent/'parent/parent').mkdir(parents=True)
    write_json(parent/'parent/parent/feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    batch=tmp_path/'batch';(batch/'parent/parent/parent').mkdir(parents=True)
    write_json(batch/'parent/parent/parent/feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    paths=tmp_path/'paths.yaml'; paths.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path/'data'),annotations_root=None,weights_file=None,
               feature_cache_root=str(tmp_path/'old'),run_root=str(tmp_path/'runs'),report_root=str(tmp_path/'reports'))))
    for cmd in ('check','prepare','run','summarize'):
        args=[cmd,'--paths',str(paths)]+(['--batch',str(batch)] if cmd=='summarize' else ['--e12-batch',str(parent)])
        if cmd=='run': args+=['--feature-index',str(tmp_path/'absent')]
        assert main(args)==2 and 'official' in capsys.readouterr().err
    assert not (tmp_path/'runs').exists() and not (tmp_path/'reports').exists()
    from hmdb51_e0.e13_workflow import run_e13
    with pytest.raises(ValueError,match='sealed'): run_e13(parent,tmp_path/'absent',{},stage='final_test')
