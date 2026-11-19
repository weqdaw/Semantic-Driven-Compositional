import copy
import csv
import numpy as np
import pytest
from hmdb51_e0.provenance import read_json,write_json,write_csv,sha256_file
from test_round2_protocol import config2,parent
from test_round2_features import round2_inputs
from test_ablation_experiment import inputs
from test_e1_workflow import e1_data,prepare


def test_e12_full_workflow_uses_support_scalers_and_pairs_both_random_controls(e1_data,tmp_path):
    from hmdb51_e0.e1_experiment import run_e1
    from hmdb51_e0.e11_workflow import prepare_features,run_e11
    from hmdb51_e0.e11_features import DEFAULT_CONFIG as C11
    from hmdb51_e0.e12_features import DEFAULT_CONFIG
    from hmdb51_e0.e12_workflow import check_parent,prepare_e12,run_e12
    from hmdb51_e0.e12_reporting import summarize,validate_runs
    e1=run_e1(e1_data['reference'],prepare(e1_data),e1_data['paths'])
    e11=run_e11(e1,prepare_features(e1,C11,e1_data['paths'],tmp_path/'e11cache'),e1_data['paths'])
    before=sha256_file(e11/'batch.json')
    preflight=check_parent(e11,DEFAULT_CONFIG)
    assert preflight['fits_executed']==0 and preflight['planned_new_runs']==240
    index=prepare_e12(e11,DEFAULT_CONFIG,e1_data['paths'],tmp_path/'e12cache')
    folder=run_e12(e11,index,e1_data['paths'])
    result=summarize(folder,tmp_path/'e12report')
    assert len(result['per_run_metrics'])==420 and len(result['per_support_metrics'])==140
    assert len(result['summary'])==14 and len(result['paired_deltas'])==120
    assert result['human_review_completed'] is False
    assert '仅程序验证' in (tmp_path/'e12report/e12-results.md').read_text(encoding='utf-8')
    for row in result['summary']:
        values=[np.mean([r['macro_f1'] for r in result['per_run_metrics'] if r['representation']==row['representation'] and r['support_seed']==seed]) for seed in range(10)]
        assert row['macro_f1_mean']==pytest.approx(np.mean(values))
        assert row['macro_f1_std']==pytest.approx(np.std(values,ddof=1))
    assert sha256_file(e11/'batch.json')==before
    batch=read_json(folder/'batch.json'); old=copy.deepcopy(batch)
    batch['runs'].pop(); write_json(folder/'batch.json',batch)
    with pytest.raises(ValueError): validate_runs(folder)
    write_json(folder/'batch.json',old)
    run=folder/next(r['directory'] for r in old['runs'] if r['representation']=='SC1')
    path=run/'scalers.npz'; content=path.read_bytes()
    with np.load(path,allow_pickle=False) as data: arrays={k:data[k] for k in data.files}
    arrays['C1_scale']*=2; np.savez_compressed(path,**arrays)
    provenance=read_json(run/'provenance.json'); saved=copy.deepcopy(provenance)
    provenance['hashes']['scalers.npz']=sha256_file(path); write_json(run/'provenance.json',provenance)
    with pytest.raises(ValueError,match='scal|support'): validate_runs(folder)
    path.write_bytes(content); write_json(run/'provenance.json',saved)
    path=run/'predictions.csv'; content=path.read_bytes()
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    rows[0]['score_clap']=str(float(rows[0]['score_clap'])+.001); write_csv(path,rows,fields)
    provenance['hashes']=copy.deepcopy(saved['hashes']); provenance['hashes']['predictions.csv']=sha256_file(path)
    write_json(run/'provenance.json',provenance)
    with pytest.raises(ValueError,match='forward|score'): validate_runs(folder)
    path.write_bytes(content); write_json(run/'provenance.json',saved)
    env=folder/'environment.json'; content=env.read_bytes(); env.write_text('{}')
    with pytest.raises(ValueError): validate_runs(folder)
    env.write_bytes(content)
    validate_runs(folder)


def test_e12_cli_blocks_synthetic_and_final_test_before_mutation(tmp_path,capsys):
    import yaml
    from hmdb51_e0.e12_cli import main,build_parser
    parser=build_parser()
    for command in ('check','prepare','audit','run','summarize'):
        with pytest.raises(SystemExit) as exc: parser.parse_args([command,'--help'])
        assert exc.value.code==0
    with pytest.raises(SystemExit): parser.parse_args(['run','--e11-batch','x','--feature-index','x','--stage','final_test'])
    p=tmp_path/'e11'; (p/'parent').mkdir(parents=True)
    write_json(p/'parent/feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    b=tmp_path/'batch'; (b/'parent/parent').mkdir(parents=True)
    write_json(b/'parent/parent/feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    paths=tmp_path/'paths.yaml'
    paths.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path/'data'),annotations_root=None,weights_file=None,
                     feature_cache_root=str(tmp_path/'old'),run_root=str(tmp_path/'runs'),report_root=str(tmp_path/'reports'))))
    for command in ('check','prepare','audit','run','summarize'):
        args=[command,'--paths',str(paths)]+(['--batch',str(b)] if command=='summarize' else ['--e11-batch',str(p)])
        if command=='audit': args+=['--audit-directory',str(tmp_path/'absent')]
        if command=='run': args+=['--feature-index',str(tmp_path/'absent.json')]
        assert main(args)==2 and 'official' in capsys.readouterr().err
    assert not (tmp_path/'runs').exists() and not (tmp_path/'reports').exists()
    from hmdb51_e0.e12_workflow import run_e12
    with pytest.raises(ValueError,match='sealed'): run_e12(p,tmp_path/'absent.json',{},stage='final_test')
