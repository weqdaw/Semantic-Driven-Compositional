import copy
import numpy as np
import pytest
from hmdb51_e0.provenance import read_json, write_json, write_csv, sha256_file
from test_round2_protocol import config2,parent
from test_round2_features import round2_inputs
from test_ablation_experiment import inputs
from test_e1_workflow import e1_data,prepare


def test_e11_pairs_330_new_with_60_existing_runs_and_rejects_corruption(e1_data,tmp_path):
    from hmdb51_e0.e1_experiment import run_e1
    from hmdb51_e0.e11_features import DEFAULT_CONFIG
    from hmdb51_e0.e11_workflow import check_parent,prepare_features,run_e11,load_features
    from hmdb51_e0.e11_reporting import summarize,validate_runs
    parent=run_e1(e1_data['reference'],prepare(e1_data),e1_data['paths'])
    before=sha256_file(parent/'batch.json')
    check=check_parent(parent,DEFAULT_CONFIG)
    assert check['fits_executed']==0 and check['planned_new_runs']==330
    index=prepare_features(parent,DEFAULT_CONFIG,e1_data['paths'],tmp_path/'e11cache')
    info,bank=load_features(index,parent)
    assert bank['VC1'].shape==(80,110) and bank['VG1'].shape==(5,80,110)
    folder=run_e11(parent,index,e1_data['paths'])
    report=summarize(folder,tmp_path/'report')
    assert len(report['per_run_metrics'])==390
    assert len(report['per_support_metrics'])==150 and len(report['summary'])==15
    assert len(report['paired_deltas'])==120
    for row in report['summary']:
        rep,k=row['representation'],row['k']
        values=[np.mean([r['macro_f1'] for r in report['per_run_metrics'] if r['representation']==rep and r['k']==k and r['support_seed']==s]) for s in range(10)]
        assert row['macro_f1_mean']==pytest.approx(np.mean(values))
        assert row['macro_f1_std']==pytest.approx(np.std(values,ddof=1))
    assert '仅程序验证' in (tmp_path/'report/e11-results.md').read_text(encoding='utf-8')
    assert sha256_file(parent/'batch.json')==before
    batch=read_json(folder/'batch.json'); original=copy.deepcopy(batch)
    batch['runs'].pop(); write_json(folder/'batch.json',batch)
    with pytest.raises(ValueError): validate_runs(folder)
    write_json(folder/'batch.json',original)
    run=folder/original['runs'][0]['directory']
    import csv
    path=run/'predictions.csv'
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    rows[0]['score_clap']=str(float(rows[0]['score_clap'])+.001)
    write_csv(path,rows,fields)
    provenance=read_json(run/'provenance.json'); provenance['hashes']['predictions.csv']=sha256_file(path)
    write_json(run/'provenance.json',provenance)
    with pytest.raises(ValueError,match='forward|score'): validate_runs(folder)
    # Rehashing a geometrically invalid bank cannot pass semantic validation.
    bank['D'][0,0]=bank['D'][0,1]
    np.savez_compressed(index.parent/'features.npz',**bank)
    info['npz_sha256']=sha256_file(index.parent/'features.npz'); write_json(index,info)
    with pytest.raises(ValueError,match='bank|geometry|derive'): load_features(index,parent)


def test_e11_cli_has_explicit_steps_and_final_test_is_sealed():
    from hmdb51_e0.e11_cli import build_parser
    parser=build_parser()
    for command in ('check','prepare','export-audit','run','summarize'):
        with pytest.raises(SystemExit) as exc: parser.parse_args([command,'--help'])
        assert exc.value.code==0
    with pytest.raises(SystemExit):
        parser.parse_args(['run','--e1-batch','x','--feature-index','x','--stage','final_test'])


def test_all_production_cli_stages_reject_synthetic_inputs_before_outputs(tmp_path,capsys):
    import yaml
    from hmdb51_e0.e11_cli import main
    parent=tmp_path/'parent'; parent.mkdir()
    write_json(parent/'feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    folder=tmp_path/'batch'; (folder/'parent').mkdir(parents=True)
    write_json(folder/'parent/feature_index_snapshot.json',{'feature_condition':'validation_only_unverified_features'})
    paths=tmp_path/'paths.yaml'
    paths.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path/'data'),annotations_root=None,weights_file=None,
                     feature_cache_root=str(tmp_path/'legacy'),run_root=str(tmp_path/'runs'),report_root=str(tmp_path/'reports'))),encoding='utf-8')
    for command in ('check','prepare','export-audit','run','summarize'):
        args=[command,'--paths',str(paths)]
        args+=['--batch',str(folder)] if command=='summarize' else ['--e1-batch',str(parent)]
        if command=='run': args+=['--feature-index',str(tmp_path/'absent.json')]
        assert main(args)==2
        assert 'official' in capsys.readouterr().err
    assert not (tmp_path/'runs').exists() and not (tmp_path/'reports').exists()
    from hmdb51_e0.e11_workflow import run_e11
    with pytest.raises(ValueError,match='sealed'):
        run_e11(parent,tmp_path/'absent.json',{},stage='final_test')
