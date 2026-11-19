"""Independent prototype reconstruction; one zero-shot reference and paired support statistics."""
from pathlib import Path
import numpy as np

from .config import CLASSES
from .e1_reporting import stats,delta_summary
from .e13_features import validate_config,schedule,prototype_model
from .e13_workflow import load_features,get_support,RUN_FILES,OFFICIAL
from .metrics import classification_metrics
from .provenance import digest,read_json,sha256_file,write_csv,write_json
from .reporting import _read_csv

PRIMARY=(('P1','P0'),('P1','P2'),('P1','P3'))
SECONDARY=(('P1','T0'),('P0','T0'),('P2','P0'))


def validate_runs(directory: Path):
    directory=Path(directory); batch=read_json(directory/'batch.json');validate_config(batch['config'])
    expected=schedule(batch['config'])
    actual=[[r['representation'],r['rotation_seed'],r['k'],r['support_seed']] for r in batch['runs']]
    if (batch['status']!='success' or batch['errors'] or batch['evaluation_stage']!='development' or
            actual!=expected or batch['expected_runs']!=expected or len({r['directory'] for r in batch['runs']})!=81 or batch['gradient_fits_executed']!=0):
        raise ValueError('incomplete/mixed E1.3 evaluation schedule')
    index,bank,context=load_features(directory/'feature_index_snapshot.json',current_code=False)
    if (batch['config']!=index['identity']['config'] or batch['protocol_id']!=index['identity']['protocol_id'] or
            batch['feature_identity']!=digest(index['identity']) or batch['feature_condition']!=index['feature_condition'] or
            batch['evaluation_ids']!=context['batch']['evaluation_ids'] or sha256_file(directory/'feature_index_snapshot.json')!=batch['feature_index_sha256'] or
            sha256_file(directory/'environment.json')!=batch['environment_sha256'] or batch['source']['files']!=index['identity']['code'] or
            digest(batch['source']['files'])!=batch['source']['snapshot_sha256']): raise ValueError('E1.3 input/source/environment identity mismatch')
    for name,expected_hash in batch['source']['files'].items():
        path=(directory/'code_snapshot'/name).resolve()
        if not path.is_relative_to((directory/'code_snapshot').resolve()) or sha256_file(path)!=expected_hash: raise ValueError('unsafe/corrupt source snapshot')
    positions={r['video_id']:j for j,r in enumerate(index['entries'])};by_id={r['video_id']:r for r in context['bundle']['rows']}
    dev=batch['evaluation_ids'];truth=np.asarray([by_id[v]['class_id'] for v in dev]);results=[]
    for record in batch['runs']:
        name=record['directory']
        if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'): raise ValueError('unsafe E1.3 run path')
        folder=directory/name;prov=read_json(folder/'provenance.json')
        if prov['status']!='success' or any(prov[k]!=v for k,v in record.items()): raise ValueError('failed/mixed E1.3 run identity')
        for k in ('protocol_id','feature_identity','evaluation_stage','evaluation_ids'):
            if prov[k]!=batch[k]: raise ValueError('E1.3 provenance mismatch')
        if set(prov['hashes'])!=set(RUN_FILES) or any(sha256_file(folder/n)!=h for n,h in prov['hashes'].items()): raise ValueError('corrupt E1.3 run artifact')
        if read_json(folder/'resolved_config.json')!=batch['config']: raise ValueError('E1.3 run configuration mismatch')
        support=get_support(context,record['support_seed']);ids=[r['video_id'] for r in support]
        if _read_csv(folder/'support_set.csv')!=[{k:str(v) for k,v in r.items()} for r in support]: raise ValueError('E1.3 support rows mismatch')
        if (len(ids)!=5*record['k'] or prov['support_ids']!=ids or prov['support_sha256']!=digest(sorted(ids)) or
                prov['support_source_groups']!=sorted({r['source_group_id'] for r in support}) or prov['feature_dimension']!=bank['q'].shape[1]):
            raise ValueError('E1.3 support/source/dimension identity mismatch')
        prototypes,visual=prototype_model(bank,record['representation'],record['rotation_seed'],[positions[v] for v in ids],np.asarray([r['class_id'] for r in support],dtype=np.int64))
        with np.load(folder/'prototypes.npz',allow_pickle=False) as saved:
            wanted=dict(prototypes=prototypes,visual_prototypes=visual)
            if set(saved.files)!=set(wanted) or any(saved[k].dtype!=v.dtype or not np.array_equal(saved[k],v) for k,v in wanted.items()):
                raise ValueError('saved prototype/model differs from support/text reconstruction')
        rows=_read_csv(folder/'predictions.csv')
        if [r['video_id'] for r in rows]!=dev: raise ValueError('E1.3 prediction coverage mismatch')
        scores=np.asarray([[float(r['score_'+c]) for c in CLASSES] for r in rows])
        for j,r in enumerate(rows):
            if (r['representation']!=record['representation'] or r['rotation_seed']!=('' if record['rotation_seed'] is None else str(record['rotation_seed'])) or
                    r['support_seed']!=('' if record['support_seed'] is None else str(record['support_seed'])) or int(r['k'])!=record['k'] or
                    r['run_id']!=name or r['evaluation_stage']!='development' or int(r['true_class_id'])!=truth[j] or int(r['predicted_class_id'])!=scores[j].argmax() or
                    r['source_group_id']!=by_id[r['video_id']]['source_group_id']): raise ValueError('E1.3 prediction truth/condition mismatch')
        if not np.isfinite(scores).all() or not np.allclose(bank['q'][[positions[v] for v in dev]]@prototypes.T,scores,atol=1e-12,rtol=1e-10):
            raise ValueError('E1.3 scores differ from reconstructed cosine forward')
        metrics=classification_metrics(truth,scores)
        if metrics!=read_json(folder/'metrics.json'): raise ValueError('E1.3 metric recomputation mismatch')
        results.append(dict(**record,metrics=metrics,support_sha256=prov['support_sha256'],prediction_path=folder/'predictions.csv'))
    return batch,results


def summarize(directory: Path,output: Path):
    directory,output=Path(directory),Path(output);batch,runs=validate_runs(directory)
    if output.exists() and any(output.iterdir()): raise ValueError('refusing to overwrite E1.3 report')
    output.mkdir(parents=True,exist_ok=True)
    names=('accuracy','macro_f1',*['recall_'+c for c in CLASSES]);per_run=[];predictions=[]
    for r in runs:
        row={k:r[k] for k in ('representation','rotation_seed','k','support_seed','support_sha256')}
        row.update(n_support=5*r['k'],accuracy=r['metrics']['accuracy'],macro_f1=r['metrics']['macro_f1'],
                   **{'recall_'+c:r['metrics']['per_class_recall'][j] for j,c in enumerate(CLASSES)})
        per_run.append(row);predictions.extend(_read_csv(r['prediction_path']))
    zero=next(r for r in per_run if r['representation']=='T0')
    summary=[dict(representation='T0',k=0,n_support_sets=0,n_unique_support_sets=0,n_evaluations=1,
                  **{m+'_mean':zero[m] for m in names},**{m+'_std':None for m in names})]
    per_support=[];by_rotation=[]
    for rep in ('P0','P1','P2','P3'):
        part=[]
        for seed in range(10):
            current=[r for r in per_run if r['representation']==rep and r['support_seed']==seed]
            if len(current)!=(5 if rep=='P3' else 1) or len({r['support_sha256'] for r in current})!=1: raise ValueError('prototype rotation/support pairing mismatch')
            row=dict(representation=rep,k=3,support_seed=seed,support_sha256=current[0]['support_sha256'],n_rotations=len(current),
                     **{m:float(np.mean([r[m] for r in current])) for m in names})
            part.append(row);per_support.append(row)
        row=dict(representation=rep,k=3,n_support_sets=10,n_unique_support_sets=len({r['support_sha256'] for r in part}),n_evaluations=50 if rep=='P3' else 10)
        for m in names:row.update(stats([r[m] for r in part],m))
        summary.append(row)
    for rotation in range(5):
        part=[r for r in per_run if r['representation']=='P3' and r['rotation_seed']==rotation]
        row=dict(representation='P3',rotation_seed=rotation,n_support_sets=10)
        for m in names:row.update(stats([r[m] for r in part],m))
        by_rotation.append(row)
    keyed={(r['representation'],r['support_seed']):r for r in per_support};deltas=[];paired=[]
    for left,right in (*PRIMARY,*SECONDARY):
        part=[]
        for seed in range(10):
            a=keyed[left,seed];b=zero if right=='T0' else keyed[right,seed]
            if right!='T0' and a['support_sha256']!=b['support_sha256']: raise ValueError('E1.3 paired support mismatch')
            row=dict(comparison=f'{left}_minus_{right}',left=left,right=right,support_seed=seed,k=3,primary=(left,right) in PRIMARY,
                     fixed_zero_shot_reference=right=='T0',**{m+'_delta':a[m]-b[m] for m in names})
            part.append(row);deltas.append(row)
        paired.append(dict(**delta_summary(part,f'{left}_minus_{right}',3,batch['feature_condition']),primary=(left,right) in PRIMARY))
    result=dict(batch=str(directory.resolve()),protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],feature_condition=batch['feature_condition'],
                evaluation_stage='development',final_test_enabled=False,gradient_fits_executed=0,n_evaluations=81,
                n_development_videos=len(batch['evaluation_ids']),summary=summary,per_run_metrics=per_run,per_support_metrics=per_support,
                paired_deltas=deltas,paired_summary=paired,summary_by_rotation_seed=by_rotation)
    for name in ('summary','per_run_metrics','per_support_metrics','paired_deltas','paired_summary','summary_by_rotation_seed'):
        write_csv(output/(name+'.csv'),result[name],list(result[name][0]))
    write_csv(output/'per_video_predictions.csv',predictions,list(predictions[0]));write_json(output/'summary.json',result)
    lines=['# E1.3 类别语义参照与支持原型','', '正式CLIP特征 development 结果' if batch['feature_condition']==OFFICIAL else '仅程序验证：合成特征，不能作为研究成绩', '',
           'T0零样本参照仅1次；P0/P1/P2各10次，P3五种旋转共50次；合计81次，不进行梯度训练。',
           '全部P条件共享原3-shot支持集合与同一维CLIP视频表示，不与旧2560维V1之间的差值作单模块归因。', '',
           '| 条件 | Macro-F1（%） | 支持集合SD（百分点） | Accuracy（%） |','|---|---:|---:|---:|']
    for r in summary:
        sd='不适用（固定参照）' if r['macro_f1_std'] is None else f"{100*r['macro_f1_std']:.2f}"
        lines.append(f"| {r['representation']} | {100*r['macro_f1_mean']:.2f} | {sd} | {100*r['accuracy_mean']:.2f} |")
    lines+=['','## 配对比较','','| 比较 | 主分析 | Macro-F1差值（百分点，均值 ± SD） | 正/平/负 |','|---|---|---:|---|']
    for r in paired:lines.append(f"| {r['comparison']} | {'是' if r['primary'] else '次要'} | {100*r['macro_f1_delta_mean']:.2f} ± {100*r['macro_f1_delta_std']:.2f} | {r['macro_f1_positive_pairs']}/{r['macro_f1_tied_pairs']}/{r['macro_f1_negative_pairs']} |")
    lines+=['','## 解释边界','',
            'P3先在同一支持集合内平均5个旋转的指标，再跨10支持取样本SD；不是预测集成或50个独立支持集合。',
            'T0仅是一组固定预测，其与10支持集合的差值不代表10次独立零样本重复。',
            'P2是固定循环错配；正交对照保文本Gram，不保证完整视觉投影分布相同。',
            'P1优于P0及错误文本对照仅支持当前类别文本辅助原型；若接近T0，支持视频的额外作用仍有限。',
            '本轮使用类别名称，资源条件与旧姿态概念不同。类别文本不是已验证的逐帧概念，也不证明关系推理。',
            'T0不使用下游支持视频，不代表CLIP预训练未见这些动作；当前development多轮使用，不能视为独立泛化验证。',
            '概念人工审核仍通过原脚本11单独进行；本轮不自动填写或声称完成。最终测试封存，无未见组合或真实溺水结论。','']
    (output/'e13-results.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
