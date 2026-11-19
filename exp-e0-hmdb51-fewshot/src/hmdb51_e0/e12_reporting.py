"""Reconstruct support scalers and model scores before E1.2 paired analysis."""
from pathlib import Path
import numpy as np

from .config import CLASSES
from .e1_reporting import stats,delta_summary
from .e12_features import build_vectors,schedule,BLOCKS
from .e12_workflow import load_prepared,get_support,RUN_FILES
from .metrics import classification_metrics
from .provenance import digest,read_json,sha256_file,write_csv,write_json
from .reporting import _read_csv

REPRESENTATIONS=('V1','C1','G1','R1','VC1','VG1',*BLOCKS)
PRIMARY=(('SVC1','SV1'),('SVC1','SVG1'),('SVC1','SVR1'))
SECONDARY=(('SC1','SG1'),('SC1','SR1'),('SV1','V1'),('SC1','C1'),('SG1','G1'),('SR1','R1'),
           ('SVC1','VC1'),('VC1','V1_lambda2'),('V1_lambda2','V1'))


def validate_runs(directory: Path):
    directory=Path(directory); batch=read_json(directory/'batch.json')
    index,context=load_prepared(directory/'feature_index_snapshot.json',current_code=False)
    expected=schedule(index['identity']['config'])
    actual=[[r['representation'],r['control_seed'],r['k'],r['support_seed'],r['lambda']] for r in batch['runs']]
    if (batch['status']!='success' or batch['errors'] or batch['evaluation_stage']!='development' or
            batch['config']!=index['identity']['config'] or batch['reference_config']!=context['reference_config'] or
            batch['protocol_id']!=index['identity']['protocol_id'] or batch['evaluation_ids']!=context['batch']['evaluation_ids'] or
            batch['feature_condition']!=index['feature_condition'] or batch['feature_identity']!=digest(index['identity']) or
            actual!=expected or batch['expected_runs']!=expected or len({r['directory'] for r in batch['runs']})!=len(expected)):
        raise ValueError('incomplete/mixed E1.2 condition/lambda/support schedule')
    if (sha256_file(directory/'feature_index_snapshot.json')!=batch['feature_index_sha256'] or
            sha256_file(directory/'environment.json')!=batch['environment_sha256'] or batch['source']['files']!=index['identity']['code'] or
            digest(batch['source']['files'])!=batch['source']['snapshot_sha256']):
        raise ValueError('corrupt E1.2 input/environment/source identity')
    for name,expected_hash in batch['source']['files'].items():
        path=(directory/'code_snapshot'/name).resolve()
        if not path.is_relative_to((directory/'code_snapshot').resolve()) or sha256_file(path)!=expected_hash:
            raise ValueError('unsafe/corrupt E1.2 code snapshot')
    by_id={r['video_id']:r for r in context['bundle']['rows']}
    positions={r['video_id']:i for i,r in enumerate(index['entries'])}
    dev=batch['evaluation_ids']; truth=np.array([by_id[v]['class_id'] for v in dev]); results=[]; scaling_rows=[]
    for record in batch['runs']:
        name=record['directory']
        if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'): raise ValueError('unsafe E1.2 run path')
        folder=directory/name; provenance=read_json(folder/'provenance.json')
        if provenance['status']!='success' or any(provenance[k]!=v for k,v in record.items()): raise ValueError('failed/mixed E1.2 run identity')
        for key in ('evaluation_stage','protocol_id','feature_identity','evaluation_ids'):
            if provenance[key]!=batch[key]: raise ValueError('E1.2 provenance mismatch')
        if set(provenance['hashes'])!=set(RUN_FILES) or any(sha256_file(folder/f)!=h for f,h in provenance['hashes'].items()):
            raise ValueError('corrupt E1.2 run artifact')
        if read_json(folder/'resolved_config.json')!=dict(e12=batch['config'],reference=context['reference_config'],**{'lambda':record['lambda']}):
            raise ValueError('E1.2 resolved config/lambda mismatch')
        support=get_support(context,record['support_seed']); ids=[r['video_id'] for r in support]
        if _read_csv(folder/'support_set.csv')!=[{k:str(v) for k,v in r.items()} for r in support]:
            raise ValueError('E1.2 support differs from original frozen IDs')
        vectors,scalers,diagnostics=build_vectors(context['raw'],record['representation'],record['control_seed'],[positions[v] for v in ids])
        with np.load(folder/'scalers.npz',allow_pickle=False) as saved:
            if set(saved.files)!=set(scalers) or any(saved[k].dtype!=v.dtype or not np.array_equal(saved[k],v) for k,v in scalers.items()):
                raise ValueError('E1.2 scaler differs from support-only recomputation')
        expected_scaling=dict(scaler_fit_ids=ids,blocks=diagnostics,feature_dimension=vectors.shape[1],post_fusion_row_l2=False)
        if read_json(folder/'scaling.json')!=expected_scaling: raise ValueError('E1.2 scaling diagnostics mismatch')
        if (provenance['support_ids']!=ids or provenance['support_sha256']!=digest(sorted(ids)) or
                provenance['support_source_groups']!=sorted({r['source_group_id'] for r in support}) or provenance['feature_dimension']!=vectors.shape[1]):
            raise ValueError('E1.2 support/source/dimension provenance mismatch')
        predictions=_read_csv(folder/'predictions.csv')
        if [r['video_id'] for r in predictions]!=dev: raise ValueError('E1.2 prediction coverage mismatch')
        scores=np.array([[float(r['score_'+c]) for c in CLASSES] for r in predictions])
        for j,r in enumerate(predictions):
            if (r['representation']!=record['representation'] or r['control_seed']!=('' if record['control_seed'] is None else str(record['control_seed'])) or
                    r['run_id']!=name or r['evaluation_stage']!='development' or int(r['k'])!=3 or int(r['support_seed'])!=record['support_seed'] or
                    float(r['lambda'])!=record['lambda'] or int(r['true_class_id'])!=truth[j] or int(r['predicted_class_id'])!=scores[j].argmax() or
                    r['source_group_id']!=by_id[r['video_id']]['source_group_id']): raise ValueError('E1.2 prediction identity/truth mismatch')
        with np.load(folder/'ridge.npz',allow_pickle=False) as saved:
            if set(saved.files)!={'W','b'}: raise ValueError('invalid E1.2 ridge arrays')
            w,b=saved['W'],saved['b']
        if (w.shape!=(vectors.shape[1],5) or b.shape!=(5,) or not np.isfinite(w).all() or not np.isfinite(b).all() or not np.isfinite(scores).all() or
                not np.allclose(vectors[[positions[v] for v in dev]]@w+b,scores,atol=1e-12,rtol=1e-10)):
            raise ValueError('E1.2 saved score differs from scaled feature/ridge forward computation')
        metrics=classification_metrics(truth,scores)
        if metrics!=read_json(folder/'metrics.json'): raise ValueError('E1.2 metrics recomputation mismatch')
        results.append(dict(**record,metrics=metrics,support_sha256=provenance['support_sha256'],reused_reference=False,prediction_path=folder/'predictions.csv'))
        scaling_rows.extend(dict(representation=record['representation'],control_seed=record['control_seed'],support_seed=record['support_seed'],**d) for d in diagnostics)
    for ref in context['references']:
        r={k:v for k,v in ref.items() if k!='prediction_relative_path'}
        results.append(dict(**r,**{'lambda':1.},reused_reference=True,prediction_path=directory/'parent'/ref['prediction_relative_path']))
    return batch,results,scaling_rows


def summarize(directory: Path,output: Path,*,audit_directory: Path | None=None):
    directory,output=Path(directory),Path(output)
    batch,runs,scaling_rows=validate_runs(directory)
    audit=None
    if audit_directory is not None:
        from .e12_audit import analyze_audit
        audit=analyze_audit(audit_directory,directory/'parent')
    if output.exists() and any(output.iterdir()): raise ValueError('refusing to overwrite E1.2 report')
    output.mkdir(parents=True,exist_ok=True)
    metrics=('accuracy','macro_f1',*['recall_'+c for c in CLASSES]); per_run=[]; predictions=[]
    for r in runs:
        row={k:r[k] for k in ('representation','control_seed','k','support_seed','lambda','support_sha256','reused_reference')}
        row.update(accuracy=r['metrics']['accuracy'],macro_f1=r['metrics']['macro_f1'],**{'recall_'+c:r['metrics']['per_class_recall'][i] for i,c in enumerate(CLASSES)})
        per_run.append(row)
        for p in _read_csv(r['prediction_path']):
            predictions.append({**{k:p[k] for k in ('video_id','source_group_id','true_class_id','predicted_class_id',*['score_'+c for c in CLASSES])},
                                **{k:row[k] for k in ('representation','control_seed','k','support_seed','lambda','reused_reference')}})
    per_support=[]; summary=[]; per_control=[]
    for rep in REPRESENTATIONS:
        part=[]
        for seed in batch['config']['support_seeds']:
            current=[r for r in per_run if r['representation']==rep and r['support_seed']==seed]
            expected=5 if rep in ('G1','R1','VG1','SG1','SR1','SVG1','SVR1') else 1
            if len(current)!=expected or len({r['support_sha256'] for r in current})!=1: raise ValueError('E1.2 random/support pairing mismatch')
            row=dict(representation=rep,k=3,support_seed=seed,support_sha256=current[0]['support_sha256'],n_controls=expected,
                     **{m:float(np.mean([r[m] for r in current])) for m in metrics})
            part.append(row); per_support.append(row)
        row=dict(representation=rep,k=3,n_support_sets=10,n_unique_support_sets=len({r['support_sha256'] for r in part}),n_controls_per_support=expected)
        for m in metrics: row.update(stats([r[m] for r in part],m))
        summary.append(row)
        if expected==5:
            for control in range(5):
                current=[r for r in per_run if r['representation']==rep and r['control_seed']==control]
                row=dict(representation=rep,k=3,control_seed=control,n_support_sets=10)
                for m in metrics: row.update(stats([r[m] for r in current],m))
                per_control.append(row)
    keyed={(r['representation'],r['support_seed']):r for r in per_support}; deltas=[]; paired=[]
    for left,right in (*PRIMARY,*SECONDARY):
        part=[]
        for seed in batch['config']['support_seeds']:
            a,b=keyed[left,seed],keyed[right,seed]
            if a['support_sha256']!=b['support_sha256']: raise ValueError('E1.2 paired support mismatch')
            row=dict(comparison=f'{left}_minus_{right}',left=left,right=right,k=3,support_seed=seed,primary=(left,right) in PRIMARY,
                     **{m+'_delta':a[m]-b[m] for m in metrics})
            part.append(row); deltas.append(row)
        paired.append(dict(**delta_summary(part,f'{left}_minus_{right}',3,batch['feature_condition']),primary=(left,right) in PRIMARY))
    main={r['comparison']:r for r in paired if r['primary']}; visual=main['SVC1_minus_SV1']; threshold=batch['config']['continuation']
    passed=(visual['macro_f1_delta_mean']>=threshold['min_macro_f1_gain'] and visual['macro_f1_positive_pairs']>=threshold['min_positive_supports'] and
            all(main[name]['macro_f1_delta_mean']>0 for name in ('SVC1_minus_SVG1','SVC1_minus_SVR1')))
    result=dict(batch=str(directory.resolve()),evaluation_stage='development',final_test_enabled=False,feature_condition=batch['feature_condition'],
                protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],new_runs=240,reused_runs=180,
                n_development_videos=len(batch['evaluation_ids']),summary=summary,per_run_metrics=per_run,per_support_metrics=per_support,
                paired_deltas=deltas,paired_summary=paired,summary_by_control_seed=per_control,scaling_diagnostics=scaling_rows,
                exploratory_continuation_gate=bool(passed),human_review_completed=bool(audit and audit['human_review_completed']),audit=audit)
    for name in ('summary','per_run_metrics','per_support_metrics','paired_deltas','paired_summary','summary_by_control_seed','scaling_diagnostics'):
        write_csv(output/(name+'.csv'),result[name],list(result[name][0]))
    write_csv(output/'per_video_predictions.csv',predictions,list(predictions[0])); write_json(output/'summary.json',result)
    if audit is not None:
        from .e12_audit import write_audit_report
        write_audit_report(audit,output/'audit')
    official=batch['feature_condition']=='official_openai_clip_vit_b32_text_concepts'
    lines=['# E1.2 概念可见性与支持集尺度匹配', '', '正式特征 development 结果' if official else '仅程序验证：合成特征，不能作为研究成绩', '',
           '主分析3-shot；240次新拟合＋180次原始参考。尺度只从每组15段支持视频估计，融合后无逐视频L2。',
           '随机控制先在同一支持集合内平均5次指标，再对10组支持集合统计；SD不是独立视频置信区间。', '',
           '| 条件 | Macro-F1 均值 ± SD（%） | Accuracy 均值 ± SD（%） |','|---|---:|---:|']
    for r in summary: lines.append(f"| {r['representation']} | {100*r['macro_f1_mean']:.2f} ± {100*r['macro_f1_std']:.2f} | {100*r['accuracy_mean']:.2f} ± {100*r['accuracy_std']:.2f} |")
    lines+=['','## 配对比较','','| 比较 | 主分析 | Macro-F1差值（百分点，均值 ± SD） | 正/平/负支持数 |','|---|---|---:|---|']
    for r in paired: lines.append(f"| {r['comparison']} | {'是' if r['primary'] else '次要诊断'} | {100*r['macro_f1_delta_mean']:.2f} ± {100*r['macro_f1_delta_std']:.2f} | {r['macro_f1_positive_pairs']}/{r['macro_f1_tied_pairs']}/{r['macro_f1_negative_pairs']} |")
    lines+=['','## 解释边界','',f"探索性继续门槛：{'达到' if passed else '未达到'}。该门槛不是显著性检验或自动确认语义机制。",
            f"人工审核：{'已完成，见audit子目录' if result['human_review_completed'] else '未完成或未提供，不能据此声称概念画面对应性已核验'}。",
            '尺度匹配只约束支持集中心化总变化量，不匹配完整协方差、特征分布或有效维度。',
            '当前development已用于多轮诊断；所有提升仍属开发证据。未修改概念、未引入概念监督训练，最终测试封存。',
            '本轮没有未见类别zero-shot、组合泛化或真实溺水识别结论。','']
    (output/'e12-results.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
