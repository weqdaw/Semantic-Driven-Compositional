"""Rebuild support models and all score branches before anchored correction analysis."""
from pathlib import Path
import numpy as np

from .config import CLASSES
from .e1_reporting import stats,delta_summary
from .e14_features import validate_config,schedule,score_model,score_branches
from .e14_workflow import load_prepared,get_support,RUN_FILES,PRED_FIELDS,OFFICIAL
from .metrics import classification_metrics
from .provenance import digest,read_json,sha256_file,write_csv,write_json
from .reporting import _read_csv

PRIMARY=(('A2','T0'),('A2','A3'),('A2','A4'))
SECONDARY=(('A1','P1'),('A2','A1'),('A1','T0'),('A2','P1'),('A2','P0'),('P1','T0'),('P1','P0'))
REPRESENTATIONS=('T0','P0','P1','A1','A2','A3','A4')


def validate_runs(directory: Path):
    directory=Path(directory);batch=read_json(directory/'batch.json');validate_config(batch['config']);expected=schedule(batch['config'])
    actual=[[r['representation'],r['rotation_seed'],r['k'],r['support_seed'],r['lambda']] for r in batch['runs']]
    if (batch['status']!='success' or batch['errors'] or batch['evaluation_stage']!='development' or actual!=expected or
            batch['expected_runs']!=expected or len({r['directory'] for r in batch['runs']})!=80 or batch['gradient_fits_executed']!=0):
        raise ValueError('incomplete/mixed E1.4 condition/lambda/support schedule')
    index,context=load_prepared(directory/'feature_index_snapshot.json',current_code=False)
    if (batch['config']!=index['identity']['config'] or batch['protocol_id']!=index['identity']['protocol_id'] or
            batch['feature_identity']!=digest(index['identity']) or batch['feature_condition']!=index['feature_condition'] or
            batch['evaluation_ids']!=context['batch']['evaluation_ids'] or sha256_file(directory/'feature_index_snapshot.json')!=batch['feature_index_sha256'] or
            sha256_file(directory/'environment.json')!=batch['environment_sha256'] or batch['source']['files']!=index['identity']['code'] or
            digest(batch['source']['files'])!=batch['source']['snapshot_sha256']):raise ValueError('E1.4 input/code/environment identity mismatch')
    for name,expected_hash in batch['source']['files'].items():
        path=(directory/'code_snapshot'/name).resolve()
        if not path.is_relative_to((directory/'code_snapshot').resolve()) or sha256_file(path)!=expected_hash:raise ValueError('unsafe/corrupt source snapshot')
    bank=context['bank'];positions={r['video_id']:j for j,r in enumerate(index['entries'])};by_id={r['video_id']:r for r in context['bundle']['rows']}
    dev=batch['evaluation_ids'];queries=bank['q'][[positions[v] for v in dev]];truth=np.asarray([by_id[v]['class_id'] for v in dev]);results=[]
    for record in batch['runs']:
        name=record['directory']
        if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'):raise ValueError('unsafe E1.4 run path')
        folder=directory/name;prov=read_json(folder/'provenance.json')
        if prov['status']!='success' or any(prov[k]!=v for k,v in record.items()):raise ValueError('failed/mixed E1.4 run identity')
        for key in ('protocol_id','feature_identity','evaluation_stage','evaluation_ids'):
            if prov[key]!=batch[key]:raise ValueError('E1.4 provenance mismatch')
        if set(prov['hashes'])!=set(RUN_FILES) or any(sha256_file(folder/n)!=h for n,h in prov['hashes'].items()):raise ValueError('corrupt E1.4 run artifact')
        if read_json(folder/'resolved_config.json')!=dict(protocol=batch['config'],**{'lambda':record['lambda']}):raise ValueError('E1.4 coefficient/config mismatch')
        support=get_support(context,record['support_seed']);ids=[r['video_id'] for r in support]
        if _read_csv(folder/'support_set.csv')!=[{k:str(v) for k,v in r.items()} for r in support]:raise ValueError('E1.4 support rows mismatch')
        if (prov['support_ids']!=ids or prov['support_sha256']!=digest(sorted(ids)) or prov['feature_dimension']!=queries.shape[1] or
                prov['support_source_groups']!=sorted({r['source_group_id'] for r in support})):raise ValueError('E1.4 support/source/dimension mismatch')
        model=score_model(bank,record['representation'],record['rotation_seed'],[positions[v] for v in ids],np.asarray([r['class_id'] for r in support],dtype=np.int64))
        with np.load(folder/'score_model.npz',allow_pickle=False) as saved:
            if set(saved.files)!=set(model) or any(saved[k].dtype!=v.dtype or not np.array_equal(saved[k],v) for k,v in model.items()):
                raise ValueError('saved score model/anchor differs from support reconstruction')
        expected_scores,text,aux=score_branches(queries,model,record['lambda'])
        rows=_read_csv(folder/'predictions.csv')
        if [r['video_id'] for r in rows]!=dev:raise ValueError('E1.4 prediction coverage mismatch')
        scores=np.asarray([[float(r['score_'+c]) for c in CLASSES] for r in rows])
        saved_text=np.asarray([[float(r['text_score_'+c]) for c in CLASSES] for r in rows])
        saved_aux=np.asarray([[float(r['auxiliary_score_'+c]) for c in CLASSES] for r in rows])
        if (not all(np.isfinite(v).all() for v in (scores,saved_text,saved_aux)) or
                not np.allclose(scores,expected_scores,atol=1e-12,rtol=1e-10) or not np.allclose(saved_text,text,atol=1e-12,rtol=1e-10) or
                not np.allclose(saved_aux,aux,atol=1e-12,rtol=1e-10) or
                not np.allclose(scores,saved_text+record['lambda']*saved_aux,atol=1e-12,rtol=1e-10)):
            raise ValueError('E1.4 text/auxiliary/final score branch differs from forward computation')
        for j,r in enumerate(rows):
            if (r['representation']!=record['representation'] or r['rotation_seed']!=('' if record['rotation_seed'] is None else str(record['rotation_seed'])) or
                    r['run_id']!=name or r['evaluation_stage']!='development' or int(r['k'])!=3 or int(r['support_seed'])!=record['support_seed'] or
                    float(r['lambda'])!=record['lambda'] or int(r['true_class_id'])!=truth[j] or int(r['predicted_class_id'])!=scores[j].argmax() or
                    r['source_group_id']!=by_id[r['video_id']]['source_group_id']):raise ValueError('E1.4 prediction truth/condition mismatch')
        metrics=classification_metrics(truth,scores)
        if metrics!=read_json(folder/'metrics.json'):raise ValueError('E1.4 metrics recomputation mismatch')
        results.append(dict(**record,metrics=metrics,support_sha256=prov['support_sha256'],prediction_path=folder/'predictions.csv',reused_reference=False))
    for ref in context['references']:
        r={k:v for k,v in ref.items() if k!='prediction_relative_path'}
        results.append(dict(**r,**{'lambda':None},prediction_path=directory/'parent'/ref['prediction_relative_path'],reused_reference=True))
    return batch,results


def summarize(directory: Path,output: Path):
    directory,output=Path(directory),Path(output);batch,runs=validate_runs(directory)
    if output.exists() and any(output.iterdir()):raise ValueError('refusing to overwrite E1.4 report')
    output.mkdir(parents=True,exist_ok=True)
    names=('accuracy','macro_f1',*['recall_'+c for c in CLASSES]);per_run=[];predictions=[];row_maps={}
    for r in runs:
        row={k:r[k] for k in ('representation','rotation_seed','k','support_seed','lambda','support_sha256','reused_reference')}
        row.update(accuracy=r['metrics']['accuracy'],macro_f1=r['metrics']['macro_f1'],**{'recall_'+c:r['metrics']['per_class_recall'][j] for j,c in enumerate(CLASSES)})
        per_run.append(row);rows=_read_csv(r['prediction_path']);row_maps[r['representation'],r['rotation_seed'],r['support_seed']]=rows
        predictions.extend(dict(p,reused_reference=r['reused_reference']) for p in rows)
    zero=next(r for r in per_run if r['representation']=='T0');zero_rows={r['video_id']:r for r in row_maps['T0',None,None]}
    transitions=[]
    for r in per_run:
        if r['representation']=='T0':continue
        rows=row_maps[r['representation'],r['rotation_seed'],r['support_seed']]
        changed=fixed=harmed=both_wrong=0
        for p in rows:
            z=zero_rows[p['video_id']]
            if p['predicted_class_id']==z['predicted_class_id']:continue
            changed+=1;a=p['predicted_class_id']==p['true_class_id'];b=z['predicted_class_id']==p['true_class_id']
            fixed+=int(a and not b);harmed+=int(b and not a);both_wrong+=int(not a and not b)
        transitions.append(dict(representation=r['representation'],rotation_seed=r['rotation_seed'],support_seed=r['support_seed'],
                                n_development_videos=len(rows),changed=changed,fixed=fixed,harmed=harmed,both_wrong=both_wrong,net_correct=fixed-harmed))
        if not r['reused_reference']:
            for p in rows:
                for c in CLASSES:
                    if not np.isclose(float(p['text_score_'+c]),float(zero_rows[p['video_id']]['score_'+c]),atol=1e-12,rtol=1e-10):
                        raise ValueError('text anchor differs from original T0 saved score')
    summary=[dict(representation='T0',k=0,n_support_sets=0,n_unique_support_sets=0,n_evaluations=1,
                  **{m+'_mean':zero[m] for m in names},**{m+'_std':None for m in names})]
    per_support=[];by_rotation=[];transition_summary=[]
    for rep in REPRESENTATIONS[1:]:
        part=[];transition_part=[]
        for seed in range(10):
            current=[r for r in per_run if r['representation']==rep and r['support_seed']==seed];n=5 if rep=='A4' else 1
            if len(current)!=n or len({r['support_sha256'] for r in current})!=1:raise ValueError('E1.4 rotation/support pairing mismatch')
            row=dict(representation=rep,k=3,support_seed=seed,support_sha256=current[0]['support_sha256'],n_rotations=n,
                     **{m:float(np.mean([r[m] for r in current])) for m in names})
            part.append(row);per_support.append(row)
            temp=[r for r in transitions if r['representation']==rep and r['support_seed']==seed]
            transition_part.append({k:float(np.mean([r[k] for r in temp])) for k in ('changed','fixed','harmed','both_wrong','net_correct')})
        row=dict(representation=rep,k=3,n_support_sets=10,n_unique_support_sets=len({r['support_sha256'] for r in part}),n_evaluations=50 if rep=='A4' else 10)
        for m in names:row.update(stats([r[m] for r in part],m))
        summary.append(row);tr=dict(representation=rep,n_support_sets=10,n_development_videos=len(batch['evaluation_ids']))
        for m in transition_part[0]:tr.update(stats([r[m] for r in transition_part],m))
        transition_summary.append(tr)
    for rotation in range(5):
        part=[r for r in per_run if r['representation']=='A4' and r['rotation_seed']==rotation]
        row=dict(representation='A4',rotation_seed=rotation,n_support_sets=10)
        for m in names:row.update(stats([r[m] for r in part],m))
        by_rotation.append(row)
    keyed={(r['representation'],r['support_seed']):r for r in per_support};deltas=[];paired=[]
    for left,right in (*PRIMARY,*SECONDARY):
        part=[]
        for seed in range(10):
            a=keyed[left,seed];b=zero if right=='T0' else keyed[right,seed]
            if right!='T0' and a['support_sha256']!=b['support_sha256']:raise ValueError('E1.4 paired support mismatch')
            row=dict(comparison=f'{left}_minus_{right}',left=left,right=right,k=3,support_seed=seed,primary=(left,right) in PRIMARY,
                     fixed_zero_shot_reference=right=='T0',**{m+'_delta':a[m]-b[m] for m in names})
            part.append(row);deltas.append(row)
        paired.append(dict(**delta_summary(part,f'{left}_minus_{right}',3,batch['feature_condition']),primary=(left,right) in PRIMARY))
    main={r['comparison']:r for r in paired if r['primary']};gain=main['A2_minus_T0'];gate=batch['config']['continuation']
    passed=(gain['macro_f1_delta_mean']>=gate['min_macro_f1_gain'] and gain['macro_f1_positive_pairs']>=gate['min_positive_supports'] and
            main['A2_minus_A3']['macro_f1_delta_mean']>0 and main['A2_minus_A4']['macro_f1_delta_mean']>0)
    result=dict(batch=str(directory.resolve()),protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],feature_condition=batch['feature_condition'],
                evaluation_stage='development',final_test_enabled=False,gradient_fits_executed=0,new_evaluations=80,reused_reference_evaluations=21,
                n_development_videos=len(batch['evaluation_ids']),exploratory_continuation_gate=bool(passed),summary=summary,per_run_metrics=per_run,
                per_support_metrics=per_support,paired_deltas=deltas,paired_summary=paired,summary_by_rotation_seed=by_rotation,
                prediction_transitions=transitions,transition_summary=transition_summary)
    for name in ('summary','per_run_metrics','per_support_metrics','paired_deltas','paired_summary','summary_by_rotation_seed','prediction_transitions','transition_summary'):
        write_csv(output/(name+'.csv'),result[name],list(result[name][0]))
    write_csv(output/'per_video_predictions.csv',predictions,[*PRED_FIELDS,'reused_reference']);write_json(output/'summary.json',result)
    lines=['# E1.4 类别文本锚定的少样本分数校正','', '正式CLIP特征 development 结果' if batch['feature_condition']==OFFICIAL else '仅程序验证：合成特征，不能作为研究成绩', '',
           '80次新增支持评估，引用T0一次及P0/P1各10次，共101组。无新编码或梯度训练。',
           '最终分数=text_score+lambda*auxiliary_score；不做融合原型归一化、softmax或温度校准。', '',
           '| 条件 | Macro-F1（%） | 支持集合SD（百分点） | Accuracy（%） |','|---|---:|---:|---:|']
    for r in summary:
        sd='不适用（固定参照）' if r['macro_f1_std'] is None else f"{100*r['macro_f1_std']:.2f}"
        lines.append(f"| {r['representation']} | {100*r['macro_f1_mean']:.2f} | {sd} | {100*r['accuracy_mean']:.2f} |")
    lines+=['','## 配对比较','','| 比较 | 主分析 | Macro-F1差值（百分点，均值 ± SD） | 正/平/负 |','|---|---|---:|---|']
    for r in paired:lines.append(f"| {r['comparison']} | {'是' if r['primary'] else '次要'} | {100*r['macro_f1_delta_mean']:.2f} ± {100*r['macro_f1_delta_std']:.2f} | {r['macro_f1_positive_pairs']}/{r['macro_f1_tied_pairs']}/{r['macro_f1_negative_pairs']} |")
    lines+=['','## 相对T0的错误转换','','| 条件 | 每组平均改对 | 每组平均改错 | 每组平均净增正确 |','|---|---:|---:|---:|']
    for r in transition_summary:lines.append(f"| {r['representation']} | {r['fixed_mean']:.2f} | {r['harmed_mean']:.2f} | {r['net_correct_mean']:.2f} |")
    lines+=['','## 解释边界','',f"探索性继续门槛：{'达到' if passed else '未达到'}；不是统计显著性或自动确认机制。",
            'A2回到T0但没有超过它，不能说明支持信息带来增量；相对较弱P1提高也不能代替A2−T0。',
            'A4只旋转视觉辅助原型，文本锚保持不变；保视觉原型Gram不代表匹配全部查询投影分布。',
            'A4先在支持内平均5次指标，再跨10支持取SD；不是预测集成。T0仅1次，不虚构10次独立零样本重复。',
            '原参考行没有新记录的分支分数，合并CSV相应单元格为空；新增行完整保存原始文本／辅助／最终分数。',
            '本轮固定lambda=0.1为主分析、lambda=1为诊断，不根据开发结果继续搜权重。',
            '开发集已多轮使用，提升仍为开发证据。最终测试封存，未检验概念落地、未见组合或真实溺水工程效果。','']
    (output/'e14-results.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
