"""Validate E1.1 evidence before computing paired, two-level summaries."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import CLASSES
from .e1_reporting import stats, delta_summary
from .e11_features import schedule
from .e11_workflow import RUN_FILES, load_features, parent_context, select_vectors
from .metrics import classification_metrics
from .provenance import digest, read_json, sha256_file, write_csv, write_json
from .reporting import _read_csv

REPRESENTATIONS=('V1','C1','G1','VC1','VG1')
COMPARISONS=(('C1','G1'),('VC1','V1'),('VC1','VG1'),('VG1','V1'))


def validate_runs(directory: Path):
    directory=Path(directory); batch=read_json(directory/'batch.json')
    index,bank=load_features(directory/'feature_index_snapshot.json',current_code=False)
    parent,c_runs,baseline,_,_,bundle,supports=parent_context(directory/'parent')
    expected=schedule(index['identity']['config'],parent['reference_config'])
    actual=[[r['representation'],r['rotation_seed'],r['k'],r['support_seed']] for r in batch['runs']]
    if (batch['status']!='success' or batch['errors'] or batch['evaluation_stage']!='development' or
            batch['config']!=index['identity']['config'] or batch['reference_config']!=parent['reference_config'] or
            batch['protocol_id']!=parent['protocol_id'] or batch['evaluation_ids']!=parent['evaluation_ids'] or
            batch['feature_condition']!=parent['feature_condition'] or batch['feature_identity']!=digest(index['identity']) or
            actual!=expected or batch['expected_runs']!=expected or len({r['directory'] for r in batch['runs']})!=len(expected)):
        raise ValueError('incomplete/mixed E1.1 batch or support/rotation schedule')
    if (sha256_file(directory/'feature_index_snapshot.json')!=batch['feature_index_sha256'] or
            sha256_file(directory/'environment.json')!=batch['environment_sha256'] or
            batch['source']['files']!=index['identity']['code'] or
            digest(batch['source']['files'])!=batch['source']['snapshot_sha256']):
        raise ValueError('corrupt E1.1 source/environment/feature identity')
    for name,expected_hash in batch['source']['files'].items():
        path=(directory/'code_snapshot'/name).resolve()
        if not path.is_relative_to((directory/'code_snapshot').resolve()) or sha256_file(path)!=expected_hash:
            raise ValueError('corrupt/unsafe E1.1 source snapshot')
    by_id={r['video_id']:r for r in bundle['rows']}
    positions={r['video_id']:i for i,r in enumerate(index['entries'])}
    dev=batch['evaluation_ids']; truth=np.array([by_id[v]['class_id'] for v in dev]); results=[]
    for record in batch['runs']:
        name=record['directory']
        if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'):
            raise ValueError('unsafe E1.1 run path')
        folder=directory/name; provenance=read_json(folder/'provenance.json')
        if provenance['status']!='success' or any(provenance[k]!=v for k,v in record.items()):
            raise ValueError('failed or mixed E1.1 run identity')
        for key in ('evaluation_stage','protocol_id','feature_identity','evaluation_ids'):
            if provenance[key]!=batch[key]: raise ValueError('E1.1 provenance identity mismatch')
        if set(provenance['hashes'])!=set(RUN_FILES) or any(sha256_file(folder/f)!=h for f,h in provenance['hashes'].items()):
            raise ValueError('corrupt E1.1 run artifacts')
        if read_json(folder/'resolved_config.json')!=dict(e11=batch['config'],reference=batch['reference_config']):
            raise ValueError('E1.1 resolved configuration mismatch')
        support=sorted([r for r in supports['records'] if r['k']==record['k'] and r['support_seed']==record['support_seed']],key=lambda r:(r['class_id'],r['rank_in_class']))
        if _read_csv(folder/'support_set.csv')!=[{k:str(v) for k,v in row.items()} for row in support]:
            raise ValueError('E1.1 support differs from frozen E1 set')
        ids=[r['video_id'] for r in support]; groups=sorted({r['source_group_id'] for r in support})
        vectors=select_vectors(bank,batch['config'],record['representation'],record['rotation_seed'])
        if (provenance['support_ids']!=ids or provenance['support_sha256']!=digest(sorted(ids)) or
                provenance['support_source_groups']!=groups or provenance['feature_dimension']!=vectors.shape[1]):
            raise ValueError('E1.1 support/source/dimension identity mismatch')
        predictions=_read_csv(folder/'predictions.csv')
        if [r['video_id'] for r in predictions]!=dev: raise ValueError('E1.1 prediction coverage mismatch')
        scores=np.array([[float(r['score_'+c]) for c in CLASSES] for r in predictions])
        for i,r in enumerate(predictions):
            if (r['representation']!=record['representation'] or r['rotation_seed']!=('' if record['rotation_seed'] is None else str(record['rotation_seed'])) or
                    r['run_id']!=name or r['evaluation_stage']!='development' or int(r['k'])!=record['k'] or
                    int(r['support_seed'])!=record['support_seed'] or int(r['true_class_id'])!=int(truth[i]) or
                    int(r['predicted_class_id'])!=int(scores[i].argmax()) or r['source_group_id']!=by_id[r['video_id']]['source_group_id']):
                raise ValueError('E1.1 prediction truth/source/condition mismatch')
        with np.load(folder/'ridge.npz',allow_pickle=False) as model:
            if set(model.files)!={'W','b'}: raise ValueError('invalid ridge array names')
            w,b=model['W'],model['b']
        if (w.shape!=(vectors.shape[1],5) or b.shape!=(5,) or not np.isfinite(w).all() or not np.isfinite(b).all() or
                not np.isfinite(scores).all() or not np.allclose(vectors[[positions[v] for v in dev]].astype(np.float64)@w+b,scores,atol=1e-12,rtol=1e-10)):
            raise ValueError('E1.1 saved scores differ from ridge forward computation')
        metrics=classification_metrics(truth,scores)
        if metrics!=read_json(folder/'metrics.json'): raise ValueError('E1.1 metrics recomputation mismatch')
        results.append(dict(**record,metrics=metrics,support_sha256=provenance['support_sha256'],reused_reference=False,prediction_path=folder/'predictions.csv'))
    reused=[]
    for runs,rep,prefix in ((c_runs,'C1',directory/'parent'),(baseline,'V1',directory/'parent/reference')):
        for r in runs:
            if r['representation']==rep:
                reused.append(dict(**r,rotation_seed=None,reused_reference=True,prediction_path=prefix/r['directory']/'predictions.csv'))
    return batch,results+reused


def geometry_diagnostics(directory: Path) -> list[dict]:
    index=read_json(directory/'feature_index_snapshot.json')
    with np.load(directory/'parent/feature_bank.npz',allow_pickle=False) as data:
        z,e,c=data['Z'].astype('f8'),data['E'].astype('f8'),data['C'].astype('f8')
    with np.load(directory/'features.npz',allow_pickle=False) as data:
        directions=data['D'].astype('f8')
    rows=[]
    for seed,d,scores in [(None,e,c),*[(s,d,z@d.T) for s,d in zip(index['identity']['config']['rotation_seeds'],directions)]]:
        values=scores.reshape(-1,len(e)); cov=np.cov(values,rowvar=False,ddof=0)
        eig=np.maximum(np.linalg.eigvalsh(cov),0); total=float(eig.sum())
        sd=np.sqrt(np.diag(cov)); den=sd[:,None]*sd[None,:]
        corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-16)
        rows.append(dict(representation='C1' if seed is None else 'G1',rotation_seed=seed,
                         gram_max_abs_error=float(np.max(np.abs(d@d.T-e@e.T))),
                         mean_abs_offdiag_score_correlation=float(np.abs(corr[~np.eye(len(e),dtype=bool)]).mean()),
                         first_pc_variance_ratio=float(eig[-1]/total) if total else 0.,
                         covariance_participation_ratio=float(total**2/(eig@eig)) if total else 0.))
    return rows


def summarize(directory: Path, output: Path) -> dict:
    directory,output=Path(directory),Path(output)
    batch,runs=validate_runs(directory)
    if output.exists() and any(output.iterdir()): raise ValueError('refusing to overwrite E1.1 report')
    output.mkdir(parents=True,exist_ok=True)
    per_run=[]; predictions=[]; metrics=('accuracy','macro_f1',*['recall_'+c for c in CLASSES])
    for r in runs:
        row={k:r[k] for k in ('representation','rotation_seed','k','support_seed','support_sha256','reused_reference')}
        row.update(accuracy=r['metrics']['accuracy'],macro_f1=r['metrics']['macro_f1'],
                   **{'recall_'+c:r['metrics']['per_class_recall'][i] for i,c in enumerate(CLASSES)})
        per_run.append(row)
        predictions.extend(dict(p,representation=r['representation'],rotation_seed=r['rotation_seed'],reused_reference=r['reused_reference']) for p in _read_csv(r['prediction_path']))
    per_support=[]; summary=[]
    shots=batch['reference_config']['sampling']['shots']; seeds=batch['reference_config']['sampling']['support_seeds']
    for rep in REPRESENTATIONS:
        for k in shots:
            part=[]
            for seed in seeds:
                current=[r for r in per_run if r['representation']==rep and r['k']==k and r['support_seed']==seed]
                if len(current)!=(5 if rep in ('G1','VG1') else 1) or len({r['support_sha256'] for r in current})!=1:
                    raise ValueError('E1.1 rotation/support pairing mismatch')
                row=dict(representation=rep,k=k,support_seed=seed,support_sha256=current[0]['support_sha256'],n_rotations=len(current),
                         **{m:float(np.mean([r[m] for r in current])) for m in metrics})
                per_support.append(row); part.append(row)
            row=dict(representation=rep,k=k,n_support_sets=len(part),n_unique_support_sets=len({r['support_sha256'] for r in part}),
                     n_development_videos=len(batch['evaluation_ids']),n_rotations_per_support=part[0]['n_rotations'])
            for m in metrics: row.update(stats([r[m] for r in part],m))
            summary.append(row)
    keyed={(r['representation'],r['k'],r['support_seed']):r for r in per_support}
    deltas=[]; paired=[]
    for a,b in COMPARISONS:
        for k in shots:
            part=[]
            for seed in seeds:
                left,right=keyed[a,k,seed],keyed[b,k,seed]
                if left['support_sha256']!=right['support_sha256']: raise ValueError('E1.1 paired support mismatch')
                row=dict(comparison=f'{a}_minus_{b}',left=a,right=b,k=k,support_seed=seed,support_sha256=left['support_sha256'],
                         **{m+'_delta':left[m]-right[m] for m in metrics})
                deltas.append(row); part.append(row)
            paired.append(delta_summary(part,f'{a}_minus_{b}',k,batch['feature_condition']))
    by_rotation=[]
    for rep in ('G1','VG1'):
        for rotation in batch['config']['rotation_seeds']:
            for k in shots:
                part=[r for r in per_run if r['representation']==rep and r['rotation_seed']==rotation and r['k']==k]
                row=dict(representation=rep,rotation_seed=rotation,k=k,n_support_sets=len(part))
                for m in metrics: row.update(stats([r[m] for r in part],m))
                by_rotation.append(row)
    result=dict(batch=str(directory.resolve()),protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],
                feature_condition=batch['feature_condition'],evaluation_stage='development',final_test_enabled=False,
                new_runs=330,reused_runs=60,per_run_metrics=per_run,per_support_metrics=per_support,summary=summary,
                paired_deltas=deltas,paired_summary=paired,summary_by_rotation_seed=by_rotation,
                geometry_diagnostics=geometry_diagnostics(directory))
    for name in ('per_run_metrics','per_support_metrics','summary','paired_deltas','paired_summary','summary_by_rotation_seed','geometry_diagnostics'):
        write_csv(output/(name+'.csv'),result[name],list(result[name][0]))
    write_csv(output/'per_video_predictions.csv',predictions,list(predictions[0]))
    write_json(output/'summary.json',result)
    official=batch['feature_condition']=='official_openai_clip_vit_b32_text_concepts'
    lines=['# E1.1 几何对照与融合诊断', '', '正式特征 development 结果' if official else '仅程序验证：合成特征，不能作为研究结果', '',
           '330 次新拟合；复用 V1/C1 共 60 次。主分析 K=3，K=1/5 为学习曲线。',
           '随机旋转先在每个支持集合内平均，再对 10 个支持集合计算均值与样本标准差；标准差不代表独立视频/独立数据集的不确定性。', '',
           '| 条件 | K | Macro-F1 均值 ± SD | Accuracy 均值 ± SD |','|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['representation']} | {r['k']} | {100*r['macro_f1_mean']:.2f} ± {100*r['macro_f1_std']:.2f} | {100*r['accuracy_mean']:.2f} ± {100*r['accuracy_std']:.2f} |")
    lines+=['','## 配对差值','', '| 比较 | K | Macro-F1 差值（百分点，均值 ± SD） | 正/平/负 |','|---|---:|---:|---|']
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['k']} | {100*r['macro_f1_delta_mean']:.2f} ± {100*r['macro_f1_delta_std']:.2f} | {r['macro_f1_positive_pairs']}/{r['macro_f1_tied_pairs']}/{r['macro_f1_negative_pairs']} |")
    lines+=['','## 解释边界','',
            '- G1 保留文本向量的 Gram 矩阵；不保证视觉投影分数协方差、边际分布或有效维度一致，详见 geometry_diagnostics.csv。',
            '- VC1 > V1 仅支持固定融合在当前 development 上存在增量；VC1 > VG1 才进一步支持概念对齐优于该几何随机对照。',
            '- 使用同一 development 进行后续诊断存在适应性选择，不能把本轮解释为独立泛化验证。',
            '- 概念画面一致性必须由独立的人工审核表取证；本报告不自动填充或推断人工判断。',
            '- 当前是已见类别的视频级 few-shot 分类，不能据此声称 zero-shot、组合泛化或溺水检测有效。final_test 未启用。','']
    (output/'e11-results.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
