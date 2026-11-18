"""Validated E1 metrics and controls, with no concept-truth claims."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .ablation_experiment import reference_context
from .config import CLASSES
from .e1_experiment import OFFICIAL,load_bank,schedule,validate_e1_index
from .metrics import classification_metrics
from .provenance import digest,read_json,sha256_file,write_csv,write_json
from .reporting import _read_csv

COMPARISONS=(('C0','V0'),('C1','V1'),('C1','C0'),('C0','R0'),('C1','R1'))


def validate_e1_runs(directory: Path) -> tuple[dict,list[dict],list[dict]]:
    directory=Path(directory); batch=read_json(directory/'batch.json')
    parent,baseline,bundle,supports=reference_context(directory/'reference')
    index=read_json(directory/'feature_index_snapshot.json')
    validate_e1_index(index,directory/'reference',parent,directory,current_code=False)
    bank=load_bank(index,directory/'feature_bank.npz')
    expected=schedule(index['identity']['config'],parent['config'])
    actual=[[r['representation'],r['projection_seed'],r['k'],r['support_seed']] for r in batch['runs']]
    if (batch['status']!='success' or batch['errors'] or batch['evaluation_stage']!='development'
            or batch['config']!=index['identity']['config'] or batch['reference_config']!=parent['config']
            or batch['protocol_id']!=parent['protocol_id'] or batch['feature_identity']!=digest(index['identity'])
            or batch['feature_condition']!=index['feature_condition'] or batch['evaluation_ids']!=parent['evaluation_ids']
            or actual!=expected or batch['expected_runs']!=expected or len({r['directory'] for r in batch['runs']})!=len(expected)):
        raise ValueError('incomplete or mixed E1 batch/condition/projection/support identity')
    if sha256_file(directory/'feature_index_snapshot.json')!=batch['feature_index_sha256']:
        raise ValueError('corrupt E1 feature index snapshot')
    if not (directory/'environment.json').is_file() or sha256_file(directory/'environment.json')!=batch.get('environment_sha256'):
        raise ValueError('corrupt E1 environment snapshot')
    if digest(batch['source']['files'])!=batch['source']['snapshot_sha256']: raise ValueError('E1 source identity mismatch')
    for name,expected_hash in batch['source']['files'].items():
        path=(directory/'code_snapshot'/name).resolve()
        if not path.is_relative_to((directory/'code_snapshot').resolve()) or sha256_file(path)!=expected_hash:
            raise ValueError('unsafe or corrupt E1 code snapshot')
    by_id={r['video_id']:r for r in bundle['rows']}
    positions={r['video_id']:i for i,r in enumerate(index['entries'])}
    dev=batch['evaluation_ids']; results=[]
    for run in batch['runs']:
        name=run['directory']
        if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'): raise ValueError('unsafe E1 run directory')
        folder=directory/name; provenance=read_json(folder/'provenance.json')
        resolved=dict(e1=batch['config'],reference=parent['config'],catalog_sha256=index['identity']['catalog_sha256'])
        if provenance['status']!='success' or provenance['run_id']!=name or read_json(folder/'resolved_config.json')!=resolved:
            raise ValueError('failed or mixed E1 run configuration')
        if any(provenance[k]!=v for k,v in run.items()): raise ValueError('E1 run condition/K/seed mismatch')
        for key in ('protocol_id','feature_identity','evaluation_ids','evaluation_stage','source'):
            if provenance[key]!=batch[key]: raise ValueError(f'E1 provenance mismatch: {key}')
        for filename,key in (('predictions.csv','predictions_sha256'),('metrics.json','metrics_sha256'),
                              ('support_set.csv','support_csv_sha256'),('ridge.npz','ridge_sha256')):
            if sha256_file(folder/filename)!=provenance[key]: raise ValueError(f'corrupt E1 artifact: {filename}')
        support=sorted([r for r in supports['records'] if r['k']==run['k'] and r['support_seed']==run['support_seed']],key=lambda r:(r['class_id'],r['rank_in_class']))
        if _read_csv(folder/'support_set.csv')!=[{k:str(v) for k,v in r.items()} for r in support]:
            raise ValueError('E1 support rows differ from frozen set')
        ids=[r['video_id'] for r in support]; groups=sorted({r['source_group_id'] for r in support})
        rep,projection=run['representation'],run['projection_seed']
        vectors=bank[rep] if projection is None else bank[rep][batch['config']['random_projection']['seeds'].index(projection)]
        dimension=vectors.shape[1]
        if (provenance['support_ids']!=ids or provenance['support_sha256']!=digest(sorted(ids))
                or provenance['n_support']!=5*run['k'] or provenance['support_source_groups']!=groups
                or provenance['n_support_sources']!=len(groups) or provenance['feature_dimension']!=dimension):
            raise ValueError('E1 support/source/feature dimension mismatch')
        predictions=_read_csv(folder/'predictions.csv')
        if [r['video_id'] for r in predictions]!=dev: raise ValueError('E1 prediction coverage mismatch')
        truth=np.array([int(r['true_class_id']) for r in predictions])
        scores=np.array([[float(r[f'score_{c}']) for c in CLASSES] for r in predictions])
        if (not np.isfinite(scores).all() or truth.tolist()!=[by_id[v]['class_id'] for v in dev]
                or [int(r['predicted_class_id']) for r in predictions]!=scores.argmax(1).tolist()):
            raise ValueError('E1 prediction truth/score/argmax mismatch')
        for r in predictions:
            if (r['representation']!=rep or r['projection_seed']!=('' if projection is None else str(projection))
                    or r['run_id']!=name or r['evaluation_stage']!='development' or int(r['k'])!=run['k']
                    or int(r['support_seed'])!=run['support_seed'] or r['source_group_id']!=by_id[r['video_id']]['source_group_id']):
                raise ValueError('E1 prediction identity/source mismatch')
        with np.load(folder/'ridge.npz',allow_pickle=False) as content:
            if set(content.files)!={'W','b'}: raise ValueError('E1 ridge array names mismatch')
            weights,intercept=content['W'],content['b']
        if (weights.shape!=(dimension,5) or intercept.shape!=(5,) or not np.isfinite(weights).all() or not np.isfinite(intercept).all()
                or not np.allclose(vectors[[positions[v] for v in dev]].astype(np.float64) @ weights+intercept,scores,atol=1e-12,rtol=1e-10)):
            raise ValueError('E1 scores differ from saved feature-bank/ridge forward computation')
        metrics=dict(**classification_metrics(truth,scores),evaluation_stage='development',representation=rep,
                     projection_seed=projection,n_support=len(ids),n_support_sources=len(groups),feature_dimension=dimension)
        if metrics!=read_json(folder/'metrics.json'): raise ValueError('E1 metrics differ from per-video recomputation')
        results.append(dict(**run,metrics=metrics,support_sha256=provenance['support_sha256']))
    return batch,results,baseline


def stats(values,prefix):
    values=np.asarray(values,dtype=np.float64)
    return {prefix+'_mean':float(values.mean()),prefix+'_std':float(values.std(ddof=1)) if len(values)>1 else None}


def delta_summary(part,comparison,k,condition):
    row=dict(comparison=comparison,k=k,feature_condition=condition,n_pairs=len(part))
    for metric in ('accuracy','macro_f1'):
        values=np.array([r[metric+'_delta'] for r in part])
        row.update(stats(values,metric+'_delta'))
        row.update({metric+'_delta_median':float(np.median(values)),metric+'_positive_pairs':int((values>0).sum()),
                    metric+'_tied_pairs':int((values==0).sum()),metric+'_negative_pairs':int((values<0).sum())})
    return row


def summarize_e1(directory: Path,output: Path) -> dict:
    batch,runs,baseline=validate_e1_runs(directory)
    directory,output=Path(directory),Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('refusing to overwrite existing E1 report')
    output.mkdir(parents=True,exist_ok=True)
    all_runs=[dict(**r,projection_seed=None,reused_reference=True) for r in baseline]+[dict(**r,reused_reference=False) for r in runs]
    metrics_names=('accuracy','macro_f1',*['recall_'+c for c in CLASSES])
    per_run=[]; all_predictions=[]
    for run in all_runs:
        m=run['metrics']
        row={k:run[k] for k in ('representation','projection_seed','k','support_seed','support_sha256','reused_reference')}
        row.update(feature_condition=batch['feature_condition'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],n_evaluated_videos=m['n_evaluated'])
        row.update({'recall_'+c:m['per_class_recall'][i] for i,c in enumerate(CLASSES)}); per_run.append(row)
        root=directory/'reference' if run['reused_reference'] else directory
        for p in _read_csv(root/run['directory']/'predictions.csv'):
            all_predictions.append(dict(p,projection_seed=run['projection_seed'],reused_reference=run['reused_reference'],feature_condition=batch['feature_condition']))
    representations=('V0','V1','C0','C1','R0','R1')
    shots=batch['reference_config']['sampling']['shots']; seeds=batch['reference_config']['sampling']['support_seeds']
    per_support=[]; summary=[]
    for rep in representations:
        for k in shots:
            part=[]
            for seed in seeds:
                current=[r for r in per_run if r['representation']==rep and r['k']==k and r['support_seed']==seed]
                if len({r['support_sha256'] for r in current})!=1: raise ValueError('E1 projection/support pairing mismatch')
                row=dict(representation=rep,k=k,support_seed=seed,feature_condition=batch['feature_condition'],
                         support_sha256=current[0]['support_sha256'],n_projections=len(current),
                         **{m:float(np.mean([r[m] for r in current])) for m in metrics_names})
                per_support.append(row); part.append(row)
            row=dict(representation=rep,k=k,feature_condition=batch['feature_condition'],n_support_sets=len(part),
                     n_unique_support_sets=len({r['support_sha256'] for r in part}),n_projections_per_support=part[0]['n_projections'],
                     n_evaluated_videos=len(batch['evaluation_ids']))
            for m in metrics_names: row.update(stats([r[m] for r in part],m))
            summary.append(row)
    indexed={(r['representation'],r['k'],r['support_seed']):r for r in per_support}
    deltas=[]; paired_summary=[]; gaps=[]; gap_summary=[]
    for left,right in COMPARISONS:
        for k in shots:
            part=[]
            for seed in seeds:
                a,b=indexed[left,k,seed],indexed[right,k,seed]
                if a['support_sha256']!=b['support_sha256']: raise ValueError('E1 paired support mismatch')
                row=dict(comparison=f'{left}_minus_{right}',left=left,right=right,k=k,support_seed=seed,
                         feature_condition=batch['feature_condition'],support_sha256=a['support_sha256'],
                         **{m+'_delta':a[m]-b[m] for m in metrics_names})
                deltas.append(row); part.append(row)
            paired_summary.append(delta_summary(part,f'{left}_minus_{right}',k,batch['feature_condition']))
    for k in shots:
        part=[]
        for seed in seeds:
            row=dict(comparison='concept_time_minus_random_time',k=k,support_seed=seed,feature_condition=batch['feature_condition'],
                     **{m+'_delta':(indexed['C1',k,seed][m]-indexed['C0',k,seed][m])-(indexed['R1',k,seed][m]-indexed['R0',k,seed][m]) for m in metrics_names})
            gaps.append(row); part.append(row)
        gap_summary.append(delta_summary(part,'concept_time_minus_random_time',k,batch['feature_condition']))
    projection_summary=[]; projection_pairs=[]
    for rep,concept in (('R0','C0'),('R1','C1')):
        for p in batch['config']['random_projection']['seeds']:
            for k in shots:
                part=[r for r in per_run if r['representation']==rep and r['projection_seed']==p and r['k']==k]
                row=dict(representation=rep,projection_seed=p,k=k,feature_condition=batch['feature_condition'],n_support_sets=len(part))
                for m in metrics_names: row.update(stats([r[m] for r in part],m))
                projection_summary.append(row)
                for r in part:
                    a=indexed[concept,k,r['support_seed']]
                    projection_pairs.append(dict(comparison=concept+'_minus_'+rep,projection_seed=p,k=k,support_seed=r['support_seed'],
                                                 feature_condition=batch['feature_condition'],**{m+'_delta':a[m]-r[m] for m in metrics_names}))
    result=dict(evaluation_stage='development',final_test_enabled=False,protocol_id=batch['protocol_id'],feature_identity=batch['feature_identity'],
                feature_condition=batch['feature_condition'],batch=str(directory.resolve()),summary=summary,per_run_metrics=per_run,
                per_support_metrics=per_support,paired_deltas=deltas,paired_summary=paired_summary,time_increment_gaps=gaps,
                time_increment_summary=gap_summary,summary_by_projection_seed=projection_summary,paired_by_projection_seed=projection_pairs,
                aggregation=batch['config']['aggregation'],human_concept_validation='not_performed')
    write_json(output/'summary.json',result)
    for name,data in (('summary.csv',summary),('per_run_metrics.csv',per_run),('per_support_metrics.csv',per_support),
                      ('paired_deltas.csv',deltas),('paired_summary.csv',paired_summary),('time_increment_gaps.csv',gaps),
                      ('time_increment_summary.csv',gap_summary),('summary_by_projection_seed.csv',projection_summary),
                      ('paired_by_projection_seed.csv',projection_pairs),('all_predictions.csv',all_predictions)):
        write_csv(output/name,data,list(data[0]))
    write_json(output/'confusion_counts.json',[dict(representation=r['representation'],projection_seed=r['projection_seed'],k=r['k'],
               support_seed=r['support_seed'],confusion_matrix=r['metrics']['confusion_matrix']) for r in all_runs])
    catalog=read_json(directory/'concept_catalog.json')
    write_json(output/'concept_catalog.json',catalog)
    index=read_json(directory/'feature_index_snapshot.json'); bank=load_bank(index,directory/'feature_bank.npz')
    bundle=read_json(directory/'reference/manifest_snapshot.json')
    by_id={r['video_id']:r for r in bundle['rows']}; positions={r['video_id']:i for i,r in enumerate(index['entries'])}
    class_means=[]
    for cid,name in enumerate(CLASSES):
        indices=[positions[v] for v in batch['evaluation_ids'] if by_id[v]['class_id']==cid]
        values=bank['C'][indices].astype(np.float64).mean(axis=(0,1))
        for j,c in enumerate(catalog['concepts']):
            class_means.append(dict(class_name=name,concept_id=c['id'],label_zh=c['label_zh'],cosine_mean=float(values[j]),
                                    n_videos=len(indices),human_validation='not_performed'))
    write_csv(output/'concept_class_means.csv',class_means,list(class_means[0]))
    _plot(summary,paired_summary,class_means,catalog,output,batch['feature_condition'])
    _write_report(result,catalog,output)
    return result


def _plot(summary,paired,class_means,catalog,output,condition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    context='E1 frozen CLIP: concept baselines' if condition==OFFICIAL else 'PROGRAM VALIDATION: synthetic/unverified features'
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,metric in zip(axes,('accuracy','macro_f1')):
        for rep in ('V0','V1','C0','C1','R0','R1'):
            part=[r for r in summary if r['representation']==rep]
            ax.errorbar([r['k'] for r in part],[100*r[metric+'_mean'] for r in part],
                        yerr=[100*r[metric+'_std'] for r in part],marker='o',capsize=3,label=rep)
        ax.set(xlabel='Labeled videos per class (K)',ylabel=metric+' (%)',ylim=(0,100),xticks=(1,3,5))
        ax.set_title(context+'\nRandom: mean within support; bars: support sample SD',fontsize=9)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output/'learning_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for left,right in COMPARISONS:
        part=[r for r in paired if r['comparison']==left+'_minus_'+right]
        ax.plot([r['k'] for r in part],[100*r['macro_f1_delta_mean'] for r in part],marker='o',label=left+' - '+right)
    ax.axhline(0,color='black',linewidth=.8)
    ax.set(xlabel='Labeled videos per class (K)',ylabel='Paired Macro-F1 difference (pp)',xticks=(1,3,5))
    ax.set_title(context+'\nMatched development/support pairs',fontsize=10); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output/'paired_differences.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    matrix=np.array([[r['cosine_mean'] for r in class_means if r['class_name']==name] for name in CLASSES])
    im=ax.imshow(matrix,aspect='auto',cmap='viridis')
    ax.set(yticks=range(5),yticklabels=CLASSES,xticks=range(len(catalog['concepts'])),xticklabels=[c['id'] for c in catalog['concepts']])
    ax.tick_params(axis='x',rotation=60,labelsize=8)
    ax.set_title(context+'\nDevelopment class means: cosine association, not concept truth',fontsize=10)
    fig.colorbar(im,ax=ax,label='Mean cosine similarity')
    fig.savefig(output/'concept_class_means.png',dpi=180); plt.close(fig)


def _write_report(result,catalog,output):
    title='# E1：显式概念基线与匹配随机投影' if result['feature_condition']==OFFICIAL else '# 仅程序验证：合成／未验证特征，不能作为真实E1结果'
    m=len(catalog['concepts'])
    text=[title,'','阶段：development；最终测试封存。引用60组V0/V1，新增C0/C1和全部R0/R1控制；共享来源划分、支持集合及λ=1。','',
          f'固定{m}个概念，C0/R0为{2*m}维，C1/R1为{5*m}维。冻结同一CLIP联合空间的文本和图像分支；概念分数为余弦，不做逐帧概念向量归一化、softmax或阈值化。C1/R1使用6/5/5分段。','',
          '随机投影种子与支持种子分开；同矩阵跨全部视频/K/支持集合复用。主表先在支持集合内平均各投影的分类指标，再跨支持集合计算样本SD；不是预测集成。','',
          '| 表示 | K | Accuracy均值 | Macro-F1均值 | F1支持集合SD |','| --- | --- | --- | --- | --- |']
    for r in result['summary']:
        text.append(f"| {r['representation']} | {r['k']} | {100*r['accuracy_mean']:.2f}% | {100*r['macro_f1_mean']:.2f}% | {100*r['macro_f1_std']:.2f}pp |")
    text+=['','| 比较 | K | ΔMacro-F1均值 | 中位数 | 正／平／负支持集合数 |','| --- | --- | --- | --- | --- |']
    for r in [*result['paired_summary'],*result['time_increment_summary']]:
        text.append(f"| {r['comparison']} | {r['k']} | {100*r['macro_f1_delta_mean']:+.2f}pp | {100*r['macro_f1_delta_median']:+.2f}pp | {r['macro_f1_positive_pairs']}/{r['macro_f1_tied_pairs']}/{r['macro_f1_negative_pairs']} |")
    text+=['','![学习曲线](learning_curves.png)','![配对差值](paired_differences.png)','![概念类别均值](concept_class_means.png)','',
           '| 概念ID | 中文名称 | 固定英文描述 |','| --- | --- | --- |']
    for c in catalog['concepts']: text.append(f"| {c['id']} | {c['label_zh']} | {c['description_en']} |")
    text+=['','概念库为助手拟定的开发描述，未进行人工概念真值验证。已知五类及E0汇总结果作为开发背景，未按E1分数搜索／筛选提示。均值热图只表示分数与类别的关联，不能证明概念被正确识别；审核模板为空白，不冒充已完成审核。','',
           'C0−V0和C1−V1同时改变维度／表示，必须结合C0−R0和C1−R1判断，不能把单独优于高维视觉基线归因于语义。C1−C0也需与R1−R0对照；时间增量差不是严格因果识别。若随机控制同样有效，语义增量未获支持。','',
           '全部随机种子保留在summary_by_projection_seed.csv和paired_by_projection_seed.csv。相同50段开发反复评价，SD不是独立来源置信区间；来源沿用AI辅助保守分组，没有完整原始来源独立性认证。','',
           '这是需K个支持标签的五类概念基线，不是零样本／未见组合评测，也未验证溺水工程效果。CLIP原本有图文预训练，新增模块不能称首次引入语义。此阶段不包含C2关系模块。','',
           f"批次：`{result['batch']}`；协议：`{result['protocol_id']}`；特征：`{result['feature_identity']}`。",'']
    (output/'e1-results.md').write_text('\n'.join(text),encoding='utf-8')
