"""Standalone validation and paired, two-level aggregation of ablation evidence."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .ablation_experiment import reference_context, reference_hashes, run_schedule, validate_ablation_index
from .config import CLASSES
from .metrics import classification_metrics
from .provenance import digest, read_json, sha256_file, write_csv, write_json
from .reporting import _read_csv

COMPARISONS = (('V1', 'V0'), ('V1', 'V1_repeat'), ('V1', 'V1_shuffle'), ('V1_repeat', 'V0'))


def validate_ablation_runs(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    directory = Path(directory)
    batch = read_json(directory / 'batch.json')
    parent, baseline, bundle, supports = reference_context(directory / 'reference')
    index = read_json(directory / 'feature_index_snapshot.json')
    validate_ablation_index(index, directory / 'reference', parent, current_code=False)
    expected = run_schedule(index['identity']['config'], parent['config'])
    actual = [[r['representation'], r['shuffle_seed'], r['k'], r['support_seed']] for r in batch['runs']]
    if (batch['status'] != 'success' or batch['errors'] or batch['evaluation_stage'] != 'development'
            or batch['config'] != index['identity']['config'] or batch['reference_config'] != parent['config']
            or batch['protocol_id'] != parent['protocol_id'] or batch['feature_identity'] != digest(index['identity'])
            or batch['feature_condition'] != parent['feature_condition'] or batch['evaluation_ids'] != parent['evaluation_ids']
            or batch['reference_evidence_sha256'] != digest(reference_hashes(directory / 'reference', parent))
            or actual != expected or batch['expected_runs'] != expected
            or len({r['directory'] for r in batch['runs']}) != len(expected)):
        raise ValueError('incomplete or mixed ablation batch/condition/seed/reference')
    if (sha256_file(directory / 'feature_index_snapshot.json') != batch['feature_index_sha256']
            or sha256_file(directory / 'permutations.csv') != index['permutations_csv_sha256']):
        raise ValueError('corrupt ablation feature/permutation snapshot')
    environment_file = directory / 'environment.json'
    if not environment_file.is_file() or sha256_file(environment_file) != batch.get('environment_sha256'):
        raise ValueError('missing or corrupt ablation environment snapshot')
    if digest(batch['source']['files']) != batch['source']['snapshot_sha256']:
        raise ValueError('code snapshot identity mismatch')
    for name, expected_hash in batch['source']['files'].items():
        path = (directory / 'code_snapshot' / name).resolve()
        if not path.is_relative_to((directory / 'code_snapshot').resolve()) or sha256_file(path) != expected_hash:
            raise ValueError('unsafe or corrupt ablation code snapshot')
    by_id = {r['video_id']: r for r in bundle['rows']}
    results = []
    for run in batch['runs']:
        name = run['directory']
        if not isinstance(name, str) or Path(name).name != name or any(c in name for c in '/\\:'):
            raise ValueError('unsafe ablation run directory')
        folder = directory / name
        provenance = read_json(folder / 'provenance.json')
        if (provenance['status'] != 'success' or provenance['run_id'] != name
                or read_json(folder / 'resolved_config.json') != dict(ablation=batch['config'], reference=parent['config'])):
            raise ValueError('failed or mixed ablation run')
        if any(provenance[key] != value for key, value in run.items()):
            raise ValueError('ablation run identity mismatch')
        for key in ('protocol_id', 'feature_identity', 'evaluation_ids', 'evaluation_stage', 'source'):
            if provenance[key] != batch[key]:
                raise ValueError(f'ablation provenance identity mismatch: {key}')
        for filename, key in (('predictions.csv', 'predictions_sha256'), ('metrics.json', 'metrics_sha256'),
                              ('support_set.csv', 'support_csv_sha256'), ('ridge.npz', 'ridge_sha256')):
            if sha256_file(folder / filename) != provenance[key]:
                raise ValueError(f'corrupt ablation artifact: {filename}')
        support = sorted((r for r in supports['records'] if r['k'] == run['k'] and r['support_seed'] == run['support_seed']),
                         key=lambda r: (r['class_id'], r['rank_in_class']))
        if _read_csv(folder / 'support_set.csv') != [{k: str(v) for k, v in r.items()} for r in support]:
            raise ValueError('ablation support CSV differs from frozen set')
        ids, groups = [r['video_id'] for r in support], sorted({r['source_group_id'] for r in support})
        if (provenance['support_ids'] != ids or provenance['support_sha256'] != digest(sorted(ids))
                or provenance['n_support'] != 5 * run['k'] or provenance['n_support_sources'] != len(groups)
                or provenance['support_source_groups'] != groups or provenance['feature_dimension'] != index['feature_dimension']):
            raise ValueError('ablation support/source/dimension mismatch')
        predictions = _read_csv(folder / 'predictions.csv')
        if [r['video_id'] for r in predictions] != batch['evaluation_ids']:
            raise ValueError('ablation prediction coverage mismatch')
        truth = np.array([int(r['true_class_id']) for r in predictions])
        scores = np.array([[float(r[f'score_{c}']) for c in CLASSES] for r in predictions])
        if (not np.isfinite(scores).all() or truth.tolist() != [by_id[v]['class_id'] for v in batch['evaluation_ids']]
                or [int(r['predicted_class_id']) for r in predictions] != scores.argmax(1).tolist()):
            raise ValueError('ablation prediction truth/score/argmax mismatch')
        for row in predictions:
            perturbation = '' if run['shuffle_seed'] is None else str(run['shuffle_seed'])
            if (row['representation'] != run['representation'] or row['shuffle_seed'] != perturbation
                    or row['run_id'] != name or row['evaluation_stage'] != 'development'
                    or int(row['k']) != run['k'] or int(row['support_seed']) != run['support_seed']
                    or row['source_group_id'] != by_id[row['video_id']]['source_group_id']):
                raise ValueError('ablation prediction condition/source/seed mismatch')
        metrics = dict(**classification_metrics(truth, scores), evaluation_stage='development',
                       representation=run['representation'], shuffle_seed=run['shuffle_seed'], n_support=len(ids),
                       n_support_sources=len(groups), feature_dimension=index['feature_dimension'])
        if metrics != read_json(folder / 'metrics.json'):
            raise ValueError('ablation metrics differ from per-video recomputation')
        results.append(dict(**run, metrics=metrics, support_sha256=provenance['support_sha256']))
    return batch, results, baseline


def _statistics(values: list[float], prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {f'{prefix}_mean': float(values.mean()),
            f'{prefix}_std': float(values.std(ddof=1)) if len(values) > 1 else None}


def summarize_ablations(directory: Path, output: Path) -> dict:
    batch, runs, baseline = validate_ablation_runs(directory)
    directory, output = Path(directory), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite nonempty ablation report')
    output.mkdir(parents=True, exist_ok=True)
    all_runs = [dict(**r, shuffle_seed=None, reused_reference=True) for r in baseline]
    all_runs += [dict(**r, reused_reference=False) for r in runs]
    per_run, all_predictions = [], []
    for run in all_runs:
        metrics = run['metrics']
        row = {k: run[k] for k in ('representation', 'shuffle_seed', 'k', 'support_seed', 'support_sha256', 'reused_reference')}
        row.update(feature_condition=batch['feature_condition'], n_evaluated_videos=metrics['n_evaluated'],
                   accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'])
        row.update({f'recall_{name}': metrics['per_class_recall'][i] for i, name in enumerate(CLASSES)})
        per_run.append(row)
        parent = directory / 'reference' if run['reused_reference'] else directory
        for prediction in _read_csv(parent / run['directory'] / 'predictions.csv'):
            all_predictions.append(dict(prediction, shuffle_seed=run['shuffle_seed'],
                                        feature_condition=batch['feature_condition'], reused_reference=run['reused_reference']))
    metrics_names = ['accuracy', 'macro_f1', *[f'recall_{name}' for name in CLASSES]]
    representations = ('V0', 'V1', 'V1_repeat', 'V1_shuffle')
    shots = batch['reference_config']['sampling']['shots']
    seeds = batch['reference_config']['sampling']['support_seeds']
    per_support = []
    for representation in representations:
        for k in shots:
            for seed in seeds:
                part = [r for r in per_run if r['representation'] == representation and r['k'] == k and r['support_seed'] == seed]
                if len({r['support_sha256'] for r in part}) != 1:
                    raise ValueError('unpaired support hashes within perturbation group')
                per_support.append(dict(representation=representation, k=k, support_seed=seed,
                                        feature_condition=batch['feature_condition'], support_sha256=part[0]['support_sha256'],
                                        n_perturbations=len(part), **{m: float(np.mean([r[m] for r in part])) for m in metrics_names}))
    summary = []
    for representation in representations:
        for k in shots:
            part = [r for r in per_support if r['representation'] == representation and r['k'] == k]
            row = dict(representation=representation, k=k, feature_condition=batch['feature_condition'],
                       n_support_sets=len(part), n_unique_support_sets=len({r['support_sha256'] for r in part}),
                       n_perturbations_per_support=part[0]['n_perturbations'], n_evaluated_videos=len(batch['evaluation_ids']))
            for m in metrics_names:
                row.update(_statistics([r[m] for r in part], m))
            summary.append(row)
    indexed = {(r['representation'], r['k'], r['support_seed']): r for r in per_support}
    deltas, paired_summary = [], []
    for left, right in COMPARISONS:
        for k in shots:
            current = []
            for seed in seeds:
                a, b = indexed[left, k, seed], indexed[right, k, seed]
                if a['support_sha256'] != b['support_sha256']:
                    raise ValueError('ablation/baseline support pairing mismatch')
                row = dict(comparison=f'{left}_minus_{right}', left=left, right=right, k=k, support_seed=seed,
                           feature_condition=batch['feature_condition'], support_sha256=a['support_sha256'],
                           **{f'{m}_delta': a[m] - b[m] for m in metrics_names})
                deltas.append(row)
                current.append(row)
            row = dict(comparison=f'{left}_minus_{right}', k=k, feature_condition=batch['feature_condition'], n_pairs=len(current))
            for m in ('accuracy', 'macro_f1'):
                values = [r[f'{m}_delta'] for r in current]
                row.update(_statistics(values, f'{m}_delta'))
                row.update({f'{m}_delta_median': float(np.median(values)), f'{m}_positive_pairs': sum(v > 0 for v in values),
                            f'{m}_tied_pairs': sum(v == 0 for v in values), f'{m}_negative_pairs': sum(v < 0 for v in values)})
            paired_summary.append(row)
    shuffle_deltas, shuffle_summary = [], []
    for perturbation in batch['config']['shuffle']['seeds']:
        for k in shots:
            part = [r for r in per_run if r['representation'] == 'V1_shuffle' and r['shuffle_seed'] == perturbation and r['k'] == k]
            row = dict(representation='V1_shuffle', shuffle_seed=perturbation, k=k,
                       feature_condition=batch['feature_condition'], n_support_sets=len(part))
            for m in metrics_names:
                row.update(_statistics([r[m] for r in part], m))
            shuffle_summary.append(row)
            for r in part:
                a = indexed['V1', k, r['support_seed']]
                shuffle_deltas.append(dict(comparison='V1_minus_V1_shuffle', k=k, support_seed=r['support_seed'],
                                           shuffle_seed=perturbation, feature_condition=batch['feature_condition'],
                                           **{f'{m}_delta': a[m] - r[m] for m in metrics_names}))
    result = dict(protocol_id=batch['protocol_id'], feature_identity=batch['feature_identity'],
                  feature_condition=batch['feature_condition'], evaluation_stage='development', final_test_enabled=False,
                  summary=summary, per_run_metrics=per_run, per_support_metrics=per_support,
                  paired_deltas=deltas, paired_summary=paired_summary, summary_by_shuffle_seed=shuffle_summary,
                  paired_by_shuffle_seed=shuffle_deltas, aggregation=batch['config']['aggregation'])
    write_json(output / 'summary.json', result)
    for filename, rows in (('summary.csv', summary), ('per_run_metrics.csv', per_run), ('per_support_metrics.csv', per_support),
                           ('paired_deltas.csv', deltas), ('paired_summary.csv', paired_summary),
                           ('summary_by_shuffle_seed.csv', shuffle_summary), ('paired_by_shuffle_seed.csv', shuffle_deltas),
                           ('permutation_predictions.csv', all_predictions)):
        write_csv(output / filename, rows, list(rows[0]))
    write_json(output / 'confusion_counts.json', [dict(representation=r['representation'], shuffle_seed=r['shuffle_seed'],
               k=r['k'], support_seed=r['support_seed'], confusion_matrix=r['metrics']['confusion_matrix']) for r in all_runs])
    shutil_source = directory / 'permutations.csv'
    (output / 'permutations.csv').write_bytes(shutil_source.read_bytes())
    _plot(summary, paired_summary, output, batch['feature_condition'])
    _write_report(result, batch, output)
    return result


def _plot(summary: list[dict], paired: list[dict], output: Path, condition: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    context = 'E0 supplementary controls: frozen CLIP' if condition == 'official_openai_clip_vit_b32' else 'PROGRAM VALIDATION: synthetic/unverified features'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, metric in zip(axes, ('accuracy', 'macro_f1')):
        for representation in ('V0', 'V1', 'V1_repeat', 'V1_shuffle'):
            part = [r for r in summary if r['representation'] == representation]
            ax.errorbar([r['k'] for r in part], [100*r[f'{metric}_mean'] for r in part],
                        yerr=[100*r[f'{metric}_std'] for r in part], marker='o', capsize=3, label=representation)
        ax.set(xlabel='Labeled videos per class (K)', ylabel=metric + ' (%)', ylim=(0, 100))
        ax.set_title(context + '\nShuffle: mean within support; bars: support sample SD', fontsize=9)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(output / 'learning_curves.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for left, right in COMPARISONS:
        part = [r for r in paired if r['comparison'] == f'{left}_minus_{right}']
        ax.plot([r['k'] for r in part], [100*r['macro_f1_delta_mean'] for r in part], marker='o', label=f'{left} - {right}')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(xlabel='Labeled videos per class (K)', ylabel='Paired Macro-F1 difference (pp)')
    ax.set_title(context + '\nMean paired differences on the same development videos', fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(output / 'paired_differences.png', dpi=180)
    plt.close(fig)


def _write_report(result: dict, batch: dict, output: Path) -> None:
    official = result['feature_condition'] == 'official_openai_clip_vit_b32'
    title = '# E0第二轮补充消融：重复均值与帧序打乱' if official else '# 仅程序验证：合成／未验证特征，不能作为真实实验结果'
    text = [title, '', '阶段：development；最终测试封存。V0/V1逐预测复核后引用原批次；新增条件沿用相同支持集合、开发视频、帧特征和λ=1。', '',
            'V1_repeat=[全局均值、总体标准差、全局均值、全局均值、全局均值]。V1_shuffle按完整video_id与独立扰动种子产生16帧排列后，计算6/5/5分段V1。两项均为5d，沿用逐块及整体L2、float64统计、float32保存。', '',
            f"扰动种子：{batch['config']['shuffle']['seeds']}。同视频排列跨K及支持集合固定；支持与开发均用同一规则，未选取最佳排列。", '',
            '打乱条件主表先对每个支持集合的各扰动指标取均值，再跨支持集合计算均值和样本SD；这不是平均预测分数后分类的集成。所有扰动单独结果均保存。SD不是独立来源泛化置信区间。', '',
            '| 表示 | K | Accuracy均值 | Macro-F1均值 | F1支持集合SD |', '| --- | --- | --- | --- | --- |']
    for r in result['summary']:
        text.append(f"| {r['representation']} | {r['k']} | {100*r['accuracy_mean']:.2f}% | {100*r['macro_f1_mean']:.2f}% | {100*r['macro_f1_std']:.2f}pp |")
    text += ['', '| 比较（左−右） | K | ΔF1均值 | ΔF1中位数 | 正／平／负支持集合数 |', '| --- | --- | --- | --- | --- |']
    for r in result['paired_summary']:
        text.append(f"| {r['comparison']} | {r['k']} | {100*r['macro_f1_delta_mean']:+.2f}pp | {100*r['macro_f1_delta_median']:+.2f}pp | {r['macro_f1_positive_pairs']}/{r['macro_f1_tied_pairs']}/{r['macro_f1_negative_pairs']} |")
    text += ['', '![学习曲线](learning_curves.png)', '![配对差值](paired_differences.png)', '',
             '解释规则：V1−repeat区分分段内容与重复均值／权重重分配；V1−shuffle诊断原有时间位置及连续分段结构。打乱没有抹去全部变化信息；结果接近不能证明时间无用，结果改善／下降也不单独证明推理机制。此处不是语义组合或溺水检测实验。', '',
             '来源边界继承原批次：用户委托的AI辅助保守来源分组，来源ID互斥不等于完整原始来源独立性认证。固定50段开发视频上的反复评价仍属探索性开发。', '',
             '`per_run_metrics.csv`保存全部引用及新增运行；`per_support_metrics.csv`保存主表聚合单位；`summary_by_shuffle_seed.csv`与`paired_by_shuffle_seed.csv`保留各扰动；`permutation_predictions.csv`含全部逐视频预测，重复评价不计为新增独立视频；`confusion_counts.json`保留原始混淆计数；`permutations.csv`保存实际排列及原始采样帧索引。', '',
             f"协议：`{batch['protocol_id']}`；衍生特征：`{batch['feature_identity']}`。", '']
    (output / 'ablation-results.md').write_text('\n'.join(text), encoding='utf-8')
