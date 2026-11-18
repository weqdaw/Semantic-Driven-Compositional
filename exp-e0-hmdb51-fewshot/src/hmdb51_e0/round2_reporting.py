"""Recompute predictions and validate every pair before reporting V1 minus V0."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np

from .config import CLASSES
from .metrics import classification_metrics
from .pipeline import official_clip_condition
from .provenance import digest, read_json, sha256_file, write_csv, write_json
from .reporting import _read_csv
from .round2_features import expected_entry, validate_frozen_index
from .round2_protocol import group_statistics, load_grouped_bundle, load_round2_supports
from .round2_runner import expected_pairs
from .source_groups import training_rows


def validated_round2_runs(directory: Path, config: dict) -> tuple[dict, list[dict]]:
    directory = Path(directory)
    batch = read_json(directory / 'batch.json')
    bundle = load_grouped_bundle(directory / 'manifest_snapshot.json', config)
    supports = load_round2_supports(directory / 'support_sets_snapshot.json', bundle, config)
    index = read_json(directory / 'feature_index_snapshot.json')
    identity = index['identity']
    validate_frozen_index(index)
    parent_index = read_json(directory / 'parent_feature_index_snapshot.json')
    validate_frozen_index(parent_index)
    if (digest(parent_index['identity']) != identity['parent_feature_identity']
            or parent_index['identity']['protocol_id'] != bundle['parent_manifest']['protocol_id']
            or parent_index['identity']['encoder'] != identity['encoder']
            or parent_index['frozen_verification'] != index['frozen_verification']):
        raise ValueError('round2 parent feature/freeze snapshot mismatch')
    pairs = expected_pairs(config)
    actual = [[r['representation'], r['k'], r['support_seed']] for r in batch['runs']]
    expected_evaluation = sorted(r['video_id'] for r in bundle['rows'] if r['experimental_role'] == 'development')
    if (batch['status'] != 'success' or batch['errors'] or batch['config'] != config
            or batch['evaluation_stage'] != 'development' or batch['protocol_id'] != bundle['protocol_id']
            or batch['parent_protocol_id'] != bundle['parent_manifest']['protocol_id']
            or batch['evaluation_ids'] != expected_evaluation or batch['group_statistics'] != group_statistics(bundle['rows'])
            or batch['expected_pairs'] != pairs or sorted(actual) != sorted(pairs)):
        raise ValueError('round2 batch is incomplete, mixed, or has invalid pairing/group identities')
    if len({r['directory'] for r in batch['runs']}) != len(pairs):
        raise ValueError('duplicate round2 run directory')
    for filename, key in (('manifest_snapshot.json', 'manifest_file_sha256'), ('support_sets_snapshot.json', 'support_file_sha256'),
                          ('feature_index_snapshot.json', 'feature_index_sha256')):
        if sha256_file(directory / filename) != batch[key]:
            raise ValueError('round2 input snapshot hash mismatch')
    condition = 'official_openai_clip_vit_b32' if official_clip_condition(identity) else 'validation_only_unverified_features'
    if (digest(identity) != batch['feature_identity'] or identity['protocol_id'] != bundle['protocol_id']
            or identity['representations'] != config['representations'] or batch['feature_condition'] != condition
            or index['feature_condition'] != condition or index['status'] != 'success' or index.get('errors')
            or index['evaluation_stage'] != 'development' or not index['frozen_verification'].get('unchanged')
            or sha256_file(directory / 'parent_feature_index_snapshot.json') != identity['parent_index_sha256']
            or index['entries'] != [expected_entry(row) for row in training_rows(bundle)]):
        raise ValueError('round2 feature snapshot identity/coverage mismatch')
    by_id = {r['video_id']: r for r in bundle['rows']}
    results = []
    for run in batch['runs']:
        name = run['directory']
        if not isinstance(name, str) or Path(name).name != name or any(char in name for char in '/\\:'):
            raise ValueError('unsafe round2 run directory')
        folder = directory / name
        provenance = read_json(folder / 'provenance.json')
        if (provenance['status'] != 'success' or provenance['run_id'] != name or provenance['directory'] != name
                or read_json(folder / 'resolved_config.json') != config):
            raise ValueError('failed or mixed round2 run')
        for key in ('protocol_id', 'feature_identity', 'evaluation_stage', 'evaluation_ids', 'source'):
            if provenance[key] != batch[key]:
                raise ValueError(f'mixed round2 run identity: {key}')
        if any(provenance[key] != run[key] for key in ('representation', 'k', 'support_seed')):
            raise ValueError('round2 run representation/K/seed mismatch')
        for filename, key in (('predictions.csv', 'predictions_sha256'), ('metrics.json', 'metrics_sha256'), ('support_set.csv', 'support_csv_sha256')):
            if sha256_file(folder / filename) != provenance[key]:
                raise ValueError(f'corrupt round2 artifact: {filename}')
        expected_support = sorted((r for r in supports['records'] if r['k'] == run['k'] and r['support_seed'] == run['support_seed']),
                                  key=lambda r: (r['class_id'], r['rank_in_class']))
        support = _read_csv(folder / 'support_set.csv')
        if support != [{key: str(value) for key, value in r.items()} for r in expected_support]:
            raise ValueError('round2 support rows differ from paired frozen set')
        ids = [r['video_id'] for r in expected_support]
        groups = sorted({r['source_group_id'] for r in expected_support})
        dimension = identity['encoder']['dimension'] * (2 if run['representation'] == 'V0' else 5)
        if (provenance['support_ids'] != ids or provenance['support_sha256'] != digest(sorted(ids))
                or provenance['n_support'] != len(ids) or provenance['support_source_groups'] != groups
                or provenance['n_support_sources'] != len(groups) or provenance['feature_dimension'] != dimension):
            raise ValueError('round2 support/source/feature dimension mismatch')
        predictions = _read_csv(folder / 'predictions.csv')
        if [r['video_id'] for r in predictions] != expected_evaluation:
            raise ValueError('round2 evaluation rows missing/duplicated/mixed')
        truth = np.array([int(r['true_class_id']) for r in predictions])
        scores = np.array([[float(r[f'score_{name}']) for name in CLASSES] for r in predictions])
        claimed = np.array([int(r['predicted_class_id']) for r in predictions])
        if truth.tolist() != [by_id[vid]['class_id'] for vid in expected_evaluation] or not np.array_equal(claimed, scores.argmax(axis=1)):
            raise ValueError('round2 prediction truth/argmax mismatch')
        for row in predictions:
            if (row['representation'] != run['representation'] or row['run_id'] != name or row['evaluation_stage'] != 'development'
                    or int(row['k']) != run['k'] or int(row['support_seed']) != run['support_seed']
                    or row['source_group_id'] != by_id[row['video_id']]['source_group_id']):
                raise ValueError('round2 prediction identity/source mismatch')
        metrics = dict(**classification_metrics(truth, scores), evaluation_stage='development', representation=run['representation'],
                       n_support=len(ids), n_support_sources=len(groups), feature_dimension=dimension)
        if metrics != read_json(folder / 'metrics.json'):
            raise ValueError('round2 saved metrics differ from per-video recomputation')
        results.append({**run, 'metrics': metrics, 'support_sha256': provenance['support_sha256']})
    return batch, results


def summarize_round2(directory: Path, output: Path, config: dict) -> dict:
    batch, runs = validated_round2_runs(directory, config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite an existing nonempty report directory')
    output.mkdir(parents=True, exist_ok=True)
    summary, confusions = [], {}
    for representation in ('V0', 'V1'):
        for k in config['sampling']['shots']:
            part = [r for r in runs if r['representation'] == representation and r['k'] == k]
            row = dict(representation=representation, feature_condition=batch['feature_condition'], k=k, n_support_sets=len(part),
                       n_unique_support_sets=len({r['support_sha256'] for r in part}),
                       n_evaluated_videos=part[0]['metrics']['n_evaluated'])
            for metric in ('accuracy', 'macro_f1'):
                values = [r['metrics'][metric] for r in part]
                row[f'{metric}_mean'], row[f'{metric}_std'] = float(np.mean(values)), float(np.std(values, ddof=1))
            for cls, name in enumerate(CLASSES):
                row[f'recall_{name}_mean'] = float(np.mean([r['metrics']['per_class_recall'][cls] for r in part]))
            summary.append(row)
            confusions[f'{representation}_{k}'] = np.sum([r['metrics']['confusion_matrix'] for r in part], axis=0).tolist()
    indexed = {(r['representation'], r['k'], r['support_seed']): r for r in runs}
    deltas = []
    for k in config['sampling']['shots']:
        for seed in config['sampling']['support_seeds']:
            v0, v1 = indexed['V0', k, seed], indexed['V1', k, seed]
            if v0['support_sha256'] != v1['support_sha256']:
                raise ValueError('V0/V1 support pairing mismatch')
            deltas.append(dict(k=k, support_seed=seed, feature_condition=batch['feature_condition'], support_sha256=v0['support_sha256'],
                               **{f'{metric}_delta': v1['metrics'][metric] - v0['metrics'][metric] for metric in ('accuracy', 'macro_f1')}))
    paired_summary = []
    for k in config['sampling']['shots']:
        row = dict(k=k, feature_condition=batch['feature_condition'], n_pairs=len(config['sampling']['support_seeds']))
        for metric in ('accuracy', 'macro_f1'):
            values = np.array([r[f'{metric}_delta'] for r in deltas if r['k'] == k])
            row.update({f'{metric}_delta_mean': float(values.mean()), f'{metric}_delta_std': float(values.std(ddof=1)),
                        f'{metric}_delta_median': float(np.median(values)), f'{metric}_positive_pairs': int((values > 0).sum()),
                        f'{metric}_tied_pairs': int((values == 0).sum()), f'{metric}_negative_pairs': int((values < 0).sum())})
        paired_summary.append(row)
    result = dict(protocol_id=batch['protocol_id'], feature_identity=batch['feature_identity'], feature_condition=batch['feature_condition'],
                  batch=str(Path(directory).resolve()), summary=summary, paired_deltas=deltas, paired_summary=paired_summary,
                  group_statistics=batch['group_statistics'], confusion_matrix_sum_over_support_sets=confusions)
    write_json(output / 'summary.json', result)
    for filename, rows in (('summary.csv', summary), ('paired_deltas.csv', deltas), ('paired_summary.csv', paired_summary)):
        write_csv(output / filename, rows, list(rows[0]))
    supports = read_json(Path(directory) / 'support_sets_snapshot.json')
    write_csv(output / 'support_source_coverage.csv', supports['source_coverage'], list(supports['source_coverage'][0]))
    for representation in ('V0', 'V1'):
        folder = output / representation
        folder.mkdir()
        _plot_round2([r for r in summary if r['representation'] == representation],
                     {str(k): confusions[f'{representation}_{k}'] for k in config['sampling']['shots']}, folder,
                     representation, batch['feature_condition'])
    official = batch['feature_condition'] == 'official_openai_clip_vit_b32'
    title = '# E0 第二轮：来源分组 V0 / V1 开发实验' if official else '# 仅程序验证：合成或未验证特征，不能作为实验结果'
    text = [title, '', '阶段：development；最终测试封存。', '',
            'V0：均值＋总体标准差，官方512维帧特征对应1024维。V1：增加早/中/晚6/5/5帧均值，对应2560维。两者逐块归一化后整体归一化；共享Z、支持集合、开发视频及λ=1岭分类。', '',
            '| 表示 | K | Accuracy（均值±种子样本标准差） | Macro-F1（均值±种子样本标准差） |',
            '| --- | --- | --- | --- |']
    for row in summary:
        text.append(f"| {row['representation']} | {row['k']} | {100*row['accuracy_mean']:.2f}% ± {100*row['accuracy_std']:.2f}pp | {100*row['macro_f1_mean']:.2f}% ± {100*row['macro_f1_std']:.2f}pp |")
    text += ['', '| K | ΔAccuracy均值 | ΔMacro-F1均值 | ΔMacro-F1中位数 | F1正/平/负种子数 |', '| --- | --- | --- | --- | --- |']
    for row in paired_summary:
        text.append(f"| {row['k']} | {100*row['accuracy_delta_mean']:+.2f}pp | {100*row['macro_f1_delta_mean']:+.2f}pp | {100*row['macro_f1_delta_median']:+.2f}pp | {row['macro_f1_positive_pairs']}/{row['macro_f1_tied_pairs']}/{row['macro_f1_negative_pairs']} |")
    text += ['', '| 类别 | 支持池视频/来源组 | 开发视频/来源组 |', '| --- | --- | --- |']
    for name, stats in batch['group_statistics'].items():
        text.append(f"| {name} | {stats['support_pool']['videos']}/{stats['support_pool']['source_groups']} | {stats['development']['videos']}/{stats['development']['source_groups']} |")
    review = read_json(Path(directory) / 'manifest_snapshot.json')['source_review']
    text += ['', f"来源审核状态：{dict(Counter(r['verification_status'] for r in review))}。审核来源组ID在支持/开发之间无交集；保守映射仍不证明真实来源、人物或摄像头完全独立。", '',
             '![V0学习曲线](V0/learning_curve.png)', '![V1学习曲线](V1/learning_curve.png)',
             '![V0混淆矩阵](V0/confusion_matrices.png)', '![V1混淆矩阵](V1/confusion_matrices.png)', '',
             '所有种子评价相同开发视频，种子波动不是独立数据重复或置信区间。V1−V0衡量分段表示增量；维度与容量同时变化，不能单独据此证明时间推理或语义组合。与第一轮V0的差异还混合了划分变化。', '',
             f"协议：`{batch['protocol_id']}`；特征：`{batch['feature_identity']}`。", '']
    (output / 'round2-results.md').write_text('\n'.join(text), encoding='utf-8')
    return result


def _plot_round2(summary: list[dict], confusions: dict, output: Path, representation: str, condition: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    context = f'E0 round2 {representation}: frozen CLIP' if condition == 'official_openai_clip_vit_b32' else f'PROGRAM VALIDATION {representation}: synthetic/unverified features'
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    shots = [r['k'] for r in summary]
    for metric, label in (('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')):
        ax.errorbar(shots, [100 * r[f'{metric}_mean'] for r in summary],
                    yerr=[100 * r[f'{metric}_std'] for r in summary], marker='o', capsize=4, label=label)
    ax.set(xlabel='Labeled videos per class (K)', ylabel='Development score (%)', xticks=shots, ylim=(0, 100))
    ax.set_title(context + '\nMean +/- sample SD across support sets', fontsize=10)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output / 'learning_curve.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 4.8), squeeze=False, layout='constrained')
    for ax, row in zip(axes[0], summary):
        matrix = np.asarray(confusions[str(row['k'])])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, matrix.max()))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES, xlabel='Predicted', ylabel='True')
        ax.set_title(context + f"\nK={row['k']}; sum of {row['n_support_sets']} support sets", fontsize=8)
    fig.savefig(output / 'confusion_matrices.png', dpi=180)
    plt.close(fig)
