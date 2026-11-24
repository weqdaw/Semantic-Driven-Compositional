"""Reconstruct cache keys, per-key votes and predictions before E1.5 analysis."""
from pathlib import Path

import numpy as np

from .config import CLASSES
from .e1_reporting import stats, delta_summary
from .e15_features import validate_config, schedule, cache_model, score_branches
from .e15_workflow import load_prepared, get_support, validate_support, evidence_payload, RUN_FILES, PRED_FIELDS, OFFICIAL
from .metrics import classification_metrics
from .provenance import digest, read_json, sha256_file, write_csv, write_json
from .reporting import _read_csv

PRIMARY = (('IC1', 'T0'), ('IC1', 'IC2'), ('IC1', 'IC3'))
SECONDARY = (('IC1', 'A2'), ('IC2', 'A2'), ('IC2', 'T0'), ('IC3', 'T0'))
REPRESENTATIONS = ('T0', 'A2', 'IC1', 'IC2', 'IC3')


def continuation_gate(paired_summary, config):
    """Apply the fixed exploratory rule to the three primary comparisons."""
    validate_config(config)
    main = {row['comparison']: row for row in paired_summary if row['primary']}
    gain = main['IC1_minus_T0']
    gate = config['continuation']
    return bool(
        gain['macro_f1_delta_mean'] >= gate['min_macro_f1_gain']
        and gain['macro_f1_positive_pairs'] >= gate['min_positive_supports']
        and main['IC1_minus_IC2']['macro_f1_delta_mean'] > 0
        and main['IC1_minus_IC3']['macro_f1_delta_mean'] > 0
    )


def _validate_arrays(path, expected, description):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected) or any(
            saved[k].dtype != value.dtype or not np.array_equal(saved[k], value) for k, value in expected.items()
        ):
            raise ValueError(f'E1.5 {description} differs from original support reconstruction')


def validate_runs(directory: Path):
    directory = Path(directory)
    batch = read_json(directory/'batch.json')
    validate_config(batch['config'])
    expected = schedule(batch['config'])
    actual = [[r['representation'], r['k'], r['support_seed']] for r in batch['runs']]
    if (batch['status'] != 'success' or batch['errors'] or batch['evaluation_stage'] != 'development'
            or actual != expected or batch['expected_runs'] != expected
            or len({r['directory'] for r in batch['runs']}) != 30 or batch['gradient_fits_executed'] != 0):
        raise ValueError('incomplete/mixed E1.5 condition/support schedule')
    index, context = load_prepared(directory/'feature_index_snapshot.json', current_code=False)
    if (batch['config'] != index['identity']['config'] or batch['protocol_id'] != index['identity']['protocol_id']
            or batch['feature_identity'] != digest(index['identity']) or batch['feature_condition'] != index['feature_condition']
            or batch['evaluation_ids'] != context['batch']['evaluation_ids']
            or sha256_file(directory/'feature_index_snapshot.json') != batch['feature_index_sha256']
            or sha256_file(directory/'environment.json') != batch['environment_sha256']
            or batch['source']['files'] != index['identity']['code']
            or digest(batch['source']['files']) != batch['source']['snapshot_sha256']):
        raise ValueError('E1.5 input/code/environment identity mismatch')
    for name, expected_hash in batch['source']['files'].items():
        path = (directory/'code_snapshot'/name).resolve()
        if not path.is_relative_to((directory/'code_snapshot').resolve()) or sha256_file(path) != expected_hash:
            raise ValueError('unsafe/corrupt E1.5 source snapshot')
    bank = context['bank']
    positions = {r['video_id']: j for j, r in enumerate(index['entries'])}
    by_id = {r['video_id']: r for r in context['bundle']['rows']}
    dev = batch['evaluation_ids']
    queries = bank['q'][[positions[v] for v in dev]]
    truth = np.asarray([by_id[v]['class_id'] for v in dev])
    reference = next(r for r in context['references'] if r['representation'] == 'T0')
    zero_rows = _read_csv(directory/'parent'/reference['prediction_relative_path'])
    if [r['video_id'] for r in zero_rows] != dev:
        raise ValueError('E1.5 fixed T0 reference coverage mismatch')
    zero_scores = np.asarray([[float(r['score_'+c]) for c in CLASSES] for r in zero_rows])
    results = []
    for record in batch['runs']:
        name = record['directory']
        if not isinstance(name, str) or Path(name).name != name or any(c in name for c in '/\\:'):
            raise ValueError('unsafe E1.5 run path')
        folder = directory/name
        prov = read_json(folder/'provenance.json')
        if prov['status'] != 'success' or any(prov[k] != value for k, value in record.items()):
            raise ValueError('failed/mixed E1.5 run identity')
        if (record['rotation_seed'] is not None or record['alpha'] != batch['config']['alpha']
                or record['beta'] != batch['config']['beta']):
            raise ValueError('E1.5 condition/coefficient mismatch')
        for key in ('protocol_id', 'feature_identity', 'evaluation_stage', 'evaluation_ids'):
            if prov[key] != batch[key]:
                raise ValueError('E1.5 provenance mismatch')
        if set(prov['hashes']) != set(RUN_FILES) or any(sha256_file(folder/n) != h for n, h in prov['hashes'].items()):
            raise ValueError('corrupt E1.5 run artifact')
        if read_json(folder/'resolved_config.json') != batch['config']:
            raise ValueError('E1.5 resolved config mismatch')
        support = get_support(context, record['support_seed'])
        ids, groups = validate_support(support, dev, by_id)
        if _read_csv(folder/'support_set.csv') != [{k: str(v) for k, v in r.items()} for r in support]:
            raise ValueError('E1.5 support rows mismatch')
        if (prov['support_ids'] != ids or prov['support_sha256'] != digest(sorted(ids))
                or prov['feature_dimension'] != queries.shape[1] or prov['support_source_groups'] != groups
                or prov['n_cache_keys'] != 15):
            raise ValueError('E1.5 support/source/cache dimension mismatch')
        model = cache_model(bank, record['representation'], [positions[v] for v in ids],
                            np.asarray([r['class_id'] for r in support]), batch['config'])
        _validate_arrays(folder/'cache_model.npz', model, 'cache model')
        expected_scores, text, cache, similarities, affinities, votes = score_branches(queries, model, batch['config'])
        _validate_arrays(folder/'cache_evidence.npz', evidence_payload(dev, ids, similarities, affinities, votes), 'cache evidence/votes')
        rows = _read_csv(folder/'predictions.csv')
        if [r['video_id'] for r in rows] != dev:
            raise ValueError('E1.5 prediction coverage mismatch')
        scores = np.asarray([[float(r['score_'+c]) for c in CLASSES] for r in rows])
        saved_text = np.asarray([[float(r['text_score_'+c]) for c in CLASSES] for r in rows])
        saved_cache = np.asarray([[float(r['cache_score_'+c]) for c in CLASSES] for r in rows])
        if (not all(np.isfinite(value).all() for value in (scores, saved_text, saved_cache))
                or not np.allclose(scores, expected_scores, atol=1e-12, rtol=1e-10)
                or not np.allclose(saved_text, text, atol=1e-12, rtol=1e-10)
                or not np.allclose(saved_text, zero_scores, atol=1e-12, rtol=1e-10)
                or not np.allclose(saved_cache, cache, atol=1e-12, rtol=1e-10)
                or not np.allclose(scores, saved_text+record['alpha']*saved_cache, atol=1e-12, rtol=1e-10)):
            raise ValueError('E1.5 text/cache/final score branch differs from forward computation or T0 anchor')
        for j, row in enumerate(rows):
            if (row['representation'] != record['representation'] or row['rotation_seed'] != '' or row['lambda'] != ''
                    or any(row['auxiliary_score_'+c] != '' for c in CLASSES)
                    or row['run_id'] != name or row['evaluation_stage'] != 'development'
                    or int(row['k']) != 3 or int(row['support_seed']) != record['support_seed']
                    or float(row['alpha']) != record['alpha'] or float(row['beta']) != record['beta']
                    or int(row['true_class_id']) != truth[j] or int(row['predicted_class_id']) != scores[j].argmax()
                    or row['source_group_id'] != by_id[row['video_id']]['source_group_id']):
                raise ValueError('E1.5 prediction truth/condition mismatch')
        metrics = classification_metrics(truth, scores)
        if metrics != read_json(folder/'metrics.json'):
            raise ValueError('E1.5 metrics recomputation mismatch')
        results.append(dict(**record, **{'lambda': None}, metrics=metrics, support_sha256=prov['support_sha256'],
                            prediction_path=folder/'predictions.csv', reused_reference=False))
    for reference in context['references']:
        fields = {k: value for k, value in reference.items() if k != 'prediction_relative_path'}
        results.append(dict(**fields, prediction_path=directory/'parent'/reference['prediction_relative_path'], reused_reference=True))
    return batch, results


def summarize(directory: Path, output: Path):
    directory, output = Path(directory), Path(output)
    batch, runs = validate_runs(directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite E1.5 report')
    output.mkdir(parents=True, exist_ok=True)
    names = ('accuracy', 'macro_f1', *['recall_'+c for c in CLASSES])
    per_run, predictions, row_maps = [], [], {}
    for run in runs:
        row = {key: run[key] for key in ('representation', 'rotation_seed', 'k', 'support_seed', 'alpha', 'beta', 'lambda',
                                         'support_sha256', 'reused_reference')}
        row.update(accuracy=run['metrics']['accuracy'], macro_f1=run['metrics']['macro_f1'],
                   **{'recall_'+c: run['metrics']['per_class_recall'][j] for j, c in enumerate(CLASSES)})
        per_run.append(row)
        rows = _read_csv(run['prediction_path'])
        row_maps[run['representation'], run['support_seed']] = rows
        predictions.extend(dict(p, reused_reference=run['reused_reference']) for p in rows)
    zero = next(r for r in per_run if r['representation'] == 'T0')
    zero_rows = {r['video_id']: r for r in row_maps['T0', None]}
    transitions = []
    for row in per_run:
        if row['representation'] == 'T0':
            continue
        rows = row_maps[row['representation'], row['support_seed']]
        changed = fixed = harmed = both_wrong = 0
        for prediction in rows:
            original = zero_rows[prediction['video_id']]
            if prediction['predicted_class_id'] == original['predicted_class_id']:
                continue
            changed += 1
            correct = prediction['predicted_class_id'] == prediction['true_class_id']
            was_correct = original['predicted_class_id'] == prediction['true_class_id']
            fixed += int(correct and not was_correct)
            harmed += int(was_correct and not correct)
            both_wrong += int(not correct and not was_correct)
        transitions.append(dict(representation=row['representation'], support_seed=row['support_seed'],
                                n_development_videos=len(rows), changed=changed, fixed=fixed, harmed=harmed,
                                both_wrong=both_wrong, net_correct=fixed-harmed))
    summary = [dict(representation='T0', k=0, n_support_sets=0, n_unique_support_sets=0, n_evaluations=1,
                    **{m+'_mean': zero[m] for m in names}, **{m+'_std': None for m in names})]
    per_support, transition_summary = [], []
    for rep in REPRESENTATIONS[1:]:
        part = [r for r in per_run if r['representation'] == rep]
        if len(part) != 10 or {r['support_seed'] for r in part} != set(range(10)):
            raise ValueError('E1.5 support coverage mismatch')
        per_support.extend(dict(representation=rep, k=3, support_seed=r['support_seed'],
                                support_sha256=r['support_sha256'], **{m: r[m] for m in names}) for r in part)
        row = dict(representation=rep, k=3, n_support_sets=10,
                   n_unique_support_sets=len({r['support_sha256'] for r in part}), n_evaluations=10)
        for metric in names:
            row.update(stats([r[metric] for r in part], metric))
        summary.append(row)
        tr = dict(representation=rep, n_support_sets=10, n_development_videos=len(batch['evaluation_ids']))
        temp = [r for r in transitions if r['representation'] == rep]
        for metric in ('changed', 'fixed', 'harmed', 'both_wrong', 'net_correct'):
            tr.update(stats([r[metric] for r in temp], metric))
        transition_summary.append(tr)
    keyed = {(r['representation'], r['support_seed']): r for r in per_support}
    deltas, paired = [], []
    for left, right in (*PRIMARY, *SECONDARY):
        part = []
        for seed in range(10):
            a = keyed[left, seed]
            b = zero if right == 'T0' else keyed[right, seed]
            if right != 'T0' and a['support_sha256'] != b['support_sha256']:
                raise ValueError('E1.5 paired support mismatch')
            row = dict(comparison=f'{left}_minus_{right}', left=left, right=right, k=3, support_seed=seed,
                       primary=(left, right) in PRIMARY, fixed_zero_shot_reference=right == 'T0',
                       **{m+'_delta': a[m]-b[m] for m in names})
            part.append(row)
            deltas.append(row)
        paired.append(dict(**delta_summary(part, f'{left}_minus_{right}', 3, batch['feature_condition']), primary=(left, right) in PRIMARY))
    passed = continuation_gate(paired, batch['config'])
    result = dict(
        batch=str(directory.resolve()), protocol_id=batch['protocol_id'], feature_identity=batch['feature_identity'],
        feature_condition=batch['feature_condition'], evaluation_stage='development', final_test_enabled=False,
        gradient_fits_executed=0, new_evaluations=30, reused_reference_evaluations=11,
        n_development_videos=len(batch['evaluation_ids']), exploratory_continuation_gate=bool(passed),
        alpha=batch['config']['alpha'], beta=batch['config']['beta'], aggregation=batch['config']['aggregation'],
        summary=summary, per_run_metrics=per_run, per_support_metrics=per_support, paired_deltas=deltas,
        paired_summary=paired, prediction_transitions=transitions, transition_summary=transition_summary,
    )
    for name in ('summary', 'per_run_metrics', 'per_support_metrics', 'paired_deltas', 'paired_summary',
                 'prediction_transitions', 'transition_summary'):
        write_csv(output/(name+'.csv'), result[name], list(result[name][0]))
    write_csv(output/'per_video_predictions.csv', predictions, [*PRED_FIELDS, 'reused_reference'])
    write_json(output/'summary.json', result)
    _write_report(output, batch, result)
    return result


def _write_report(output, batch, result):
    lines = [
        '# E1.5 类别文本锚定的实例缓存少样本校正', '',
        '正式CLIP特征 development 结果' if batch['feature_condition'] == OFFICIAL else '仅程序验证：合成特征，不能作为研究成绩', '',
        '30次新增支持评估，引用T0一次和A2十次，共41组。无新编码或梯度训练。',
        f"固定alpha={result['alpha']}，beta={result['beta']}；每类3个支持槽位取平均票数，不作参数搜索。",
        'IC1保留实例；IC2同类槽位全部替换为单位化均值原型；IC3保留IC1键并将缓存标签固定循环右移一类。',
        '最终分数=text_score+alpha*cache_score；cache_score=逐类求和(exp(-beta*(1-cosine))/3)。', '',
        '| 条件 | Macro-F1（%） | 支持集合SD（百分点） | Accuracy（%） |', '|---|---:|---:|---:|',
    ]
    for row in result['summary']:
        sd = '不适用（固定参照）' if row['macro_f1_std'] is None else f"{100*row['macro_f1_std']:.2f}"
        lines.append(f"| {row['representation']} | {100*row['macro_f1_mean']:.2f} | {sd} | {100*row['accuracy_mean']:.2f} |")
    lines += ['', '## 配对比较', '', '| 比较 | 主分析 | Macro-F1差值（百分点，均值 ± SD） | 正/平/负 |', '|---|---|---:|---|']
    for row in result['paired_summary']:
        lines.append(f"| {row['comparison']} | {'是' if row['primary'] else '次要'} | {100*row['macro_f1_delta_mean']:.2f} ± {100*row['macro_f1_delta_std']:.2f} | {row['macro_f1_positive_pairs']}/{row['macro_f1_tied_pairs']}/{row['macro_f1_negative_pairs']} |")
    lines += ['', '## 相对T0的错误转换', '', '| 条件 | 每组平均改对 | 每组平均改错 | 每组平均净增正确 |', '|---|---:|---:|---:|']
    for row in result['transition_summary']:
        lines.append(f"| {row['representation']} | {row['fixed_mean']:.2f} | {row['harmed_mean']:.2f} | {row['net_correct_mean']:.2f} |")
    lines += [
        '', '## 解释边界', '',
        f"探索性继续门槛：{'达到' if result['exploratory_continuation_gate'] else '未达到'}；不是统计显著性或自动确认机制。",
        '门槛：IC1相对T0平均Macro-F1提高至少2个百分点、至少8/10支持集合提高，并且平均优于IC2和IC3。',
        '超过A2或IC2但未超过T0，不能声称少样本支持信息带来整体净增益。',
        'IC1−IC2比较保留实例与单位化均值压缩；压缩同时改变键方向与查询投影，不能排他地认定多模态机制。',
        'IC3只改变缓存标签，原支持视频、真实标签和文本锚保持不变；优于一个固定错配不等于完整语义机制识别。',
        '固定alpha/beta及每类平均票数是本轮方案；alpha=0.1不代表与E1.4分支尺度匹配或最优。阴性结果不否定全部缓存方案。',
        'T0只有一次固定预测；支持SD仅反映10组支持集合变化，重复预测不是独立视频或独立零样本重复。',
        '引用T0/A2的缓存列留空；A2原有lambda与视觉辅助分数保留，不伪造新的缓存票数。',
        '各run/cache_evidence.npz保存查询ID、支持ID及逐支持相似度/亲和度/票数；行列与原身份绑定。',
        '开发集已多轮使用；最终测试封存。未检验概念落地、未见组合或真实溺水工程效果。', '',
    ]
    (output/'e15-results.md').write_text('\n'.join(lines), encoding='utf-8')
