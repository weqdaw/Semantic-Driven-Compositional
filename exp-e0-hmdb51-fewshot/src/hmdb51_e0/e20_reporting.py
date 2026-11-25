"""Read-only progress and concept-grounding reports from frozen human forms."""
from pathlib import Path
import shutil

from .e1_experiment import OFFICIAL
from .e20_metrics import analyze_rows
from .e20_workflow import load_packet
from .e11_workflow import tree_hashes
from .provenance import digest, now, sha256_file, source_identity, write_csv, write_json
from .config import ROOT


def summarize(packet, output):
    context = load_packet(packet)
    manifest, packet = context['manifest'], context['packet']
    if any(sha256_file(packet/manifest['forms'][view]) != expected for view, expected in context['form_hashes'].items()):
        raise ValueError('E2.0 judgments changed before reporting')
    packet_hashes = tree_hashes(packet)
    output = Path(output).resolve()
    if output.is_relative_to(packet) or packet.is_relative_to(output):
        raise ValueError('report output overlaps protected E2.0 packet')
    for protected in (Path(manifest['e11_batch']).resolve(), Path(manifest['audit_directory']).resolve()):
        if output.is_relative_to(protected):
            raise ValueError('report output overlaps protected original evidence')
    if output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite E2.0 report')
    output.mkdir(parents=True, exist_ok=True)
    forms, locks = context['forms'], manifest['locks']
    ready = all(v in locks for v in ('crop', 'original'))
    independent = all(v in locks for v in ('review-crop', 'review-original'))
    adjudicated = 'adjudication' in locks
    progress = [dict(view=v, n_rows=len(rows), n_judged=sum(bool(r['judgment'].strip()) for r in rows),
                     n_blank=sum(not r['judgment'].strip() for r in rows), frozen=v in locks,
                     elapsed_seconds=sum(float(r['elapsed_seconds']) for r in rows if r['judgment'].strip()))
                for v, rows in forms.items()]
    result = dict(packet=str(packet), evaluation_stage='development', final_test_enabled=False,
        feature_condition=manifest['feature_condition'], protocol_id=manifest['protocol_id'],
        n_videos=10, n_frames=30, n_frame_concept_pairs=180,
        n_source_groups=len({r['source_group_id'] for r in context['mapping']}),
        analysis_ready=ready, independent_review_completed=independent, adjudication_completed=adjudicated,
        progress=progress, human_time_seconds=sum(r['elapsed_seconds'] for r in progress),
        concept_coverage=[], crop_retention=[], concept_alignment=[], matched_scores=[], agreement_rows=[],
        n_usable_concepts=0, exploratory_continuation_gate=False, created_at=now())
    if ready:
        secondary = [r for view in ('review-crop', 'review-original') if view in locks for r in forms[view]]
        # Unfrozen adjudications never silently become analysis labels.
        adjudication = forms['adjudication'] if adjudicated else []
        result.update(analyze_rows(context['mapping'], context['scores'], forms['crop'], forms['original'],
                                   secondary, adjudication, independent and adjudicated, manifest['config']))
    for view, path in manifest['forms'].items():
        target = output/'judgment_snapshots'/f'{view}.csv'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(packet/path, target)
    for name in ('concept_coverage', 'crop_retention', 'concept_alignment', 'matched_scores', 'agreement_rows', 'progress'):
        rows = result[name]
        if rows:
            write_csv(output/(name+'.csv'), rows, list(rows[0]))
    if ready:
        errors = ['# E2.0 概念分数诊断案例', '', '以下仅比较人工crop判断与原裁剪模型分数，不推断原图模型性能。', '']
        for concept in manifest['config']['concept_ids']:
            candidates = [r for r in result['matched_scores'] if r['concept_id'] == concept and r['judgment'] in ('present', 'absent')]
            false_high = sorted((r for r in candidates if r['judgment'] == 'absent'), key=lambda r: -r['correct_score'])[:3]
            false_low = sorted((r for r in candidates if r['judgment'] == 'present'), key=lambda r: r['correct_score'])[:3]
            errors += [f'## {concept}', '', '| 匿名帧 | 人工判断 | 正确分数 | 错配分数 |', '|---|---|---:|---:|']
            for row in [*false_high, *false_low]:
                errors.append(f"| {row['opaque_id']} | {row['judgment']} | {row['correct_score']:.6f} | {row['wrong_score']:.6f} |")
            errors.append('')
        (output/'error_cases.md').write_text('\n'.join(errors), encoding='utf-8')
    if tree_hashes(packet) != packet_hashes:
        raise ValueError('E2.0 packet changed during reporting')
    # Snapshot the exact packet manifest and analysis implementation used.
    shutil.copyfile(packet/'input_manifest.json', output/'input_manifest.snapshot.json')
    write_json(output/'provenance.json', dict(packet_manifest_sha256=sha256_file(packet/'input_manifest.json'),
        source=source_identity(ROOT), config=manifest['config'], gradient_fits_executed=0,
        image_encoding_executed=False, video_decoding_executed=False))
    write_json(output/'summary.json', result)
    _write_report(output, result)
    return result


def _value(value):
    return 'NA' if value is None else f'{value:.3f}'


def _write_report(output, result):
    lines = ['# E2.0 概念可见性与视觉落地诊断', '',
        '正式CLIP特征 development 诊断' if result['feature_condition'] == OFFICIAL else '仅程序验证：合成材料，不是人工或模型研究成绩', '',
        '10视频、30帧、6概念、180对；相关帧／概念不当作独立样本。',
        '状态：主审核双视图已冻结。' if result['analysis_ready'] else '状态：人工审核未完成或尚未冻结；只报告进度，不生成分数对齐结论。',
        f"独立复核完成：{result['independent_review_completed']}；分歧裁决完成：{result['adjudication_completed']}。", '',
        '| 阶段 | 已判断 | 空白 | 已冻结 |', '|---|---:|---:|---|']
    for row in result['progress']:
        lines.append(f"| {row['view']} | {row['n_judged']} | {row['n_blank']} | {row['frozen']} |")
    if result['analysis_ready']:
        lines += ['', '## 每概念分数对齐', '', '| 概念 | present/absent/uncertain | 正确AUROC | 错配AUROC | 差值 | AP | 正例比例 | 覆盖充足 |', '|---|---|---:|---:|---:|---:|---:|---|']
        for row in result['concept_alignment']:
            lines.append(f"| {row['concept_id']} | {row['n_present']}/{row['n_absent']}/{row['n_uncertain']} | {_value(row['correct_auroc'])} | {_value(row['wrong_auroc'])} | {_value(row['auroc_delta'])} | {_value(row['correct_average_precision'])} | {_value(row['prevalence'])} | {row['coverage_sufficient']} |")
        lines += ['', '## 裁剪保留', '', '| 概念 | 原图present | 裁剪仍present | 保留率 | 原图present而裁剪uncertain |', '|---|---:|---:|---:|---:|']
        for row in result['crop_retention']:
            lines.append(f"| {row['concept_id']} | {row['n_original_present']} | {row['n_retained_present']} | {_value(row['retention_rate'])} | {row['original_present_crop_uncertain']} |")
    lines += ['', '## 判断与边界', '',
        f"探索性继续门槛：{'达到' if result['exploratory_continuation_gate'] else '尚未达到'}；不是统计显著性或自动确认机制。",
        f"初步可用概念数：{result['n_usable_concepts']}；人工记录时间合计{result['human_time_seconds']:.1f}秒（包含复核／裁决，属于新增监督成本）。",
        '正确分数只与crop人工标签配对；original只诊断可见性／裁剪保留，不存在新的原图模型性能对照。',
        '只有主审核双视图都冻结后才显示分数；未冻结裁决不进入指标，原始双方记录始终保留。',
        '复核者名称区分仅为记录约束，不能证明实际人员独立或完全双盲；私有目录不是操作系统访问控制。',
        '不确定行单列；单类AUC/AP为NA，正负覆盖不足不进入推进判据。区间按来源组重采样，不充当独立显著性证据。',
        '未验证动作分类提升、时间关系、未见组合或真实溺水检测。最终测试封存。', '']
    (output/'e20-results.md').write_text('\n'.join(lines), encoding='utf-8')
