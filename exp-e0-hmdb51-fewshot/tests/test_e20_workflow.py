import copy
import csv
from pathlib import Path

import pytest
from PIL import Image

from hmdb51_e0.provenance import digest, write_csv, write_json, read_json, sha256_file


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def source(tmp_path, monkeypatch):
    from hmdb51_e0 import e20_workflow as workflow
    from hmdb51_e0.e20_protocol import CONCEPTS
    from hmdb51_e0.e11_workflow import tree_hashes
    packet, batch = tmp_path/'source', tmp_path/'e11'
    packet.mkdir()
    batch.mkdir()
    write_json(batch/'batch.json', {'fixture': True})
    frames, scores, template = [], [], []
    groups = {}
    for v in range(10):
        vid = f'clap/secret_movie_{v}.avi'
        groups[vid] = f'g{v}'
        for t in (2, 7, 13):
            original, crop = f'images/{v}-{t}-original.png', f'images/{v}-{t}-crop.png'
            for filename, size in ((original, (250, 224)), (crop, (224, 224))):
                (packet/filename).parent.mkdir(exist_ok=True)
                Image.new('RGB', size, color=(v*10, t, 1)).save(packet/filename)
            frames.append(dict(video_id=vid, video_sha256=digest(vid), frame_position_zero_based=str(t),
                               frame_index=str(t), original_image=original, clip_image=crop))
            for cid in CONCEPTS:
                template.append(dict(video_id=vid, frame_position_zero_based=str(t), frame_index=str(t), concept_id=cid,
                                     concept_label='姿态'+cid, description_en='a person in a pose'))
                scores.append(dict(video_id=vid, frame_position_zero_based=str(t), frame_index=str(t), concept_id=cid,
                                   cosine_score=str({2: .9, 7: .1, 13: .8}[t])))
    write_csv(packet/'judgments.template.csv', template, list(template[0]))
    write_csv(packet/'model_scores.after_review.csv', scores, list(scores[0]))
    write_csv(packet/'frames.csv', frames, list(frames[0]))
    context = dict(protocol_id='fixture', feature_condition='validation_only_unverified_features', frames=frames,
                   template=template, scores=scores, groups=groups, audit_directory=packet, e11_batch=batch)
    def fake_source(e11, audit):
        assert Path(e11) == batch and Path(audit) == packet
        return dict(context, audit_hashes=tree_hashes(packet), batch_hashes=tree_hashes(batch))
    monkeypatch.setattr(workflow, 'source_context', fake_source)
    return context, dict(dataset_root=tmp_path/'data', feature_cache_root=tmp_path/'old',
                         report_root=tmp_path/'reports'), fake_source


def fill(path, reviewer, *, changed=False):
    values = rows(path)
    for i, row in enumerate(values):
        row.update(judgment='absent' if i % 3 == 0 else 'present', reason_code='clear', reviewer=reviewer,
                   reviewed_at='2026-11-24T12:00:00+08:00', evidence='synthetic visible evidence', elapsed_seconds='1')
    if changed and values:
        values[0]['judgment'] = 'present' if values[0]['judgment'] == 'absent' else 'absent'
    write_csv(path, values, list(values[0]))


def test_blank_packet_preserves_sources_and_only_opens_original_after_crop_freeze(source, tmp_path):
    from hmdb51_e0.e20_workflow import prepare_e20, freeze_view, load_packet
    from hmdb51_e0.e20_reporting import summarize
    from hmdb51_e0.e20_protocol import DEFAULT_CONFIG
    from hmdb51_e0.e11_workflow import tree_hashes
    context, paths, _ = source
    before = tree_hashes(context['audit_directory'])
    packet = prepare_e20(context['e11_batch'], context['audit_directory'], DEFAULT_CONFIG, paths, tmp_path/'out')
    assert tree_hashes(context['audit_directory']) == before
    crop = packet/'crop/judgments.csv'
    assert len(rows(crop)) == 180
    assert len(list((packet/'crop/images').glob('*.png'))) == 30
    public = crop.read_text(encoding='utf-8')+(packet/'crop/index.html').read_text(encoding='utf-8')
    assert 'secret_movie' not in public and '.avi' not in public and 'cosine_score' not in public
    assert not (packet/'original').exists()
    result = summarize(packet, tmp_path/'progress')
    assert result['analysis_ready'] is False and result['concept_alignment'] == []
    assert not (tmp_path/'progress/matched_scores.csv').exists()
    with pytest.raises(ValueError, match='incomplete'):
        freeze_view(packet, 'crop')
    with pytest.raises(ValueError, match='order|first|unavailable'):
        freeze_view(packet, 'original')
    fill(crop, 'primary')
    freeze_view(packet, 'crop')
    assert (packet/'original/index.html').exists()
    assert len(rows(packet/'original/judgments.csv')) == 180
    changed = rows(crop)
    changed[0]['evidence'] = 'edited after freezing'
    write_csv(crop, changed, list(changed[0]))
    with pytest.raises(ValueError, match='frozen|lock'):
        load_packet(packet)


def test_full_review_disagreements_metrics_and_rehashed_tampering(source, tmp_path):
    from hmdb51_e0.e20_workflow import prepare_e20, freeze_view, load_packet
    from hmdb51_e0.e20_reporting import summarize
    from hmdb51_e0.e20_protocol import DEFAULT_CONFIG
    context, paths, _ = source
    packet = prepare_e20(context['e11_batch'], context['audit_directory'], DEFAULT_CONFIG, paths, tmp_path/'out')
    for view in ('crop', 'original'):
        fill(packet/view/'judgments.csv', 'primary')
        freeze_view(packet, view)
    single = summarize(packet, tmp_path/'single')
    assert single['analysis_ready'] and not single['independent_review_completed']
    assert single['exploratory_continuation_gate'] is False
    assert len(single['concept_coverage']) == 12 and len(single['concept_alignment']) == 6
    assert len(single['matched_scores']) == 180
    for view in ('review-crop', 'review-original'):
        path = packet/view/'judgments.csv'
        assert len(rows(path)) == 36
        fill(path, 'primary')
        with pytest.raises(ValueError, match='independent|reviewer'):
            freeze_view(packet, view)
        fill(path, 'second', changed=True)
        freeze_view(packet, view)
    adjudication = packet/'adjudication/judgments.csv'
    assert len(rows(adjudication)) > 0
    fill(adjudication, 'adjudicator')
    freeze_view(packet, 'adjudication')
    result = summarize(packet, tmp_path/'report')
    assert result['independent_review_completed'] and result['adjudication_completed']
    assert len(result['agreement_rows']) == 72
    assert (tmp_path/'report/e20-results.md').exists()
    assert '仅程序验证' in (tmp_path/'report/e20-results.md').read_text(encoding='utf-8')
    # Even updating an immutable file hash cannot turn different model scores
    # into the original validated source scores.
    scorefile = packet/'private/model_scores.csv'
    scores = rows(scorefile)
    scores[0]['cosine_score'] = '0.123456'
    write_csv(scorefile, scores, list(scores[0]))
    manifest = read_json(packet/'input_manifest.json')
    manifest['immutable']['private/model_scores.csv'] = sha256_file(scorefile)
    write_json(packet/'input_manifest.json', manifest)
    with pytest.raises(ValueError, match='score|reconstruct'):
        load_packet(packet)


def test_output_protection_and_cli_synthetic_final_test_guards(source, tmp_path, capsys):
    from hmdb51_e0.e20_cli import main, build_parser
    from hmdb51_e0.e20_workflow import prepare_e20
    from hmdb51_e0.e20_protocol import DEFAULT_CONFIG
    context, paths, _ = source
    with pytest.raises(ValueError, match='protected|overlap'):
        prepare_e20(context['e11_batch'], context['audit_directory'], DEFAULT_CONFIG, paths, context['audit_directory']/'bad')
    import yaml
    values = dict(dataset_root=str(paths['dataset_root']), feature_cache_root=str(paths['feature_cache_root']),
                  annotations_root=None, weights_file=None, run_root=str(tmp_path/'runs'), report_root=str(tmp_path/'reports'))
    pathsfile = tmp_path/'paths.yaml'
    pathsfile.write_text(yaml.safe_dump(values))
    assert main(['check', '--e11-batch', str(context['e11_batch']), '--audit-directory', str(context['audit_directory']),
                 '--paths', str(pathsfile)]) == 2
    assert 'official' in capsys.readouterr().err
    assert main(['prepare', '--e11-batch', str(context['e11_batch']), '--audit-directory', str(context['audit_directory']),
                 '--paths', str(pathsfile), '--output', str(tmp_path/'blocked')]) == 2
    assert 'official' in capsys.readouterr().err and not (tmp_path/'blocked').exists()
    packet = prepare_e20(context['e11_batch'], context['audit_directory'], DEFAULT_CONFIG, paths, tmp_path/'synthetic')
    for command in ('status', 'freeze', 'summarize'):
        args = [command, '--packet', str(packet), '--paths', str(pathsfile)]
        if command == 'freeze':
            args += ['--view', 'crop']
        if command == 'summarize':
            args += ['--output', str(tmp_path/'blocked-report')]
        assert main(args) == 2 and 'official' in capsys.readouterr().err
    assert not (tmp_path/'blocked-report').exists()
    with pytest.raises(SystemExit):
        build_parser().parse_args(['prepare', '--e11-batch', 'x', '--audit-directory', 'x', '--stage', 'final_test'])
    assert not (tmp_path/'reports').exists()


def test_freeze_rejects_a_csv_changed_after_validation_before_opening_originals(source, tmp_path, monkeypatch):
    from hmdb51_e0 import e20_workflow as workflow
    from hmdb51_e0.e20_protocol import DEFAULT_CONFIG
    context, paths, _ = source
    packet = workflow.prepare_e20(context['e11_batch'], context['audit_directory'], DEFAULT_CONFIG, paths, tmp_path/'out')
    path = packet/'crop/judgments.csv'
    fill(path, 'primary')
    actual = workflow.load_packet
    def concurrent_edit(*args, **kwargs):
        loaded = actual(*args, **kwargs)
        edited = rows(path)
        edited[0]['evidence'] = 'changed while freezing'
        write_csv(path, edited, list(edited[0]))
        return loaded
    monkeypatch.setattr(workflow, 'load_packet', concurrent_edit)
    with pytest.raises(ValueError, match='changed|concurrent'):
        workflow.freeze_view(packet, 'crop')
    assert not (packet/'original').exists()
