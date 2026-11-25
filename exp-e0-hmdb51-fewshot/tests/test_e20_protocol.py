import copy
import pytest


def test_config_is_fixed_and_review_sampling_is_independent_of_row_order():
    from hmdb51_e0.e20_protocol import DEFAULT_CONFIG, validate_config, review_frame_ids
    rows = [dict(opaque_id=f'f{v}-{t}', anonymous_video_id=f'v{v}') for v in range(10) for t in range(3)]
    selected = review_frame_ids(rows)
    assert selected == review_frame_ids(list(reversed(rows)))
    assert len(selected) == 6 and len({v.split('-')[0] for v in selected}) == 6
    validate_config(DEFAULT_CONFIG)
    bad = copy.deepcopy(DEFAULT_CONFIG)
    bad['final_test_enabled'] = True
    with pytest.raises(ValueError):
        validate_config(bad)


def test_human_forms_reject_identity_edits_partial_fields_and_invalid_times():
    from hmdb51_e0.e20_protocol import validate_judgments, HUMAN_FIELDS
    template = [dict(opaque_id='f1', concept_id='pose01', view='crop', image='images/f1.png',
                     concept_label='双手靠近', description_en='a pose', **{k: '' for k in HUMAN_FIELDS})]
    assert validate_judgments(template, template) == template
    with pytest.raises(ValueError, match='incomplete'):
        validate_judgments(template, template, complete=True)
    row = dict(template[0], judgment='uncertain', reason_code='crop_missing', reviewer='A',
               reviewed_at='2026-11-24T12:00:00+08:00', evidence='hands cropped', elapsed_seconds='2.5')
    assert validate_judgments([row], template, complete=True)[0]['judgment'] == 'uncertain'
    for key, value in [('opaque_id', 'other'), ('reviewed_at', '2026-11-24T12:00:00'),
                       ('elapsed_seconds', 'nan'), ('reason_code', 'invented')]:
        bad = dict(row, **{key: value})
        with pytest.raises(ValueError):
            validate_judgments([bad], template)
    with pytest.raises(ValueError):
        validate_judgments([dict(template[0], reviewer='A')], template)
