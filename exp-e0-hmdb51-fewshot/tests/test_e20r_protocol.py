import copy
import numpy as np
import pytest
from PIL import Image


def candidates():
    return [dict(video_id=f'{c}/{i}.avi',class_id=c,source_group_id=f'g{c}-{i//2}',experimental_role='support_pool')
            for c in range(5) for i in range(12)]


def test_selection_is_balanced_source_capped_and_ignores_input_order():
    from hmdb51_e0.e20r_protocol import select_representative,DEFAULT_CONFIG,validate_config
    rows=candidates();selected=select_representative(rows,{'excluded'})
    assert selected==select_representative(rows[::-1],{'excluded'})
    assert len(selected)==20 and all(sum(r['class_id']==c for r in selected)==4 for c in range(5))
    assert len({r['source_group_id'] for r in selected})>=10
    assert max(sum(x['source_group_id']==r['source_group_id'] for x in selected) for r in selected)<=2
    bad=copy.deepcopy(DEFAULT_CONFIG);bad['final_test_enabled']=True
    with pytest.raises(ValueError):validate_config(bad)
    rows[0]['experimental_role']='final_test'
    assert all(r['video_id']!=rows[0]['video_id'] for r in select_representative(rows,set()))
    with pytest.raises(ValueError):select_representative(rows[:3],set())


def test_letterbox_preserves_content_and_uses_fixed_padding_without_stretching():
    from hmdb51_e0.e20r_media import letterbox
    image=Image.new('RGB',(400,200),(255,0,0));out=np.asarray(letterbox(image))
    assert out.shape==(224,224,3)
    np.testing.assert_array_equal(out[0,0],[123,117,104])
    np.testing.assert_array_equal(out[56,0],[255,0,0])
    np.testing.assert_array_equal(out[167,223],[255,0,0])
    np.testing.assert_array_equal(out[168,0],[123,117,104])


def test_ai_origin_is_explicit_and_cannot_count_as_independent_human():
    from hmdb51_e0.e20r_protocol import validate_form,HUMAN_FIELDS,fully_human
    template=[dict(opaque_id='f1',concept_id='pose01',view='native',image='images/f1.png',concept_label='姿态',description_en='a pose',**{k:'' for k in HUMAN_FIELDS})]
    row=dict(template[0],judgment='present',reason_code='clear',reviewer='AI',reviewed_at='2026-11-24T12:00:00+08:00',evidence='AI provisional',elapsed_seconds='0',reviewer_origin='ai')
    assert validate_form([row],template,complete=True)==[row]
    assert not fully_human([[row]])
    row['reviewer_origin']='human';assert fully_human([[row]])
    row['reviewer_origin']=''
    with pytest.raises(ValueError):validate_form([row],template,complete=True)


def test_common_input_comparison_excludes_uncertain_and_disagreeing_truth():
    from hmdb51_e0.e20r_reporting import paired_input_metrics
    result=paired_input_metrics([
        dict(native_label='present',letterbox_label='present',native_score=.2,letterbox_score=.9,source_group_id='a'),
        dict(native_label='absent',letterbox_label='absent',native_score=.8,letterbox_score=.1,source_group_id='b'),
        dict(native_label='uncertain',letterbox_label='present',native_score=1.,letterbox_score=0.,source_group_id='c'),
        dict(native_label='present',letterbox_label='absent',native_score=0.,letterbox_score=1.,source_group_id='d')])
    assert result['n_common_consistent']==2 and result['n_conflicting_labels']==1
    assert result['native_auroc_common']==0 and result['letterbox_auroc_common']==1
    assert result['letterbox_minus_native_common']==1
    assert result['n_letterbox_newly_determinable']==1


def test_usable_concept_requires_coverage_and_boundary_tolerance():
    from hmdb51_e0.e20r_reporting import usable_concept
    row=dict(determinable_fraction=.70,n_present=5,n_absent=5,n_present_videos=3,n_absent_videos=3,
             correct_auroc=.70,auroc_delta=.05-1e-14)
    assert usable_concept(row)
    for key,value in [('n_present',4),('n_absent_videos',2),('determinable_fraction',.69),('correct_auroc',None),('auroc_delta',.049)]:
        assert not usable_concept(dict(row,**{key:value}))


def test_adjudication_displays_each_disputed_input_view():
    from hmdb51_e0.e20r_workflow import _html
    rows=[dict(opaque_id='F0001',concept_id='pose01',description_en='a pose',operational_rule='visible evidence',
               view=view,image=f'../{view}/images/F0001.png') for view in ('native','letterbox')]
    page=_html('adjudication',rows)
    assert page.count('<img ')==2
    assert '../native/images/F0001.png' in page and '../letterbox/images/F0001.png' in page
