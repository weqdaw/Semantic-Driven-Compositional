import copy
import numpy as np
import pytest
from PIL import Image
from hmdb51_e0.e20r2_protocol import (DEFAULT_CONFIG,validate_config,target_template,validate_targets,
    roi_bounds,roi_image,transformed_box,eligible,calibration_ids)

def sample():
    m=[dict(opaque_id='F0001',anonymous_video_id='V001',decoded_frame_index=2)]
    t=target_template(m,{'F0001':(400,200)})
    r=dict(t[0],person_status='single',target_status='selected',x1='100',y1='20',x2='180',y2='180',
        shoulders_visible='yes',hands_visible='yes',waist_visible='yes',target_description='person blue',
        reviewer='AI',reviewer_origin='ai',reviewed_at='2026-11-24T21:00:00+08:00',evidence='synthetic fixture',elapsed_seconds='0')
    return r,t

def test_roi_geometry_and_validity_with_no_fabricated_missing_target():
    r,t=sample();assert validate_targets([r],t,True)==[r] and eligible(r)
    assert roi_bounds(r)==(92,4,188,196)
    image=Image.fromarray(np.zeros((200,400,3),dtype='uint8'));assert roi_image(image,r).size==(224,224)
    assert np.allclose(transformed_box(r,'native'),(0,22.4,89.6,201.6))
    missing=dict(r,target_status='absent',person_status='absent',x1='',x2='',y1='',y2='')
    validate_targets([missing],t,True);assert roi_image(image,missing) is None and not eligible(missing)
    with pytest.raises(ValueError):validate_targets([dict(missing,x1='1')],t,True)

def test_sealed_config_and_box_identity_guards():
    c=copy.deepcopy(DEFAULT_CONFIG);c['final_test_enabled']=True
    with pytest.raises(ValueError):validate_config(c)
    r,t=sample()
    for changed in (dict(r,x2='401'),dict(r,x1=r['x2']),dict(r,reviewer_origin=''),dict(r,target_id='other')):
        with pytest.raises(ValueError):validate_targets([changed],t,True)
    t2=t+[dict(t[0],opaque_id='F0002',target_id='V001-F0002')]
    r2=dict(r,opaque_id='F0002',target_id='V001-F0002',target_description='different frame person')
    validate_targets([r,r2],t2,True)
    with pytest.raises(ValueError,match='identity'):validate_targets([r,dict(r2,target_id=r['target_id'])],t2,True)

def test_calibration_subset_spans_distinct_videos():
    m=[dict(opaque_id=f'F{i:04d}',anonymous_video_id=f'V{i//3:03d}',decoded_frame_index=i%3) for i in range(60)]
    ids=calibration_ids(m);assert len(ids)==12 and ids==calibration_ids(m[::-1])
    assert len({r['anonymous_video_id'] for r in m if r['opaque_id'] in ids})==12
