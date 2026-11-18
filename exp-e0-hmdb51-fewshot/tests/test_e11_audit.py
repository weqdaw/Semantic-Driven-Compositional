import copy
import numpy as np
import pytest
from PIL import Image
from hmdb51_e0.e11_features import DEFAULT_CONFIG


def native_encoder():
    import importlib.metadata
    from torchvision.transforms import Compose,Resize,CenterCrop,InterpolationMode,ToTensor,Normalize
    from hmdb51_e0.clip_features import preprocessing_identity
    from clip.clip import _convert_image_to_rgb
    transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),_convert_image_to_rgb,
                       ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    return {**{k:importlib.metadata.version(k) for k in ('pillow','torchvision')},'preprocess':preprocessing_identity(transform)}


def test_audit_selection_is_frozen_and_native_view_matches_actual_preprocess():
    from hmdb51_e0.e11_audit import select_videos,native_crop
    from torchvision.transforms import Resize,CenterCrop,InterpolationMode
    ids=[f'walk/{i}.avi' for i in range(50)]
    assert select_videos(ids,DEFAULT_CONFIG)==select_videos(list(reversed(ids)),DEFAULT_CONFIG)
    assert len(select_videos(ids,DEFAULT_CONFIG))==10
    rgb=np.random.default_rng(12).integers(0,256,size=(91,171,3),dtype=np.uint8)
    expected=CenterCrop(224)(Resize(224,interpolation=InterpolationMode.BICUBIC)(Image.fromarray(rgb))).convert('RGB')
    np.testing.assert_array_equal(np.asarray(native_crop(rgb,native_encoder())),np.asarray(expected))
    bad=native_encoder(); bad['preprocess'][0]['size']=256
    with pytest.raises(ValueError,match='preprocess|crop'): native_crop(rgb,bad)


def test_export_uses_only_frozen_dev_frames_and_blank_human_judgments(tmp_path,monkeypatch):
    from hmdb51_e0 import e11_audit as audit
    from hmdb51_e0.provenance import sha256_file,read_json
    from hmdb51_e0.video_io import decoder_identity,midpoint_indices
    from test_video_integration import make_video
    import csv
    root=tmp_path/'videos'; records=[]; entries=[]; ids=[]
    for i in range(10):
        vid=f'walk/{i}.avi'; path=root/vid; make_video(path,17,40+i)
        ids.append(vid); h=sha256_file(path)
        records.append(dict(video_id=vid,relative_path=vid,sha256=h,frame_count=17))
        entries.append(dict(video_id=vid,video_sha256=h,frame_indices=midpoint_indices(17).tolist()))
    # A sealed video that cannot be decoded must never be touched.
    sealed=root/'walk/sealed.avi'; sealed.write_bytes(b'not a video')
    records.append(dict(video_id='walk/sealed.avi',relative_path='walk/sealed.avi',sha256=sha256_file(sealed),frame_count=17))
    concepts=[dict(id=c,label_zh=c,description_en='a pose',visibility_note='visible') for c in DEFAULT_CONFIG['audit']['concept_ids']]
    parent=tmp_path/'parent'; parent.mkdir()
    from hmdb51_e0.provenance import write_json
    write_json(parent/'concept_catalog.json',{'concepts':concepts})
    index={'entries':entries,'identity':{'encoder':native_encoder()}}
    bundle={'rows':records,'parent_manifest':{'decoder':decoder_identity()}}
    context=({'evaluation_ids':ids,'protocol_id':'fixture','feature_condition':'validation_only_unverified_features'},[],[],index,
             {'C':np.zeros((10,16,6),dtype='f4')},bundle,{})
    monkeypatch.setattr(audit,'parent_context',lambda p:context)
    paths={'dataset_root':root,'feature_cache_root':tmp_path/'legacy'}
    output=audit.export_audit(parent,DEFAULT_CONFIG,paths,tmp_path/'audit')
    rows=list(csv.DictReader((output/'judgments.csv').open(encoding='utf-8',newline='')))
    assert len(rows)==180 and all(r['judgment']=='' for r in rows)
    assert all('score' not in r and 'true_class_id' not in r for r in rows)
    assert len(list((output/'images').glob('*.png')))==60
    manifest=read_json(output/'audit_manifest.json')
    assert manifest['human_review_completed'] is False
    assert set(manifest['selected_video_ids'])==set(ids)
    assert {int(r['frame_index']) for r in rows}==set(midpoint_indices(17)[[2,7,13]])
    # A changed input must fail rather than silently generate unrelated frames.
    (root/ids[0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash|changed'):
        audit.export_audit(parent,DEFAULT_CONFIG,paths,tmp_path/'other_audit')
