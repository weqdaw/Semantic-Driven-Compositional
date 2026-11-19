import copy
import numpy as np
import pytest


def test_unit_video_features_and_explicit_text_mapping_and_common_geometry():
    from hmdb51_e0.e13_features import derive_bank,prototype_model,DEFAULT_CONFIG,schedule,validate_config
    rng=np.random.default_rng(154); z=rng.normal(size=(20,16,6)); z=(z/np.linalg.norm(z,axis=2,keepdims=True)).astype('f4')
    e=rng.normal(size=(5,6)); e=(e/np.linalg.norm(e,axis=1,keepdims=True)).astype('f4')
    bank=derive_bank(z,e,DEFAULT_CONFIG)
    assert len(schedule(DEFAULT_CONFIG))==81
    assert schedule(DEFAULT_CONFIG)[0]==['T0',None,0,None]
    np.testing.assert_allclose(np.linalg.norm(bank['q'],axis=1),1,atol=1e-12)
    for q,d in zip(bank['Q'],bank['D']):
        np.testing.assert_allclose(q.T@q,np.eye(6),atol=1e-12)
        np.testing.assert_allclose(d@d.T,bank['T']@bank['T'].T,atol=1e-12)
    support=list(range(15)); labels=np.repeat(np.arange(5),3)
    p,_=prototype_model(bank,'P0',None,support,labels)
    mixed,_=prototype_model(bank,'P2',None,support,labels)
    expected=p+bank['T'][[1,2,3,4,0]]; expected/=np.linalg.norm(expected,axis=1,keepdims=True)
    np.testing.assert_allclose(mixed,expected,atol=1e-12)
    changed={k:v.copy() for k,v in bank.items()}; changed['q'][15:]*=-1
    actual,_=prototype_model(changed,'P2',None,support,labels)
    np.testing.assert_array_equal(actual,mixed)
    t,visual=prototype_model(bank,'T0',None,[],np.array([],dtype=int))
    assert visual.shape==(0,6); np.testing.assert_array_equal(t,bank['T'])
    with pytest.raises(ValueError): prototype_model(bank,'T0',None,support,labels)
    bad=copy.deepcopy(DEFAULT_CONFIG); bad['wrong_text_rows']=[0,1,2,3,4]
    with pytest.raises(ValueError): validate_config(bad)


def test_degenerate_or_wrong_supports_and_cancelling_fusion_are_rejected():
    from hmdb51_e0.e13_features import derive_bank,prototype_model,DEFAULT_CONFIG,unit_rows
    with pytest.raises(ValueError): unit_rows(np.zeros((2,3)))
    z=np.ones((15,16,6))/np.sqrt(6); e=np.eye(5,6)
    bank=derive_bank(z,e,DEFAULT_CONFIG); labels=np.repeat(np.arange(5),3)
    with pytest.raises(ValueError,match='support'): prototype_model(bank,'P0',None,list(range(14)),labels[:14])
    p,_=prototype_model(bank,'P0',None,list(range(15)),labels)
    bank['T']=-p
    with pytest.raises(ValueError,match='degenerate'): prototype_model(bank,'P1',None,list(range(15)),labels)


def test_class_text_tokenizer_is_bound_to_original_joint_space_ancestor():
    from hmdb51_e0.e13_workflow import validate_class_text
    from hmdb51_e0.clip_features import CHECKPOINT_SHA256
    image=dict(implementation='openai/CLIP',architecture='ViT-B/32',dimension=512,checkpoint_sha256=CHECKPOINT_SHA256,
               parameter_sha256_before='a'*64,clip_source_sha256='b'*64)
    text={**image,'text_encoder':'native_openai_clip','tokenizer':dict(context_length=77,truncate=False,bpe_sha256='c'*64,tokenizer_source_sha256='d'*64)}
    frozen=dict(unchanged=True,before='a'*64,after='a'*64)
    context=dict(image_encoder=image,reference_text_encoder=copy.deepcopy(text))
    validate_class_text(context,text,frozen)
    for key in ('bpe_sha256','tokenizer_source_sha256'):
        wrong=copy.deepcopy(text);wrong['tokenizer'][key]='e'*64
        with pytest.raises(ValueError,match='tokenizer|ancestor'):validate_class_text(context,wrong,frozen)
