import copy
import numpy as np
import pytest


def test_text_anchor_is_fixed_and_visual_mapping_rotation_and_raw_score_sum_are_exact():
    from hmdb51_e0.e14_features import score_model,score_branches,DEFAULT_CONFIG,schedule,validate_config
    from hmdb51_e0.e13_features import derive_bank,DEFAULT_CONFIG as C13,prototype_model,unit_rows
    rng=np.random.default_rng(913)
    z=rng.normal(size=(20,16,6));z=(z/np.linalg.norm(z,axis=2,keepdims=True)).astype('f4')
    e=rng.normal(size=(5,6));e=(e/np.linalg.norm(e,axis=1,keepdims=True)).astype('f4')
    bank=derive_bank(z,e,C13);ids=list(range(15));labels=np.repeat(np.arange(5),3)
    p,_=prototype_model(bank,'P0',None,ids,labels)
    for rep,rotation,lambda_ in (('A1',None,1.),('A2',None,.1),('A3',None,.1),('A4',2,.1)):
        model=score_model(bank,rep,rotation,ids,labels)
        np.testing.assert_array_equal(model['text_prototypes'],bank['T'])
        np.testing.assert_array_equal(model['visual_prototypes'],p)
        expected=p[[1,2,3,4,0]] if rep=='A3' else (p@bank['Q'][rotation] if rep=='A4' else p)
        np.testing.assert_allclose(model['auxiliary_prototypes'],expected,atol=1e-12)
        final,text,aux=score_branches(bank['q'][15:],model,lambda_)
        np.testing.assert_allclose(final,text+lambda_*aux,atol=1e-12)
        np.testing.assert_allclose(text,bank['q'][15:]@bank['T'].T,atol=1e-12)
        assert not np.allclose(np.linalg.norm(model['score_weights'],axis=1),1)
    a1=score_model(bank,'A1',None,ids,labels)
    old,_=prototype_model(bank,'P1',None,ids,labels)
    np.testing.assert_allclose(unit_rows(a1['score_weights']),old,atol=1e-12)
    changed={k:v.copy() for k,v in bank.items()};changed['q'][15:]*=-1
    other=score_model(changed,'A2',None,ids,labels)
    for k,v in other.items():np.testing.assert_array_equal(v,score_model(bank,'A2',None,ids,labels)[k])
    assert len(schedule(DEFAULT_CONFIG))==80
    bad=copy.deepcopy(DEFAULT_CONFIG);bad['lambdas']['A2']=.2
    with pytest.raises(ValueError):validate_config(bad)


def test_invalid_rotation_and_branch_weight_mismatch_are_rejected():
    from hmdb51_e0.e14_features import score_model,score_branches
    with pytest.raises(ValueError):score_model({},'A2',0,[],np.array([]))
    with pytest.raises(ValueError):score_model({},'A4',5,[],np.array([]))
    with pytest.raises(ValueError):score_branches(np.eye(5),dict(text_prototypes=np.eye(5),auxiliary_prototypes=np.eye(5),score_weights=np.eye(5)),.1)
