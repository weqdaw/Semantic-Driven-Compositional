import copy
import numpy as np
import pytest


def test_support_only_scaling_matches_trace_and_query_cannot_change_parameters():
    from hmdb51_e0.e12_features import build_vectors
    rng=np.random.default_rng(827)
    raw={'V1':rng.normal(size=(20,30)), 'C1':rng.normal(size=(20,10))*.001+7.}
    support=list(range(15))
    fused,params,diagnostics=build_vectors(raw,'SVC1',None,support)
    changed={k:v.copy() for k,v in raw.items()}
    for x in changed.values(): x[15:]+=100000
    other,other_params,_=build_vectors(changed,'SVC1',None,support)
    for k in params: np.testing.assert_array_equal(params[k],other_params[k])
    np.testing.assert_array_equal(fused[:15],other[:15])
    np.testing.assert_allclose(fused[:15].mean(0),0,atol=1e-10)
    assert np.mean(np.sum(fused[:15]**2,axis=1))==pytest.approx(1.)
    assert np.mean(np.sum(fused[:15,:30]**2,axis=1))==pytest.approx(.5)
    assert all(d['after_centered_variance']==pytest.approx(1.) for d in diagnostics)
    assert np.std(np.linalg.norm(fused[:15],axis=1))>1e-3  # No per-video L2.
    assert np.all(raw['C1']<8.)  # Inputs remain untouched.
    with pytest.raises(ValueError,match='degenerate'):
        build_vectors({'C1':np.ones((20,3))},'SC1',None,support)


def test_lambda2_is_exact_half_visual_energy_control_and_not_a_hyperparameter_search():
    from hmdb51_e0.e12_features import DEFAULT_CONFIG,schedule,validate_config,build_vectors
    from hmdb51_e0.ridge import fit_ridge
    rng=np.random.default_rng(21); x=rng.normal(size=(20,6)); labels=np.arange(15)%5
    w,b=fit_ridge(x[:15],labels,2.)
    padded=np.column_stack((x/np.sqrt(2),np.zeros((20,4))))
    wp,bp=fit_ridge(padded[:15],labels,1.)
    np.testing.assert_allclose(x@w+b,padded@wp+bp,atol=1e-12)
    vectors,params,_=build_vectors({'V1':x},'V1_lambda2',None,list(range(15)))
    np.testing.assert_array_equal(vectors,x); assert params=={}
    rows=schedule(DEFAULT_CONFIG)
    assert len(rows)==240 and len({tuple(r) for r in rows})==240
    assert {r[2] for r in rows}=={3}
    assert sum(r[0]=='V1_lambda2' and r[4]==2 for r in rows)==10
    bad=copy.deepcopy(DEFAULT_CONFIG); bad['shots']=[1,3,5]
    with pytest.raises(ValueError): validate_config(bad)
