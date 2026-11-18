import copy
import numpy as np
import pytest

from hmdb51_e0.e11_features import DEFAULT_CONFIG, orthogonal_matrix, equal_fusion, derive_bank, validate_config, schedule


def test_common_orthogonal_transform_preserves_text_geometry_and_changes_alignment():
    from hmdb51_e0.e1_features import concept_pool, cosine_scores
    rng=np.random.default_rng(29)
    e=rng.normal(size=(4,6)); e=(e/np.linalg.norm(e,axis=1,keepdims=True)).astype('f4')
    z=rng.normal(size=(2,16,6)); z=(z/np.linalg.norm(z,axis=2,keepdims=True)).astype('f4')
    c1=np.stack([concept_pool(cosine_scores(v,e),'C1') for v in z])
    bank=derive_bank({'E':e,'Z':z,'C1':c1},DEFAULT_CONFIG)
    assert bank['G1'].shape==(5,2,20) and bank['VC1'].shape==(2,50)
    for j,seed in enumerate(range(5)):
        q=orthogonal_matrix(6,seed)
        np.testing.assert_array_equal(q,orthogonal_matrix(6,seed))
        np.testing.assert_allclose(q.T@q,np.eye(6),atol=1e-12)
        np.testing.assert_allclose(bank['D'][j]@bank['D'][j].T,e@e.T,atol=1e-6)
        np.testing.assert_allclose(bank['D'][j],e@q,atol=1e-7)
    assert not np.allclose(bank['G1'][0],c1)


def test_fusion_has_equal_block_energy_and_rejects_degenerate_features():
    fused=equal_fusion(np.array([[3.,4.]]),np.array([[1.,2.,2.]]))
    assert np.sum(fused[0,:2]**2)==pytest.approx(.5)
    assert np.sum(fused[0,2:]**2)==pytest.approx(.5)
    for a,b in (([[0.,0.]],[[1.,2.]]),([[float('nan')]],[[1.]]),([[1.],[2.]],[[1.]])):
        with pytest.raises(ValueError): equal_fusion(a,b)


def test_protocol_is_fixed_and_schedule_has_330_new_fits():
    cfg=copy.deepcopy(DEFAULT_CONFIG); validate_config(cfg)
    reference={'sampling':{'shots':[1,3,5],'support_seeds':list(range(10))}}
    rows=schedule(cfg,reference)
    assert len(rows)==330 and len({tuple(r) for r in rows})==330
    assert sum(r[0]=='VC1' and r[1] is None for r in rows)==30
    cfg['rotation_seeds']=[1,2,3,4,5]
    with pytest.raises(ValueError): validate_config(cfg)
