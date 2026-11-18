import copy
import numpy as np
import pytest


def test_cosine_scores_are_not_renormalized_across_concepts():
    from hmdb51_e0.e1_features import cosine_scores
    z=np.tile(np.array([.6,.8,0.],dtype=np.float32),(16,1))
    e=np.array([[1.,0,0],[0,0,1.]],dtype=np.float32)
    c=cosine_scores(z,e)
    np.testing.assert_allclose(c,np.tile([.6,0.],(16,1)),atol=1e-7)
    assert c.dtype==np.float32


def test_concept_pool_has_literal_global_and_ordered_blocks():
    from hmdb51_e0.e1_features import concept_pool
    c=np.array([[1.,0]]*6+[[0.,1]]*5+[[-1.,0]]*5,dtype=np.float32)
    blocks=[np.array([1.,5.])/16,np.sqrt([175.,55.])/16,
            np.array([1.,0]),np.array([0.,1]),np.array([-1.,0])]
    expected=np.concatenate([b/(np.linalg.norm(b)+1e-6) for b in blocks])
    expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(concept_pool(c,'C1'),expected,atol=1e-7)
    np.testing.assert_array_equal(concept_pool(c,'C0'),concept_pool(c,'R0'))
    np.testing.assert_array_equal(concept_pool(c,'C1'),concept_pool(c,'R1'))
    np.testing.assert_allclose(concept_pool(c,'C0'),concept_pool(c[::-1],'C0'),atol=1e-7)
    assert not np.allclose(concept_pool(c,'C1'),concept_pool(c[::-1],'C1'))
    constant=np.tile(np.array([.2,.4],dtype=np.float32),(16,1))
    assert np.count_nonzero(concept_pool(constant,'C1')[2:4])==0


def test_random_projection_is_fixed_unit_directions_with_matching_concept_count():
    from hmdb51_e0.e1_features import random_directions,cosine_scores
    a=random_directions(6,3,0)
    assert a.shape==(3,6) and a.dtype==np.float32
    np.testing.assert_allclose(np.linalg.norm(a,axis=1),1,atol=1e-6)
    np.testing.assert_array_equal(a,random_directions(6,3,0))
    assert not np.allclose(a,random_directions(6,3,1))
    z=np.tile(np.array([1.,0,0,0,0,0],dtype=np.float32),(16,1))
    np.testing.assert_allclose(cosine_scores(z,a),np.tile(a[:,0],(16,1)),atol=1e-7)


@pytest.mark.parametrize('kind',['short','nan','range','zero','unknown'])
def test_pool_rejects_invalid_similarity_sequences(kind):
    from hmdb51_e0.e1_features import concept_pool
    c=np.tile(np.array([.2,.4],dtype=np.float32),(16,1)); condition='C0'
    if kind=='short': c=c[:15]
    elif kind=='nan': c[0,0]=np.nan
    elif kind=='range': c[0,0]=2
    elif kind=='zero': c[:]=0
    else: condition='C2'
    with pytest.raises(ValueError): concept_pool(c,condition)


@pytest.mark.parametrize('kind',['duplicate_id','duplicate_prompt','label_prompt','empty_prompt','unknown_key'])
def test_catalog_rejects_ambiguous_concepts(kind):
    from hmdb51_e0.e1_features import load_catalog,validate_catalog
    from hmdb51_e0.config import ROOT
    catalog=load_catalog(ROOT/'configs/e1.concepts.v1.yaml')
    catalog=copy.deepcopy(catalog)
    if kind=='duplicate_id': catalog['concepts'][1]['id']=catalog['concepts'][0]['id']
    elif kind=='duplicate_prompt': catalog['concepts'][1]['description_en']=catalog['concepts'][0]['description_en']
    elif kind=='label_prompt': catalog['concepts'][0]['description_en']='a person performing the action clap'
    elif kind=='empty_prompt': catalog['concepts'][0]['description_en']=' '
    else: catalog['task_labels_for_fitting']=['clap']
    with pytest.raises(ValueError): validate_catalog(catalog)


def test_text_alignment_rejects_unrelated_checkpoint_and_fake_text_on_real_images():
    from hmdb51_e0.e1_features import validate_text_alignment
    from hmdb51_e0.clip_features import CHECKPOINT_SHA256
    image={'implementation':'openai/CLIP','architecture':'ViT-B/32','dimension':512,
           'checkpoint_sha256':CHECKPOINT_SHA256,'parameter_sha256_before':'a'*64,'clip_source_sha256':'b'*64}
    text={**image,'text_encoder':'native_openai_clip','tokenizer':{'context_length':77,'truncate':False,
          'bpe_sha256':'c'*64,'tokenizer_source_sha256':'d'*64}}
    frozen={'unchanged':True,'before':'a'*64,'after':'a'*64}
    validate_text_alignment(image,text,frozen)
    for mutation in ('checkpoint_sha256','parameter_sha256_before','clip_source_sha256'):
        wrong={**text,mutation:'e'*64}
        with pytest.raises(ValueError): validate_text_alignment(image,wrong,frozen)
    with pytest.raises(ValueError): validate_text_alignment(image,{'synthetic_test_fixture':True,'dimension':512},{'unchanged':True})


@pytest.mark.parametrize('seeds',[[0],[0,1,2,3,4,5],[1,2,3,4,5]])
def test_e1_protocol_does_not_silently_reduce_or_expand_the_five_control_seeds(seeds):
    from hmdb51_e0.e1_features import DEFAULT_CONFIG,validate_e1_config
    config=copy.deepcopy(DEFAULT_CONFIG); config['random_projection']['seeds']=seeds
    with pytest.raises(ValueError,match='projection|five|seed'): validate_e1_config(config)
