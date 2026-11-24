"""Keep the exploratory decision separate from statistical significance."""
import pytest


@pytest.mark.parametrize('gain,positive,collapsed,wrong,expected', [
    (.02, 8, .001, .001, True),
    (.03, 10, .01, .01, True),
    (.019999999, 10, .01, .01, False),
    (.03, 7, .01, .01, False),
    (.03, 10, 0., .01, False),
    (.03, 10, .01, 0., False),
    (.03, 10, -.01, .01, False),
    (.03, 10, .01, -.01, False),
])
def test_continuation_requires_gain_consistency_and_both_controls(gain, positive, collapsed, wrong, expected):
    # These literal cases catch an OR in place of AND, a strict threshold that
    # wrongly rejects .02/8, or treating a tied control as a positive result.
    from hmdb51_e0.e15_reporting import continuation_gate
    from hmdb51_e0.e15_features import DEFAULT_CONFIG
    paired = [
        dict(comparison='IC1_minus_T0', primary=True, macro_f1_delta_mean=gain, macro_f1_positive_pairs=positive),
        dict(comparison='IC1_minus_IC2', primary=True, macro_f1_delta_mean=collapsed),
        dict(comparison='IC1_minus_IC3', primary=True, macro_f1_delta_mean=wrong),
        dict(comparison='IC1_minus_A2', primary=False, macro_f1_delta_mean=1.),
    ]
    assert continuation_gate(paired, DEFAULT_CONFIG) is expected
