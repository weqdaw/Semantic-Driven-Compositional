import numpy as np
import pytest

from hmdb51_e0.clip_features import pool_features
from hmdb51_e0.ridge import fit_ridge
from hmdb51_e0.metrics import classification_metrics


def test_pooling_uses_population_std_and_is_order_invariant():
    raw = np.tile([[1.0, 0], [0, 1]], (8, 1))
    z, vector = pool_features(raw)
    np.testing.assert_allclose(z, raw)
    np.testing.assert_allclose(vector, [0.5, 0.5, 0.5, 0.5], atol=1e-7)
    np.testing.assert_allclose(pool_features(raw[::-1])[1], vector)
    # Constant frames must yield a zero std block, not NaN.
    _, constant = pool_features(np.tile([[3.0, 4]], (16, 1)))
    np.testing.assert_allclose(constant, [0.6, 0.8, 0, 0], atol=1e-6)


@pytest.mark.parametrize("raw", [np.zeros((16, 2)), np.ones((15, 2)), np.full((16, 2), np.nan)])
def test_invalid_encoder_features_fail(raw):
    with pytest.raises(ValueError):
        pool_features(raw)


def test_ridge_matches_hand_solution_and_unpenalized_intercept():
    # n=2, centered X=[-1,+1]; W=(X'X + 2*lambda)^-1 X'Y.
    x = np.array([[0.0], [2.0]])
    w, b = fit_ridge(x, np.array([0, 1]), lambda_=1.0)
    np.testing.assert_allclose(w, [[-0.25, 0.25, 0, 0, 0]])
    np.testing.assert_allclose(b, [0.75, 0.25, 0, 0, 0])
    shifted_w, shifted_b = fit_ridge(x + 1000, np.array([0, 1]))
    np.testing.assert_allclose(shifted_w, w)
    np.testing.assert_allclose((x + 1000) @ shifted_w + shifted_b, x @ w + b)


def test_metrics_class_order_ties_and_missing_class_zero_f1():
    truth = np.array([0, 1, 2, 3, 4])
    scores = np.eye(5)
    scores[4] = 0  # five-way tie -> class 0
    result = classification_metrics(truth, scores)
    assert result["accuracy"] == 0.8
    assert result["macro_f1"] == pytest.approx(11 / 15)
    assert result["per_class_recall"] == [1, 1, 1, 1, 0]
    assert result["confusion_matrix"] == [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]


def test_invalid_labels_and_scores_are_rejected():
    with pytest.raises(ValueError):
        fit_ridge(np.ones((2, 3)), np.array([0, 5]))
    with pytest.raises(ValueError):
        classification_metrics(np.arange(5), np.full((5, 5), np.inf))
