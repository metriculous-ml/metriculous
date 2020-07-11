import numpy as np
import pytest
from scipy.stats import entropy
from sklearn import metrics as sklmetrics

from metriculous import metrics as metrics
from metriculous.metrics import (
    normalized,
    sensitivity_at_x_specificity,
    specificity_at_x_sensitivity,
    top_n_accuracy,
)


# --- normalized -----------------------------------------------------------------------
def test_normalized() -> None:
    # fmt: off
    result = metrics.normalized(np.array([
        [.0, .0],
        [.1, .1],
        [.2, .3],
        [.5, .5],
        [.6, .4],
        [1., 4.],
        [0., 1.],
        [0., 1e-20],
    ]))

    expected = np.array([
        [.5, .5],
        [.5, .5],
        [.4, .6],
        [.5, .5],
        [.6, .4],
        [.2, .8],
        [0., 1.],
        [0., 1.],
    ])

    assert np.allclose(result, expected, atol=0.0)
    # fmt: on


# --- cross-entropy --------------------------------------------------------------------
def test_cross_entropy_zero() -> None:
    ce = metrics.cross_entropy(
        target_probas=np.array([[1.0, 0.0], [1.0, 0.0]]),
        pred_probas=np.array([[1.0, 0.0], [1.0, 0.0]]),
        epsilon=1e-15,
    )
    np.testing.assert_allclose(ce, 0.0, atol=1e-15)


def test_cross_entropy_certainty_in_targets() -> None:
    target_probas = np.array([[1.0, 0.0], [1.0, 0.0]])
    pred_probas = np.array([[0.6, 0.4], [0.1, 0.9]])
    eps = 1e-15
    ce = metrics.cross_entropy(target_probas, pred_probas, epsilon=eps)
    ll = sklmetrics.log_loss(target_probas, pred_probas, eps=eps)
    np.testing.assert_allclose(ce, ll)


def test_cross_entropy_general_fuzz_test() -> None:
    rng = np.random.RandomState(42)
    for _ in range(10):
        probas = normalized(rng.rand(100, 2))
        ce = metrics.cross_entropy(probas, probas)
        scipy_entropy = np.sum(entropy(probas.T)) / len(probas)
        np.testing.assert_allclose(ce, scipy_entropy)


# --- A vs B AUROC ---------------------------------------------------------------------
def test_a_vs_b_auroc() -> None:
    value = metrics.a_vs_b_auroc(
        target_ints=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        predicted_probas=np.array(
            [
                [0.8, 0.0, 0.2],  # correct
                [0.9, 0.1, 0.0],  # correct
                [0.7, 0.1, 0.2],  # correct
                [0.1, 0.8, 0.1],  # correct
                [0.1, 0.8, 0.1],  # correct
                [0.1, 0.6, 0.3],  # correct
                [0.9, 0.0, 0.1],  # wrong
                [0.1, 0.9, 0.0],  # wrong
                [0.1, 0.0, 0.9],  # wrong
            ]
        ),
        class_a=0,
        class_b=1,
    )
    assert value == 1.0


def test_a_vs_b_auroc_symmetry() -> None:
    """Check that result is the same when classes are swapped."""
    rng = np.random.RandomState(42)

    for _ in range(50):
        probas = normalized(rng.rand(100, 4))
        target_ints = rng.randint(0, 4, size=len(probas))

        a1b2 = metrics.a_vs_b_auroc(
            target_ints=target_ints, predicted_probas=probas, class_a=1, class_b=2
        )

        a2b1 = metrics.a_vs_b_auroc(
            target_ints=target_ints, predicted_probas=probas, class_a=2, class_b=1
        )
        np.testing.assert_allclose(a1b2, a2b1, atol=1e-15)


def test_a_vs_b_auroc_zeros() -> None:
    """Check case with zeros in all interesting columns."""
    value = metrics.a_vs_b_auroc(
        target_ints=np.array([0, 1]),
        predicted_probas=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        class_a=0,
        class_b=1,
    )
    assert value is not None
    # we just want to make sure this did not crash
    assert 0.0 <= value <= 1.0


def test_a_vs_b_auroc_none() -> None:
    """Check case where it should return None."""
    rng = np.random.RandomState(42)

    for _ in range(50):
        probas = normalized(rng.rand(100, 4))
        target_ints = rng.randint(0, 1, size=len(probas))

        value = metrics.a_vs_b_auroc(
            target_ints=target_ints, predicted_probas=probas, class_a=1, class_b=2
        )
        assert value is None


# --- sensitivity at specificity -------------------------------------------------------
def test_sensitivity_at_x_specificity() -> None:
    """Test AUC 0.5 prediction."""
    n = 500
    labels = np.concatenate((np.zeros(n), np.ones(n)))
    randoms = np.random.random(n)
    positive_probas = np.concatenate((randoms, randoms + 1e-9))

    for at in np.linspace(0.1, 0.9, num=9):
        sens, spec = sensitivity_at_x_specificity(
            target_ints=labels, positive_probas=positive_probas, at_specificity=at
        )
        assert sens is not None
        assert spec is not None
        np.testing.assert_allclose(spec, at, atol=0.003)
        np.testing.assert_allclose(sens, 1.0 - spec, atol=0.003)


# --- specificity at sensitivity -------------------------------------------------------
def test_specificity_at_x_sensitivity() -> None:
    """Test AUC 0.5 prediction."""
    n = 500
    labels = np.concatenate((np.zeros(n), np.ones(n)))
    randoms = np.random.random(n)
    positive_probas = np.concatenate((randoms, randoms + 1e-9))

    for at in np.linspace(0.1, 0.9, num=9):
        spec, sens = specificity_at_x_sensitivity(
            target_ints=labels, positive_probas=positive_probas, at_sensitivity=at
        )
        assert spec is not None
        assert sens is not None

        np.testing.assert_allclose(sens, at, atol=0.003)
        np.testing.assert_allclose(spec, 1.0 - sens, atol=0.003)


# --- top N accuracy -------------------------------------------------------------------
def test_top_n_accuracy_all_correct() -> None:
    np.random.seed(42)
    n_classes = 30
    for i in range(5):
        target_ints = np.random.randint(0, n_classes, size=100)
        pred_probas = np.eye(n_classes)[target_ints] + np.random.rand(
            len(target_ints), n_classes
        )
        for n in [1, 2, 3, 40, 100]:
            assert top_n_accuracy(target_ints, pred_probas, n) == 1.0


def test_top_n_accuracy() -> None:
    target_ints = np.array([3, 1, 4])
    # fmt:off
    pred_probas = np.array([
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
    ])
    # fmt:on
    assert 1 / 3 == top_n_accuracy(target_ints, pred_probas, n=1)
    assert 2 / 3 == top_n_accuracy(target_ints, pred_probas, n=2)
    assert 2 / 3 == top_n_accuracy(target_ints, pred_probas, n=3)
    assert 3 / 3 == top_n_accuracy(target_ints, pred_probas, n=4)
    assert 3 / 3 == top_n_accuracy(target_ints, pred_probas, n=5)
    assert 3 / 3 == top_n_accuracy(target_ints, pred_probas, n=999)


def test_top_n_accuracy__sample_weights_default() -> None:
    """
    Checks that passing in a uniform sample_weights vector does the same as passing
    `None` or using the default.
    """
    target_ints = np.array([3, 1, 4])

    # fmt:off
    pred_probas = np.array([
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
    ])
    # fmt:on

    assert top_n_accuracy(target_ints, pred_probas, n=1) == top_n_accuracy(
        target_ints, pred_probas, n=1, sample_weights=np.ones_like(target_ints)
    )

    assert top_n_accuracy(
        target_ints, pred_probas, n=1, sample_weights=None
    ) == top_n_accuracy(
        target_ints, pred_probas, n=1, sample_weights=np.ones_like(target_ints)
    )


def test_top_n_accuracy__sample_weights() -> None:
    """
    Same test as above, with additional zero-weighted samples, should get same output.
    """
    target_ints = np.array([3, 1, 4, 1, 1, 1])
    sample_weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    # fmt:off
    pred_probas = np.array([
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
    ])
    # fmt:on
    assert 1 / 3 == top_n_accuracy(
        target_ints, pred_probas, n=1, sample_weights=sample_weights
    )
    assert 2 / 3 == top_n_accuracy(
        target_ints, pred_probas, n=2, sample_weights=sample_weights
    )
    assert 2 / 3 == top_n_accuracy(
        target_ints, pred_probas, n=3, sample_weights=sample_weights
    )
    assert 3 / 3 == top_n_accuracy(
        target_ints, pred_probas, n=4, sample_weights=sample_weights
    )
    assert 3 / 3 == top_n_accuracy(
        target_ints, pred_probas, n=5, sample_weights=sample_weights
    )
    assert 3 / 3 == top_n_accuracy(
        target_ints, pred_probas, n=999, sample_weights=sample_weights
    )


def test_top_n_accuracy__sample_weights_scaled() -> None:
    """
    Checks that scaling the weight vector does not change the results.
    """
    target_ints = np.array([3, 1, 4, 1, 1, 1])
    sample_weights = np.array([2.4, 0.5, 2.1, 0.01, 0.9, 35.7])
    # fmt:off
    pred_probas = np.array([
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
    ])
    # fmt:on

    assert top_n_accuracy(
        target_ints, pred_probas, n=1, sample_weights=sample_weights
    ) == top_n_accuracy(
        target_ints, pred_probas, n=1, sample_weights=42.0 * sample_weights
    )


def test_top_n_accuracy__sample_weights_all_zeros() -> None:
    """
    Checks that passing in zero vector `sample_weights` raises `ZeroDivisionError`.
    """
    target_ints = np.array([3, 1, 4, 1, 1, 1])
    sample_weights = np.zeros_like(target_ints)
    # fmt:off
    pred_probas = np.array([
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
    ])
    # fmt:on

    with pytest.raises(ZeroDivisionError):
        _ = top_n_accuracy(target_ints, pred_probas, n=1, sample_weights=sample_weights)


def test_top_n_accuracy__sample_weights_negative() -> None:
    """
    Checks that an exception is raised if at least one of the sample weights is
    negative.
    """
    target_ints = np.array([3, 1, 4, 1, 1, 1])
    sample_weights = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    # fmt:off
    pred_probas = np.array([
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
        [.3, .4, .2, .1, .0],  # target in top 4
        [.4, .3, .2, .1, .0],  # target in top 2
        [.0, .1, .2, .3, .4],  # target in top 1
    ])
    # fmt:on
    with pytest.raises(AssertionError):
        _ = top_n_accuracy(target_ints, pred_probas, n=1, sample_weights=sample_weights)
