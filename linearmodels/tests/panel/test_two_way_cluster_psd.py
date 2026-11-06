"""
The two-way clustered covariance is not always positive semi-definite.

In small panels a variance of the two-way clustered covariance can be negative.
The results then issue a NegativeVarianceWarning that names the parameters, and
fix_psd=True sets the negative eigenvalues of the covariance to zero, as in
Cameron, Gelbach and Miller (2011).
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
from linearmodels.shared.covariance import clip_negative_eigenvalues
from linearmodels.shared.exceptions import NegativeVarianceWarning

TWO_WAY = {"cov_type": "clustered", "cluster_entity": True, "cluster_time": True}


def small_panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([range(6), range(5)], names=["entity", "time"])
    data = pd.DataFrame(
        {"x": rng.standard_normal(30), "y": rng.standard_normal(30)}, index=index
    )
    data["const"] = 1.0
    return data


def two_way_by_hand(y: pd.Series, x: pd.DataFrame) -> np.ndarray:
    """
    n / (n - k) (X'X)^-1 (S_entity + S_time - S_entity,time) (X'X)^-1

    The scale n / (n - k) is the default debiased=True.
    """
    xa = np.asarray(x)
    xpxi = np.linalg.inv(xa.T @ xa)
    eps = np.asarray(y) - xa @ (xpxi @ xa.T @ np.asarray(y))
    scores = pd.DataFrame(xa * eps[:, None], index=x.index)

    def meat(level) -> np.ndarray:
        sums = np.asarray(scores.groupby(level=level).sum())
        return sums.T @ sums

    s = meat("entity") + meat("time") - meat(["entity", "time"])
    nobs, nvar = xa.shape
    return nobs / (nobs - nvar) * xpxi @ s @ xpxi


@pytest.fixture
def indefinite():
    # The example of the issue: the variance of const is negative
    data = small_panel(4)
    return PooledOLS(data.y, data[["const", "x"]])


@pytest.fixture
def definite():
    data = small_panel(0)
    return PooledOLS(data.y, data[["const", "x"]])


def test_clip_negative_eigenvalues():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    eigenvalues = np.array([-2.0, -0.5, 1.0, 3.0])
    cov = (q * eigenvalues) @ q.T
    fixed = clip_negative_eigenvalues(cov)
    assert_allclose(fixed, (q * np.array([0.0, 0.0, 1.0, 3.0])) @ q.T, atol=1e-14)
    assert_allclose(fixed, fixed.T, rtol=0, atol=0)
    assert_allclose(np.linalg.eigvalsh(fixed), [0.0, 0.0, 1.0, 3.0], atol=1e-14)


def test_clip_negative_eigenvalues_psd_unchanged():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 4))
    cov = x.T @ x / 50
    assert clip_negative_eigenvalues(cov) is cov


def test_negative_variance_warns(indefinite):
    res = indefinite.fit(**TWO_WAY)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        std_errors = res.std_errors
    # Only the warning that names the parameter, not NumPy's from np.sqrt
    assert len(record) == 1
    assert record[0].category is NegativeVarianceWarning
    message = str(record[0].message)
    assert "variance of const is negative" in message
    assert "fix_psd=True" in message
    # The covariance is not changed by default
    data = small_panel(4)
    expected = two_way_by_hand(data.y, data[["const", "x"]])
    assert_allclose(res.cov, expected)
    assert expected[0, 0] < 0
    assert np.isnan(std_errors["const"])
    assert np.isfinite(std_errors["x"])


def test_negative_variances_warn():
    rng = np.random.default_rng(3)
    index = pd.MultiIndex.from_product([range(6), range(5)], names=["entity", "time"])
    data = pd.DataFrame(
        rng.standard_normal((30, 3)), index=index, columns=["x1", "x2", "y"]
    )
    data["const"] = 1.0
    res = PooledOLS(data.y, data[["const", "x1", "x2"]]).fit(**TWO_WAY)
    with pytest.warns(
        NegativeVarianceWarning, match="variances of x1, x2 are negative"
    ):
        std_errors = res.std_errors
    assert np.isfinite(std_errors["const"])
    assert std_errors[["x1", "x2"]].isna().all()


def test_negative_variance_warning_is_runtime_warning(indefinite):
    res = indefinite.fit(**TWO_WAY)
    with pytest.warns(RuntimeWarning, match="variance of const is negative"):
        assert res.cov.shape == (2, 2)


def test_fix_psd(indefinite):
    expected = indefinite.fit(**TWO_WAY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NegativeVarianceWarning)
        expected_cov = expected.cov
    res = indefinite.fit(**TWO_WAY, fix_psd=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cov = res.cov
        std_errors = res.std_errors
        pvalues = res.pvalues
    assert_allclose(cov, clip_negative_eigenvalues(np.asarray(expected_cov)))
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-15)
    assert np.all(np.isfinite(std_errors))
    assert np.all(np.isfinite(pvalues))
    assert_allclose(res.params, expected.params)


def test_fix_psd_definite_unchanged(definite):
    expected = definite.fit(**TWO_WAY)
    res = definite.fit(**TWO_WAY, fix_psd=True)
    assert_allclose(res.cov, expected.cov, rtol=0, atol=0)


def test_fix_psd_one_way_unchanged(indefinite):
    expected = indefinite.fit(cov_type="clustered", cluster_entity=True)
    res = indefinite.fit(cov_type="clustered", cluster_entity=True, fix_psd=True)
    assert_allclose(res.cov, expected.cov, rtol=0, atol=0)


@pytest.mark.parametrize("model", [PanelOLS, RandomEffects])
def test_fix_psd_models(model):
    data = small_panel(4)
    exog = data[["x"]] if model is PanelOLS else data[["const", "x"]]
    mod = model(data.y, exog)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NegativeVarianceWarning)
        expected = np.asarray(mod.fit(**TWO_WAY).cov)
    res = mod.fit(**TWO_WAY, fix_psd=True)
    assert_allclose(res.cov, clip_negative_eigenvalues(expected))


def test_fix_psd_clusters(indefinite):
    # Two cluster variables given in clusters instead of the flags
    index = indefinite.dependent.index
    clusters = pd.DataFrame(
        {"entity": index.codes[0], "time": index.codes[1]}, index=index
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NegativeVarianceWarning)
        flags = np.asarray(indefinite.fit(**TWO_WAY).cov)
        expected = np.asarray(
            indefinite.fit(cov_type="clustered", clusters=clusters).cov
        )
    assert_allclose(expected, flags)
    res = indefinite.fit(cov_type="clustered", clusters=clusters, fix_psd=True)
    assert_allclose(res.cov, clip_negative_eigenvalues(expected))


def test_fix_psd_not_accepted(indefinite):
    with pytest.raises(ValueError, match="'fix_psd' is not accepted"):
        indefinite.fit(cov_type="robust", fix_psd=True)


def test_fix_psd_not_bool(indefinite):
    with pytest.raises(TypeError, match="fix_psd"):
        indefinite.fit(**TWO_WAY, fix_psd=1)
