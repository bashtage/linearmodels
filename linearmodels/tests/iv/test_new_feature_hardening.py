"""
Checks of the instrumental variable features that were added after version 7.0

* The tests of the rank of the first-stage coefficient matrix, which are
  cragg_donald, cragg_donald_f, kleibergen_paap and kleibergen_paap_f: that the
  documentation is well-formed, and the behavior with weights and clusters that
  it describes.
* IVGMMResults.centered_j_stat: the units of the data, continuously updating
  GMM, weights, clusters, kernels and singular covariances.
"""

import inspect
import re

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IVGMM, IVGMMCUE
from linearmodels.iv.common import (
    cragg_donald,
    cragg_donald_f,
    kleibergen_paap,
    kleibergen_paap_f,
)
from linearmodels.iv.results import FirstStageResults, IVGMMResults
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic


def simulate(seed, nobs=300, ninstr=4, nendog=1, violation=0.0, weighted=False):
    """
    Heteroskedastic IV data with a constant and an exogenous regressor

    The last instrument is invalid if violation is not 0. The weights are None
    unless weighted.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((nobs, ninstr))
    v = rng.standard_normal((nobs, nendog))
    u = (1 + 0.5 * np.abs(z[:, 0])) * (0.5 * v.sum(1) + rng.standard_normal(nobs))
    endog = z @ rng.uniform(0.3, 0.8, (ninstr, nendog)) + v
    exog = np.column_stack([np.ones(nobs), rng.standard_normal(nobs)])
    y = exog @ [1.0, 0.5] + endog.sum(1) + violation * z[:, -1] + u
    w = rng.uniform(0.3, 3.0, nobs) if weighted else None
    return y, exog, endog, z, w


# ----------------------------------------------------------------------------
# Tests of the rank of the first-stage coefficient matrix
# ----------------------------------------------------------------------------

DOCUMENTED = {
    "cragg_donald": cragg_donald,
    "cragg_donald_f": cragg_donald_f,
    "kleibergen_paap": kleibergen_paap,
    "kleibergen_paap_f": kleibergen_paap_f,
    "FirstStageResults.cragg_donald": FirstStageResults.cragg_donald,
    "FirstStageResults.cragg_donald_f": FirstStageResults.cragg_donald_f,
    "FirstStageResults.kleibergen_paap": FirstStageResults.kleibergen_paap,
    "FirstStageResults.kleibergen_paap_f": FirstStageResults.kleibergen_paap_f,
}


def footnote_problems(doc):
    """Footnotes that are defined and not cited, and citations not defined"""
    defined = set(re.findall(r"^\s*\.\. \[(\w+)\]", doc, flags=re.MULTILINE))
    cited = set(re.findall(r"\[(\w+)\]_", doc))
    return sorted(defined - cited), sorted(cited - defined)


def test_footnote_check_finds_problems():
    doc = "Text [1]_\n\nReferences\n----------\n.. [1] A\n.. [2] B\n"
    assert footnote_problems(doc) == (["2"], [])
    assert footnote_problems("A citation [3]_") == ([], ["3"])
    assert footnote_problems("References\n----------\nA\n\nB\n") == ([], [])


@pytest.mark.parametrize("name", list(DOCUMENTED))
def test_footnotes_are_cited(name):
    # Sphinx warns "Footnote [n] is not referenced" for each of them. Use
    # author-year references, as other docstrings do, or cite the footnotes
    doc = inspect.getdoc(DOCUMENTED[name])
    assert footnote_problems(doc) == ([], [])


SOURCES = {
    "cragg_donald": [
        "Cragg, J. G., & Donald, S. G. (1993)",
        "Anderson, T. W. (1951)",
        "Stock, J. H., & Yogo, M. (2005)",
    ],
    "cragg_donald_f": ["Stock, J. H., & Yogo, M. (2005)"],
    "kleibergen_paap": [
        "Kleibergen, F., & Paap, R. (2006)",
        "Kleibergen, F., & Schaffer, M. E. (2007)",
    ],
    "kleibergen_paap_f": [
        "Kleibergen, F., & Paap, R. (2006)",
        "Olea, J. L. M., & Pflueger, C. (2013)",
        "Baum, C. F., Schaffer, M. E., & Stillman, S. (2007)",
    ],
}


@pytest.mark.parametrize("name", list(SOURCES))
def test_sources_are_listed(name):
    # Removing the footnote markup keeps the references
    doc = inspect.getdoc(DOCUMENTED[name])
    references = doc.split("References\n----------\n")[1]
    for source in SOURCES[name]:
        assert source in references


# ----------------------------------------------------------------------------
# IVGMMResults.centered_j_stat
# ----------------------------------------------------------------------------


def moment_conditions(res, y, exog, endog, z, w):
    """Moment conditions of the weighted model at the estimates"""
    sw = np.ones(y.shape[0]) if w is None else np.sqrt(w / w.mean())
    x = np.column_stack([exog, endog]) * sw[:, None]
    instruments = np.column_stack([exog, z]) * sw[:, None]
    eps = y * sw - x @ res.params.to_numpy()
    return instruments * eps[:, None]


def centered_covariance(
    g, weight_type, nvar, *, debiased=False, clusters=None, bandwidth=None
):
    """The covariance of the moments about their mean, from the definition"""
    nobs = g.shape[0]
    gc = g - g.mean(0)
    if weight_type == "robust":
        s = gc.T @ gc / nobs
        scale = nobs / (nobs - nvar)
    elif weight_type == "kernel":
        s = gc.T @ gc
        for lag in range(1, bandwidth + 1):
            gamma = gc[lag:].T @ gc[:-lag]
            s = s + (1 - lag / (bandwidth + 1)) * (gamma + gamma.T)
        s = s / nobs
        scale = nobs / (nobs - nvar)
    else:
        sums = pd.DataFrame(gc).groupby(clusters).sum().to_numpy()
        s = sums.T @ sums / nobs
        ngroups = sums.shape[0]
        scale = (nobs - 1) / (nobs - nvar) * ngroups / (ngroups - 1)
    return s * scale if debiased else s


def weight_options(weight_type, nobs):
    if weight_type == "kernel":
        return {"kernel": "bartlett", "bandwidth": 3}
    if weight_type == "clustered":
        return {"clusters": np.arange(nobs) // 10}
    return {}


def expected_centered_j(res, data, weight_type, debiased, options):
    y, exog, endog, z, w = data
    g = moment_conditions(res, y, exog, endog, z, w)
    nvar = exog.shape[1] + endog.shape[1]
    s_c = centered_covariance(
        g,
        weight_type,
        nvar,
        debiased=debiased,
        clusters=options.get("clusters"),
        bandwidth=options.get("bandwidth"),
    )
    gbar = g.mean(0)
    return g.shape[0] * gbar @ np.linalg.solve(s_c, gbar)


@pytest.mark.parametrize("debiased", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("weight_type", ["robust", "kernel", "clustered"])
def test_centered_j_gmm_matches_the_definition(weight_type, weighted, debiased):
    data = simulate(20, violation=0.3, weighted=weighted)
    y, exog, endog, z, w = data
    options = weight_options(weight_type, y.shape[0])
    mod = IVGMM(
        y,
        exog,
        endog,
        z,
        weights=w,
        weight_type=weight_type,
        debiased=debiased,
        **options,
    )
    res = mod.fit()
    expected = expected_centered_j(res, data, weight_type, debiased, options)
    assert type(res.centered_j_stat) is WaldTestStatistic
    assert res.centered_j_stat.df == 3
    assert_allclose(res.centered_j_stat.stat, expected, rtol=1e-9)
    # The violation of the moment conditions makes the statistics differ
    assert not np.isclose(res.centered_j_stat.stat, res.j_stat.stat, rtol=1e-3)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("weight_type", ["robust", "kernel", "clustered"])
def test_centered_j_cue_matches_the_definition(weight_type, weighted):
    data = simulate(21, violation=0.3, weighted=weighted)
    y, exog, endog, z, w = data
    options = weight_options(weight_type, y.shape[0])
    mod = IVGMMCUE(y, exog, endog, z, weights=w, weight_type=weight_type, **options)
    res = mod.fit()
    assert isinstance(res, IVGMMResults)
    expected = expected_centered_j(res, data, weight_type, False, options)
    assert_allclose(res.centered_j_stat.stat, expected, rtol=1e-8)
    # The CUE centers by default, and so the weight matrix of the objective is
    # the covariance of the statistic
    assert res.weight_config["center"] is True
    assert_allclose(res.centered_j_stat.stat, res.j_stat.stat, rtol=1e-8)
    assert res.centered_j_stat.df == res.j_stat.df == 3


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("weight_type", ["robust", "kernel", "clustered"])
def test_centered_j_cue_without_centering(weight_type, weighted):
    data = simulate(22, violation=0.3, weighted=weighted)
    y, exog, endog, z, w = data
    options = weight_options(weight_type, y.shape[0])
    mod = IVGMMCUE(
        y, exog, endog, z, weights=w, weight_type=weight_type, center=False, **options
    )
    res = mod.fit()
    expected = expected_centered_j(res, data, weight_type, False, options)
    assert res.weight_config["center"] is False
    assert_allclose(res.centered_j_stat.stat, expected, rtol=1e-8)
    assert abs(res.centered_j_stat.stat - res.j_stat.stat) > 1e-3
    if weight_type == "robust":
        # S = S_c + gbar gbar' at the estimates, as in iterated GMM
        nobs = y.shape[0]
        j = res.j_stat.stat
        assert_allclose(res.centered_j_stat.stat, j / (1 - j / nobs), rtol=1e-8)


@pytest.mark.parametrize("weighted", [False, True])
def test_centered_j_cue_with_too_few_clusters(weighted):
    # The centered cluster sums add to zero, so that the covariance has at most
    # nclusters - 1 non-zero eigenvalues, and the model has 5 moment conditions.
    # The uncentered covariance of the CUE has nclusters
    y, exog, endog, z, w = simulate(23, ninstr=3, violation=0.3, weighted=weighted)
    outcomes = {}
    for nclusters in (5, 6):
        mod = IVGMMCUE(
            y,
            exog,
            endog,
            z,
            weights=w,
            weight_type="clustered",
            clusters=np.arange(y.shape[0]) % nclusters,
            center=False,
        )
        res = mod.fit()
        outcomes[nclusters] = res
        assert np.isfinite(res.j_stat.stat)
        assert res.j_stat.df == 2
    invalid = outcomes[5].centered_j_stat
    assert isinstance(invalid, InvalidTestStatistic)
    assert np.isnan(invalid.stat)
    assert np.isnan(invalid.pval)
    assert invalid.critical_values is None
    assert "singular" in str(invalid)
    assert "Centered J-test" in str(invalid)
    valid = outcomes[6].centered_j_stat
    assert type(valid) is WaldTestStatistic
    assert valid.df == 2
    assert np.isfinite(valid.stat)


@pytest.mark.parametrize("scale", [1e9, 1e-9])
@pytest.mark.parametrize("column", ["instrument", "exog"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("weight_type", ["robust", "unadjusted", "kernel", "clustered"])
def test_centered_j_does_not_depend_on_the_units_of_the_data(
    weight_type, weighted, column, scale
):
    # Measuring a variable in other units multiplies a moment condition by a
    # constant. The J statistic does not change, and the covariance of the
    # moment conditions is not singular. The test of its rank used the largest
    # eigenvalue of the covariance and so rejected any covariance for which the
    # variances of the moment conditions differed by a factor of 1e16
    y, exog, endog, z, w = simulate(24, violation=0.3, weighted=weighted)
    options = weight_options(weight_type, y.shape[0])

    def fit(exog_in, z_in):
        mod = IVGMM(
            y, exog_in, endog, z_in, weights=w, weight_type=weight_type, **options
        )
        return mod.fit()

    base = fit(exog, z)
    exog_in, z_in = exog.copy(), z.copy()
    if column == "instrument":
        z_in[:, 1] *= scale
    else:
        exog_in[:, 1] *= scale
    res = fit(exog_in, z_in)
    assert type(base.centered_j_stat) is WaldTestStatistic
    assert_allclose(res.j_stat.stat, base.j_stat.stat, rtol=1e-6)
    assert type(res.centered_j_stat) is WaldTestStatistic
    assert res.centered_j_stat.df == base.centered_j_stat.df
    assert_allclose(res.centered_j_stat.stat, base.centered_j_stat.stat, rtol=1e-6)
    assert_allclose(res.centered_j_stat.pval, base.centered_j_stat.pval, rtol=1e-4)


def test_centered_j_singular_covariance_is_still_found_with_other_units():
    # The check of the rank does not accept a covariance that is singular. There
    # are 5 moment conditions and 5 clusters, so that the covariance that is
    # centered has rank 4
    y, exog, endog, z, _ = simulate(25, ninstr=3, violation=0.3)
    z = z.copy()
    z[:, 1] *= 1e9
    mod = IVGMM(
        y,
        exog,
        endog,
        z,
        weight_type="clustered",
        clusters=np.arange(y.shape[0]) % 5,
    )
    res = mod.fit()
    assert isinstance(res.centered_j_stat, InvalidTestStatistic)
    assert "singular" in str(res.centered_j_stat)
