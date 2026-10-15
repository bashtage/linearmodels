"""
Checks of the instrumental variable features that were added after version 7.0

* IVJIVE: the docstring examples, formulas with weights and missing values,
  prediction, comparison with the other estimators, the summary and the
  first-stage tests of the rank.
* The tests of the rank of the first-stage coefficient matrix, which are
  cragg_donald, cragg_donald_f, kleibergen_paap and kleibergen_paap_f: that the
  documentation is well-formed, and the behavior with weights and clusters that
  it describes.
* IVGMMResults.centered_j_stat: the units of the data, continuously updating
  GMM, weights, clusters, kernels and singular covariances.
"""

import doctest
import inspect
import pickle
import re
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.datasets import wage
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVJIVE, IVLIML, compare
from linearmodels.iv.common import (
    cragg_donald,
    cragg_donald_f,
    kleibergen_paap,
    kleibergen_paap_f,
)
import linearmodels.iv.model
from linearmodels.iv.results import FirstStageResults, IVGMMResults
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.tests._utility import assert_same_text


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


def simulated_frame(seed, nobs=200, weighted=True):
    """The same type of data in a DataFrame, with names for a formula"""
    y, exog, endog, z, w = simulate(seed, nobs=nobs, ninstr=3, weighted=weighted)
    data = pd.DataFrame(
        {
            "y": y,
            "x1": exog[:, 1],
            "x2": endog[:, 0],
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
        }
    )
    if weighted:
        data["w"] = w
    return data


FORMULA = "y ~ 1 + x1 + [x2 ~ z1 + z2 + z3]"


# ----------------------------------------------------------------------------
# IVJIVE
# ----------------------------------------------------------------------------


def run_docstring_examples(obj, name):
    """Run the examples in the docstrings of an object and of its members"""
    finder = doctest.DocTestFinder(exclude_empty=True)
    failures = []
    with_examples = set()
    for test in finder.find(obj, name, globs={}):
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingValueWarning)
            runner.run(test, out=failures.append)
        if test.examples:
            with_examples.add(test.name)
    return with_examples, failures


def test_jive_docstring_examples_run():
    with_examples, failures = run_docstring_examples(IVJIVE, "IVJIVE")
    assert not failures, "".join(failures)
    assert {"IVJIVE", "IVJIVE.from_formula"} <= with_examples


def test_jive_from_formula_example_has_every_term():
    # exper ** 2 is the power of a set of terms in a formula, which is exper, and
    # not its square. Without I() the example silently omits the term that it
    # names
    doc = IVJIVE.from_formula.__doc__
    formula = re.search(r'formula = "(.*)"', doc).group(1)
    assert "exper ** 2" not in formula.replace("I(exper ** 2)", "")
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = IVJIVE.from_formula(formula, wage.load())
    assert "I(exper ** 2)" in list(mod.fit().params.index)


def test_jive_is_in_the_model_module_all():
    assert "IVJIVE" in linearmodels.iv.model.__all__
    assert linearmodels.iv.model.IVJIVE is IVJIVE


def test_jive_from_formula_weights_and_missing_values():
    # Only the exogenous regressors of a formula can contain missing values
    data = simulated_frame(1)
    data.loc[[4, 77, 10], "x1"] = np.nan
    with pytest.warns(MissingValueWarning, match="missing values"):
        mod = IVJIVE.from_formula(FORMULA, data, weights=data.w)
    res = mod.fit()
    keep = data.notnull().all(axis=1)
    clean = data[keep]
    expected = IVJIVE.from_formula(FORMULA, clean, weights=clean.w).fit()
    unweighted = IVJIVE.from_formula(FORMULA, clean).fit()
    assert res.nobs == keep.sum() == 197
    assert list(res.params.index) == ["Intercept", "x1", "x2"]
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)
    assert_allclose(res.resids, expected.resids, rtol=1e-10)
    # The weights are used
    assert not np.allclose(res.params, unweighted.params, rtol=1e-3)
    assert mod.formula == FORMULA


def test_jive_predict_with_formula_data_and_weights():
    data = simulated_frame(2)
    res = IVJIVE.from_formula(FORMULA, data, weights=data.w).fit(cov_type="robust")
    new = data.iloc[:7]
    x_new = np.column_stack([np.ones(7), new.x1, new.x2])
    expected = x_new @ res.params.to_numpy()
    pred = res.predict(data=new)
    assert list(pred.columns) == ["predictions"]
    assert pred.index.equals(new.index)
    assert_allclose(pred.to_numpy().squeeze(), expected, rtol=1e-12)
    # The dependent variable and the instruments are not needed
    only_regressors = res.predict(data=new[["x1", "x2"]])
    assert_allclose(only_regressors.to_numpy().squeeze(), expected, rtol=1e-12)
    # In sample predictions are the fitted values, whatever the weights
    in_sample = res.predict(data=data)
    assert_allclose(in_sample.to_numpy(), res.fitted_values.to_numpy(), rtol=1e-10)
    assert_allclose(res.predict().to_numpy(), res.fitted_values.to_numpy())
    # Values of the regressors are an alternative to data
    exog_new = new[["x1"]].assign(const=1.0)[["const", "x1"]]
    arrays = res.predict(exog_new, new[["x2"]])
    assert_allclose(arrays.to_numpy().squeeze(), expected, rtol=1e-12)
    with pytest.raises(ValueError, match="but not both"):
        res.predict(new[["x1"]], new[["x2"]], data=new)


def test_jive_compare_with_other_estimators_and_weights():
    y, exog, endog, z, w = simulate(3, weighted=True)
    results = {
        "JIVE": IVJIVE(y, exog, endog, z, weights=w).fit(),
        "2SLS": IV2SLS(y, exog, endog, z, weights=w).fit(),
        "LIML": IVLIML(y, exog, endog, z, weights=w).fit(),
        "GMM": IVGMM(y, exog, endog, z, weights=w).fit(),
    }
    comparison = compare(results)
    assert list(comparison.params.columns) == list(results)
    for name, res in results.items():
        assert_allclose(comparison.params[name].to_numpy(), res.params.to_numpy())
        assert_allclose(comparison.tstats[name].to_numpy(), res.tstats.to_numpy())
        assert_allclose(comparison.rsquared[name], res.rsquared)
        assert comparison.nobs[name] == res.nobs
    methods = comparison.estimator_method
    assert methods.to_dict() == {
        "JIVE": "IV-JIVE",
        "2SLS": "IV-2SLS",
        "LIML": "IV-LIML",
        "GMM": "IV-GMM",
    }
    assert set(comparison.cov_estimator) == {"robust"}
    text = str(comparison.summary)
    for label in ("JIVE", "2SLS", "LIML", "GMM", "IV-JIVE", "IV-GMM"):
        assert label in text
    # A list of results, stars and the other precisions
    as_list = compare(list(results.values()), precision="std_errors", stars=True)
    assert_allclose(as_list.std_errors.iloc[:, 0], results["JIVE"].std_errors)
    assert "***" in str(as_list.summary)


def test_jive_compare_formula_models():
    data = simulated_frame(4)
    results = {
        "JIVE": IVJIVE.from_formula(FORMULA, data, weights=data.w).fit(),
        "2SLS": IV2SLS.from_formula(FORMULA, data, weights=data.w).fit(),
    }
    comparison = compare(results)
    assert list(comparison.params.index) == ["Intercept", "x1", "x2"]
    assert_allclose(comparison.params["JIVE"], results["JIVE"].params)


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "kernel", "clustered"])
def test_jive_summary_with_weights_and_small_sample_adjustment(cov_type):
    y, exog, endog, z, w = simulate(5, weighted=True)
    options = {}
    if cov_type == "kernel":
        options["bandwidth"] = 3
    elif cov_type == "clustered":
        options["clusters"] = np.arange(y.shape[0]) // 12
    res = IVJIVE(y, exog, endog, z, weights=w).fit(
        cov_type=cov_type, debiased=True, **options
    )
    text = str(res.summary)
    assert "IV-JIVE" in text
    assert "Debiased: True" in text
    assert "JIVE Covariance" in text
    # The model F-statistic of a debiased model is an F
    assert res.f_statistic.dist_name.startswith("F(")
    assert res.f_statistic.dist_name in text
    assert "Endogenous: endog" in text
    assert "Instruments: instruments.0" in text
    assert max(len(line) for line in text.split("\n")) <= 78


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    ("cov_type", "options"),
    [
        ("unadjusted", {}),
        ("robust", {}),
        ("clustered", {"clusters": np.arange(300) // 10}),
        ("kernel", {"kernel": "bartlett", "bandwidth": 4}),
    ],
)
def test_jive_first_stage_rank_tests_match_2sls(cov_type, options, weighted):
    # The first stage does not depend on the estimator, only on the covariance
    # estimator, the weights and the clusters of the model
    y, exog, endog, z, w = simulate(10, ninstr=5, nendog=2, weighted=weighted)
    jive = IVJIVE(y, exog, endog, z, weights=w).fit(cov_type=cov_type, **options)
    tsls = IV2SLS(y, exog, endog, z, weights=w).fit(cov_type=cov_type, **options)
    stats_jive, stats_tsls = jive.first_stage, tsls.first_stage
    assert type(stats_jive.kleibergen_paap) is WaldTestStatistic
    assert stats_jive.kleibergen_paap.df == 4
    for name in ("kleibergen_paap_f", "cragg_donald_f"):
        assert_allclose(getattr(stats_jive, name), getattr(stats_tsls, name))
    for name in ("kleibergen_paap", "cragg_donald"):
        assert_allclose(getattr(stats_jive, name).stat, getattr(stats_tsls, name).stat)


@pytest.mark.parametrize("estimator", [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE])
@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "kernel", "clustered"])
def test_first_stage_of_every_estimator_and_covariance(estimator, cov_type):
    # The covariance configuration of the results is used by the first stage,
    # and differs by the estimator
    y, exog, endog, z, w = simulate(30, nobs=200, ninstr=3, weighted=True)
    options = {}
    if cov_type == "clustered":
        options["clusters"] = np.arange(200) // 10
    elif cov_type == "kernel":
        options["bandwidth"] = 3
    res = estimator(y, exog, endog, z, weights=w).fit(cov_type=cov_type, **options)
    first_stage = res.first_stage
    assert "First Stage Estimation Results" in str(first_stage)
    assert first_stage.diagnostics.shape == (1, 6)
    assert type(first_stage.kleibergen_paap) is WaldTestStatistic
    assert first_stage.kleibergen_paap.df == 3
    assert np.isfinite(first_stage.kleibergen_paap_f)
    assert np.isfinite(first_stage.cragg_donald_f)


@pytest.mark.parametrize("estimator", [IV2SLS, IVGMM, IVGMMCUE, IVJIVE])
def test_results_can_be_pickled(estimator):
    y, exog, endog, z, w = simulate(31, nobs=200, ninstr=3, weighted=True)
    clusters = np.arange(200) // 10
    res = estimator(y, exog, endog, z, weights=w).fit(
        cov_type="clustered", clusters=clusters
    )
    clone = pickle.loads(pickle.dumps(res))
    assert_allclose(clone.params, res.params)
    assert_allclose(clone.cov, res.cov)
    assert_same_text(str(clone.summary), str(res.summary))
    first_stage, original = clone.first_stage, res.first_stage
    assert_allclose(first_stage.kleibergen_paap.stat, original.kleibergen_paap.stat)
    assert_allclose(first_stage.kleibergen_paap_f, original.kleibergen_paap_f)
    assert_allclose(first_stage.cragg_donald_f, original.cragg_donald_f)
    if isinstance(res, IVGMMResults):
        assert_allclose(clone.centered_j_stat.stat, res.centered_j_stat.stat)
        assert_same_text(str(clone.centered_j_stat), str(res.centered_j_stat))


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


def rank_design(seed, nobs=300, ninstr=4, nendog=2, weighted=False):
    rng = np.random.default_rng(seed)
    exog = np.column_stack([np.ones(nobs), rng.standard_normal(nobs)])
    z = rng.standard_normal((nobs, ninstr))
    pi = 0.4 * rng.standard_normal((ninstr, nendog))
    scale = np.exp(0.5 * z[:, [0]])
    x = z @ pi + 0.5 * exog[:, [1]] + scale * rng.standard_normal((nobs, nendog))
    w = rng.uniform(0.3, 3.0, nobs) if weighted else None
    return x, z, exog, w


COV_CASES = [
    ("unadjusted", None),
    ("robust", None),
    ("clustered", {"clusters": np.arange(300) // 10}),
    ("kernel", {"bandwidth": 4}),
]


def all_rank_statistics(endog, instr, exog):
    out = [cragg_donald(endog, instr, exog).stat, cragg_donald_f(endog, instr, exog)]
    for cov_type, config in COV_CASES:
        out.append(kleibergen_paap(endog, instr, exog, cov_type, config).stat)
        out.append(kleibergen_paap_f(endog, instr, exog, cov_type, config))
    return np.array(out)


def test_rank_statistics_do_not_depend_on_the_scale_of_the_weights():
    # The arrays are the data multiplied by the square root of the weights
    x, z, exog, w = rank_design(1, weighted=True)
    root = np.sqrt(w)[:, None]
    base = all_rank_statistics(root * x, root * z, root * exog)
    assert np.isfinite(base).all()
    for scale in (1e-6, 1 / 37.5, 37.5, 1e6):
        factor = np.sqrt(scale)
        arrays = [factor * root * a for a in (x, z, exog)]
        assert_allclose(all_rank_statistics(*arrays), base, rtol=1e-8)


def test_first_stage_does_not_depend_on_the_scale_of_the_weights():
    y, exog, endog, z, w = simulate(2, weighted=True)
    options = {"cov_type": "clustered", "clusters": np.arange(y.shape[0]) // 10}
    base = IV2SLS(y, exog, endog, z, weights=w).fit(**options).first_stage
    scaled = IV2SLS(y, exog, endog, z, weights=37.5 * w).fit(**options).first_stage
    ones = IV2SLS(y, exog, endog, z, weights=np.ones(y.shape[0])).fit(**options)
    plain = IV2SLS(y, exog, endog, z).fit(**options)
    for name in ("kleibergen_paap_f", "cragg_donald_f"):
        assert_allclose(getattr(scaled, name), getattr(base, name), rtol=1e-10)
        assert_allclose(
            getattr(ones.first_stage, name), getattr(plain.first_stage, name)
        )
    assert_allclose(scaled.kleibergen_paap.stat, base.kleibergen_paap.stat)
    # The weights matter
    assert not np.isclose(base.kleibergen_paap_f, plain.first_stage.kleibergen_paap_f)


def test_the_constant_of_a_weighted_model_is_the_root_of_the_weights():
    # The functions use the weighted arrays, including the constant
    y, exog, endog, z, w = simulate(3, weighted=True)
    first_stage = (
        IV2SLS(y, exog, endog, z, weights=w).fit(cov_type="robust").first_stage
    )
    root = np.sqrt(w / w.mean())[:, None]
    stat = kleibergen_paap(root * endog, root * z, root * exog, "robust").stat
    assert_allclose(stat, first_stage.kleibergen_paap.stat, rtol=1e-10)
    assert_allclose(
        cragg_donald_f(root * endog, root * z, root * exog),
        first_stage.cragg_donald_f,
        rtol=1e-10,
    )
    # A column of ones is not the constant of the weighted model
    unit_constant = exog.copy()
    unit_constant[:, 0] = 1.0
    wrong = kleibergen_paap(root * endog, root * z, unit_constant, "robust").stat
    assert not np.isclose(wrong, first_stage.kleibergen_paap.stat, rtol=1e-3)


@pytest.mark.parametrize("estimator", [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("cov_type", ["unadjusted", "robust", "clustered"])
def test_first_stage_rank_tests_do_not_depend_on_the_estimator(
    estimator, cov_type, weighted
):
    y, exog, endog, z, w = simulate(12, ninstr=4, nendog=2, weighted=weighted)
    options = {}
    if cov_type == "clustered":
        options["clusters"] = np.arange(y.shape[0]) // 10
    reference = IV2SLS(y, exog, endog, z, weights=w).fit(cov_type=cov_type, **options)
    res = estimator(y, exog, endog, z, weights=w).fit(cov_type=cov_type, **options)
    expected, actual = reference.first_stage, res.first_stage
    assert_allclose(actual.kleibergen_paap.stat, expected.kleibergen_paap.stat)
    assert_allclose(actual.kleibergen_paap_f, expected.kleibergen_paap_f)
    assert_allclose(actual.cragg_donald.stat, expected.cragg_donald.stat)
    assert_allclose(actual.cragg_donald_f, expected.cragg_donald_f)


@pytest.mark.parametrize(("ninstr", "nendog"), [(4, 2), (5, 1), (6, 2), (5, 3)])
def test_number_of_clusters_needed(ninstr, nendog):
    # The covariance of the LM statistic has dimension ninstr - nendog + 1, and
    # that of the Wald F statistic has scores with a mean of zero, so that it
    # needs one more cluster. The LM statistic is computed without the Wald
    # covariance, and so does not warn if it is full rank
    x, z, exog, _ = rank_design(11, nobs=240, ninstr=ninstr, nendog=nendog)
    df = ninstr - nendog + 1

    def config(nclusters):
        return {"clusters": np.arange(x.shape[0]) % nclusters}

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*rank deficient.*")
        lm = kleibergen_paap(x, z, exog, "clustered", config(df))
        wald = kleibergen_paap_f(x, z, exog, "clustered", config(df + 1))
    assert lm.df == df
    assert np.isfinite(lm.stat)
    assert np.isfinite(wald)
    with pytest.warns(UserWarning, match="rank deficient"):
        short = kleibergen_paap_f(x, z, exog, "clustered", config(df))
    assert np.isfinite(short)
    with pytest.warns(UserWarning, match="rank deficient"):
        lm_short = kleibergen_paap(x, z, exog, "clustered", config(df - 1))
    assert lm_short.df == df - 1


def test_number_of_clusters_needed_through_the_model():
    # The covariance of the model has to be full rank, which needs more clusters
    y, exog, endog, z, w = simulate(13, nobs=240, ninstr=6, nendog=2, weighted=True)
    df = 5

    def first_stage(nclusters):
        clusters = np.arange(y.shape[0]) % nclusters
        mod = IV2SLS(y, exog, endog, z, weights=w)
        return mod.fit(cov_type="clustered", clusters=clusters).first_stage

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*rank deficient.*")
        assert first_stage(df).kleibergen_paap.df == df
        assert np.isfinite(first_stage(df + 1).kleibergen_paap_f)
    with pytest.warns(UserWarning, match="rank deficient"):
        _ = first_stage(df).kleibergen_paap_f
    with pytest.warns(UserWarning, match="rank deficient"):
        _ = first_stage(df - 1).kleibergen_paap


def test_cragg_donald_ignores_the_covariance_estimator_and_clusters():
    y, exog, endog, z, w = simulate(15, weighted=True)
    mod = IV2SLS(y, exog, endog, z, weights=w)
    clusters = np.arange(y.shape[0]) // 10
    fits = [
        mod.fit(cov_type="unadjusted"),
        mod.fit(cov_type="robust"),
        mod.fit(cov_type="clustered", clusters=clusters),
        mod.fit(cov_type="kernel", bandwidth=4),
    ]
    cragg = [fit.first_stage.cragg_donald.stat for fit in fits]
    cragg_f = [fit.first_stage.cragg_donald_f for fit in fits]
    assert_allclose(cragg, cragg[0])
    assert_allclose(cragg_f, cragg_f[0])
    # and the Kleibergen-Paap statistic uses it
    kp = [fit.first_stage.kleibergen_paap.stat for fit in fits]
    assert len({round(value, 6) for value in kp}) == len(kp)


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
    # The uncentered covariance has nclusters. The objective function of the CUE
    # is constant if there are as many clusters as moment conditions, so that the
    # CUE needs 6 clusters, which are enough for the centered covariance too, and
    # the case where only the uncentered covariance can be estimated is the
    # two-step estimator
    y, exog, endog, z, w = simulate(23, ninstr=3, violation=0.3, weighted=weighted)
    outcomes = {}
    for nclusters in (5, 6):
        config = {
            "weights": w,
            "weight_type": "clustered",
            "clusters": np.arange(y.shape[0]) % nclusters,
            "center": False,
        }
        mod = IVGMM(y, exog, endog, z, **config)
        res = mod.fit()
        outcomes[nclusters] = res
        assert np.isfinite(res.j_stat.stat)
        assert res.j_stat.df == 2
        cue = IVGMMCUE(y, exog, endog, z, **config)
        if nclusters == 5:
            with pytest.raises(ValueError, match="as many clusters"):
                cue.fit()
        else:
            assert type(cue.fit().centered_j_stat) is WaldTestStatistic
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
