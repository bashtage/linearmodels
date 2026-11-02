Confidence intervals with a weak instrument
==========================================

:meth:`~linearmodels.iv.results.IVResults.tf_conf_int` constructs a conservative
95% tF interval for one endogenous regressor and one excluded instrument. It is
an explicit alternative to the conventional :meth:`~linearmodels.iv.results.IVResults.conf_int`.
It supports IV2SLS with unadjusted or heteroskedasticity-robust covariance,
``debiased=False``, and constant observation weights. Other configurations raise
``ValueError``. Included exogenous controls are allowed.

For example, use proximity to a four-year college as an instrument for education:

.. doctest::

    >>> from linearmodels.datasets import card
    >>> from linearmodels.iv import IV2SLS
    >>> data = card.load()
    >>> columns = ["lwage", "educ", "exper", "expersq", "black", "south", "smsa", "nearc4"]
    >>> data = data[columns].dropna()
    >>> formula = "lwage ~ 1 + exper + expersq + black + south + smsa + [educ ~ nearc4]"
    >>> result = IV2SLS.from_formula(formula, data).fit(cov_type="robust")
    >>> interval = result.tf_conf_int()
    >>> list(interval.index), list(interval.columns)
    (['educ'], ['lower', 'upper'])
    >>> conventional = result.conf_int().loc["educ"]
    >>> bool(interval.loc["educ", "lower"] <= conventional["lower"])
    True
    >>> bool(interval.loc["educ", "upper"] >= conventional["upper"])
    True

The first-stage statistic is the squared t statistic for the excluded instrument,
using the fitted model's covariance convention. The coefficient and SE are reused.
Neither the original SE nor the conventional p-values and intervals are changed.

The implementation interpolates in F using the upward-rounded values in Table 3a
of the `February 16, 2022 author version
<https://eml.berkeley.edu/~jmccrary/LMMP_2_15_22.pdf>`_ of Lee, McCrary, Moreira and
Porter's `Valid t-Ratio Inference for IV <https://doi.org/10.1257/aer.20211063>`_.
Section 3.2 justifies this conservative interpolation by convexity. It uses an
unbounded interval below F=4, including the small range where the smooth tF
interval is bounded, and a critical value of 1.96 at and above F=104.67.

This adjustment is calibrated only for 95% confidence. It is not a p-value
calculation or a guarantee of finite-sample coverage. Instrument validity and the
model's other identifying and sampling assumptions still matter. The newer VtF
procedure uses additional information and can yield shorter intervals; this
method implements tF, not VtF. The existing ``anderson_rubin`` property is a test
of overidentification, not a weak-instrument confidence set for this coefficient.
