Confidence intervals with a weak instrument
===========================================

:meth:`~linearmodels.iv.results.IVResults.tf_conf_int` constructs a tF interval
for one endogenous regressor and one excluded instrument. It is
an explicit alternative to the conventional :meth:`~linearmodels.iv.results.IVResults.conf_int`.
It supports IV2SLS with unadjusted or heteroskedasticity-robust covariance,
``debiased=False``, and constant observation weights. Other configurations raise
``ValueError``. Included exogenous controls are allowed.

Both table lookup and numerical calibration require the optional
`tf-inference <https://pypi.org/project/tf-inference/>`_ package, Python
3.12 or later, NumPy 1.26 or later, SciPy 1.12 or later and PyArrow 18 or later.
Install the validated release before running the example::

    python -m pip install tf-inference==0.1.0

The package is imported only when tF inference is requested. Ordinary model
fitting and conventional inference do not require it. If it is missing,
``tf_conf_int`` raises ``ImportError`` with installation guidance.

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
    >>> interval_99 = result.tf_conf_int(level=0.99)
    >>> bool(interval_99.loc["educ", "lower"] < interval.loc["educ", "lower"])
    True
    >>> bool(interval_99.loc["educ", "upper"] > interval.loc["educ", "upper"])
    True

The first-stage statistic is the squared t statistic for the excluded instrument,
using the fitted model's covariance convention. The coefficient and SE are reused.
Neither the original SE nor the conventional p-values and intervals are changed.

The default ``method="table"`` calls tf-inference's public ``critical_value``
function, using that package's tables generated from Appendix B.3
of Lee, McCrary, Moreira and Porter's
`Valid t-Ratio Inference for IV <https://doi.org/10.1257/aer.20211063>`_.
The `author manuscript <https://eml.berkeley.edu/~jmccrary/LMMP_2_15_22.pdf>`_
contains the construction. The package owns the tables, interpolation and
caching; linearmodels does not bundle calibration data. Once installed, table
lookup requires no network access or numerical calibration.

``level`` is a confidence level, matching ``conf_int(level=...)``; the default
is 0.95. The validated package's table levels are 0.75, 0.8, 0.85, 0.875,
0.9, 0.925, 0.94, 0.95,
0.96, 0.97, 0.975, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995 and 0.9999.
No interpolation across confidence levels or automatic numerical computation
is performed. An unavailable level raises an error.

For an unconventional level, explicitly request numerical calibration::

    interval = result.tf_conf_int(level=0.963, method="compute")

The first computation can take several seconds; the calibration is cached by
level for later calls. The numerical implementation accepts levels between
0.75 and 0.9999. For custom numerical settings, inspection of calibration
diagnostics, or export of reusable constants, use tf-inference directly with
``alpha=1-level``. The finite calibration search should be checked beyond its
search settings before relying on an unconventional level for research.

The unbounded-interval threshold is the squared normal critical value for the
requested level. At 95% it is about 3.84145882. F strictly above this boundary
gives finite bounds, including the interval between this boundary and 4.
For F=10 the 95% critical value is about 3.4303537; an estimate of 1 with SE=0.4
therefore gives bounds of approximately [-0.37214, 2.37214]. Each confidence
level has its own calibrated plateau. At 99%, for example, its critical value
is about 2.72549, above the normal 99% critical value of 2.57583.

The calibration package interpolates the regularized squared critical value
``(F - boundary) * critical(F)**2``, which has a finite limit at the boundary.
The tables' license, checksums, construction settings and validation provenance
are maintained in tf-inference.

Calibration uses deterministic integration, a finite nuisance-parameter grid
and local searches. Held-out checks and grid refinement assess the numerical
approximation; they do not prove a uniform size bound. This implementation does
not claim the conservative rounding guarantee of the paper's printed-table
interpolation. It is not a p-value calculation or a guarantee of finite-sample
coverage. Instrument validity and the
model's other identifying and sampling assumptions still matter. The newer VtF
procedure uses additional information and can yield shorter intervals; this
method implements tF, not VtF. The existing ``anderson_rubin`` property is a test
of overidentification, not a weak-instrument confidence set for this coefficient.
