Change Log
==========

Version 4.20
------------
* Correct calculation of first-stage F-statistic in IV models.

Version 4.19
------------
* Minor release to fix a wheel-building issue on Python 3.9

Version 4.18
------------
* Improved performance of :meth:`~linearmodels.iv.absorbing.AbsorbingLS.fit` by
  deferring some operations.
* Added support for the method available in `PyHDFE <https://pypi.org/project/pyhdfe>`_  in
  :class:`~linearmodels.iv.absorbing.AbsorbingLS`. These methods can only be
  used when the variables absorbed are categorical (i.e., fixed-effects only) and
  when the model is unweighted.
* Added alternative measures of :math:`R^2` using the squared correlation. See
  :meth:`~linearmodels.panel.results.PanelResults.corr_squared_overall`,
  :meth:`~linearmodels.panel.results.PanelResults.corr_squared_within`, and
  :meth:`~linearmodels.panel.results.PanelResults.corr_squared_between` (:issue:`157`).
* Added a clustered covariance estimator
  (:class:`linearmodels.system.covariance.ClusteredCovariance`) for system regressions
  (:issue:`241`).
* Fixed a bug in :class:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`
  which used incorrect values for a tuning parameter in the bandwidth estimation
  for the Parzen and Quadratic Spectral kernels (:issue:`242`).

Version 4.17
------------
* Fixed various typing issues (:issue:`239`, :issue:`240`).

Version 4.16
------------
* Verify typing using mypy (:issue:`232`, :issue:`234`, :issue:`235`, :issue:`238`).
* Added typing to all public-facing classes and methods (:issue:`228`, :issue:`229`).
* Added :class:`~linearmodels.panel.results.FamaMacBethResults` which has
  the property :meth:`~linearmodels.panel.results.FamaMacBethResults.all_params`
  that contains a (nobs, nparam) DataFrame of parameters estimated in each time
  period (:issue:`230`).

Version 4.15
------------
* Blackened the code.
* Added McElroy's and Berndt's measures of system fit (:issue:`215`).
* Removed support for Python 3.5 inline with NEP-29 (:issue:`222`).

Version 4.14
------------
* Fixed issue where datasets were not installed with wheels (:issue:`217`).
* Switched to property-cached to inherit cached property from property (:issue:`211`).
* Removed all use of :class:`pandas.Panel` (:issue:`211`).

Version 4.13
------------
* Added :class:`~linearmodels.iv.absorbing.AbsorbingLS` which allows a large number
  of variables to be absorbed. This model can handle very high-dimensional dummy
  variables and has been tested using up to 1,000,000 categories in a data set
  with 5,000,000 observations.
* Fixed a bug when estimating weighted panel models that have repeated observations
  (i.e., more than one observation per entity and time id).
* Added ``drop_absorbed`` option to :class:`~linearmodels.panel.model.PanelOLS`
  which automatically drops variables that are absorbed by fixed effects.
  (:issue:`206`)
* Added optional Cythonized node selection for dropping singletons
* Added preconditioning to the dummy variable matrix when ``use_lsmr-True``
  in :func:`~linearmodels.panel.model.PanelOLS.fit`. In models with many
  effects, this can reduce run time by a factor of 4 or more.

Version 4.12
------------
* Added an option to drop singleton observations in
  :class:`~linearmodels.panel.model.PanelOLS` by setting the keyword argument
  ``singletons-False``. When ``False``, singelton observations are dropped
  before the model is fit, so the the result is *as-if* the observations were
  never in ``exog`` or ``dependent``.
* Added a method to construct the 2-core graph for 2-way effects models, which
  allows singleton observations with no effect on estimated slopes to be
  excluded. (:issue:`191`)
* Added support for LSMR estimation of parameters in
  :func:`~linearmodels.panel.model.PanelOLS.fit` through the keyword argument
  ``use_lsmr``. LSMR is a sparse estimation method that can be used to extend
  :class:`~linearmodels.panel.model.PanelOLS` to more than two effects.
* Fixed a bug where IV models estimated with only exogenous regressors where
  not being correctly labeled as OLS models in output. (:issue:`185`)
* Added ``wald_test`` to panel-model results.
* Renamed ``test_linear_constraint`` to ``wald_test``
* Added a low-memory option to :func:`~linearmodels.panel.model.PanelOLS.fit`
  that avoids constructing dummy variables. Only used when both ``entity_effects``
  and ``time_effects`` are ``True``. By default, the low memory algorithm will be
  used whenever constructing the dummy variable array would require more than
  1 GiB. (:issue:`182`)
* Added an option in model comparison (:func:`~linearmodels.iv.results.compare` and
  :func:`~linearmodels.panel.results.compare`) to report standard errors or pvalues
  instead of t-stats. (:issue:`178`)

Version 4.11
------------
* Fixed a bug which did not correctly check the rank of the
  cross-section regression in :class:`~linearmodels.panel.model.FamaMacBeth` (:issue:`176`)
* Fixed a bug which failed to correctly check rank conditions when
  specifying asset pricing models (:issue:`173`)
* Switched to external package cached-property to manage caching instead of
  custom and less-well-tested solution (:issue:`172`)

Version 4.10
------------
* Fixed a bug where weights were incorrectly calculated for HAC covariances
  when the weight function was ``'parzen'`` or ``'gallant'`` (:issue:`170`)

Version 4.9
-----------
* Changed the return type of Wooldridge's over identification test when
  invalid to ``InvalidTestStatistic``
* Add typing information to IV models
* Allow optimization parameters to be passed to :class:`~linearmodels.iv.model.IVGMMCUE`
* Removed internal use of pandas Panel
* Improved performance in panel models when using
  :func:`~linearmodels.panel.model.PanelOLS.from_formula`
* Switched to retaining index column names when original input index is named
* Modified tests that were not well conceived
* Added spell check to documentation build
* Improve docstring for ``summary`` properties

Version 4.8
-----------
* Corrected bug that prevented single character names in IV formulas
* Corrected kappa estimation in LIML when there are no exogenous regressors

Version 4.7
-----------
* Improved performance of Panel estimators by optimizing data structure
  construction

Version 4.6
-----------
* Added a license

Version 4.5
-----------
* Added System GMM estimator
* Added automatic bandwidth for kernel-based GMM weighting estimators
* Cleaned up HAC estimation across models
* Added ``predict`` method to IV, Panel and System model to allow out-of-sample
  prediction and simplify retrieval of in-sample results
* Fixed small issues with Fama-MacBeth which previously ignored weights

Version 4.0
-----------
* Added Seemingly Unrelated Regression (SUR) Estimator
* Added Three-stage Least Squares (3SLS) Estimator

Version 3.0
-----------
* Added Fama-MacBeth estimator for panels
* Added linear factor models for asset pricing applications

  * Time-series estimation using traded factors
  * 2- and 3-step estimation using OLS
  * GMM Estimation

Version 2.0
-----------
* Added panel models -- fixed effects, random effects, between,
  first difference and pooled OLS.
* Addition of two-way clustering to some of the IV models (2SLS, LIML)

Version 1.0
-----------
* Added Instrumental Variable estimators -- 2SLS, LIML and
  k-class, GMM and continuously updating GMM.
