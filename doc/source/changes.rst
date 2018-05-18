Change Log
----------

Since 4.7
=========
* Corrected bug that prevented single character names in IV formulas
* Corrected kappa estimation in LIML when there are no exogenous regressors

Version 4.7
===========
* Improved performance of Panel estimators by optimizing data structure
  construction

Version 4.6
===========
* Added a license

Version 4.5
===========
* Added System GMM estimator
* Added automatic bandwidth for kernel-based GMM weighting estimators
* Cleaned up HAC estimation across models
* Added `predict` method to IV, Panel and System model to allow out-of-sample
  prediction and simplify retrieval of in-sample results
* Fixed small issues with Fama-MacBeth which previously ignored weights

Version 4.0
===========
* Added Seemingly Unrelated Regression (SUR) Estimator
* Added Three-stage Least Squares (3SLS) Estimator

Version 3.0
===========
* Added Fama-MacBeth estimator for panels
* Added linear factor models for asset pricing applications

  * Time-series estimation using traded factors
  * 2- and 3-step estimation using OLS
  * GMM Estimation

Version 2.0
===========
* Added panel models -- fixed effects, random effects, between,
  first difference and pooled OLS.
* Addition of two-way clustering to some of the IV models (2SLS, LIML)

Version 1.0
===========
* Added Instrumental Variable estimators -- 2SLS, LIML and
  k-class, GMM and continuously updating GMM.