.. _panel-implementation-choices:

Implementation Choices
----------------------

While the implmentation of the panel estimators is similar to Stata, there
are some differenced worth noting.

Clustered Covariance with Fixed Effects
=======================================
By default, Stata does not adjust the covariance estimator for the degrees
of freedom used to estimate included effects.  This can have a substantial
effect on the estimated parameter covariance when the number of time periods
is small (e.g. 2 or 3).  It also means running LSDV and using a model with
fixed effects will produce different results. By default, covariance estimators
are always adjusted for degrees of freedom consumed by effects.  This can be
overridden using by setting the fit option ``count_effects=False``.

R2 definitions
==============
