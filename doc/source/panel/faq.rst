.. _panel-implementation-choices:

Implementation Choices
----------------------

While the implmentation of the panel estimators is similar to Stata, there
are some differenced worth noting.

Clustered Covariance with Fixed Effects
=======================================
When suing clustered standard errors and entity effects, it isn't necessary
to adjust for estimated effects. ``PanelOLS`` attempts to detect when this is
the case and automatically adjust the degree of freedom. This can be
overridden using by setting the fit option ``auto_df=False`` and then
changing the value of ``count_effects``.

R2 definitions
==============
The :math:`R^2` definitions are all designed so that the reported value will
match the original model using the estimated parameters.  This differs from
other packages, such as Stata, which use a correlation based measure which
ignores the estimated intercept (if included) and allows for affine
adjustments to estimated parameters. The main reported :math:`R^2`
(``rsquared`` in returned results) is always the :math:`R^2` from
the actual model fit, after adjusting the data for:

* weights (all estimators)
* effects (:class:`~linearmodels.panel.model.PanelOLS`)
* recentering (:class:`~linearmodels.panel.model.RandomEffects`)
* within entity aggregation (:class:`~linearmodels.panel.model.BetweenOLS`)
* differencing (:class:`~linearmodels.panel.model.FirstDifferenceOLS`)
