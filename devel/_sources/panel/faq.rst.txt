.. _panel-implementation-choices:

Implementation Choices
----------------------

While the implementation of the panel estimators is similar to Stata, there
are some differenced worth noting.

Clustered Covariance with Fixed Effects
=======================================
When using clustered standard errors and entity effects, it is not necessary
to adjust for estimated effects. ``PanelOLS`` attempts to detect when this is
the case and automatically adjust the degree of freedom. This can be
overridden using by setting the fit option ``auto_df=False`` and then
changing the value of ``count_effects``.

.. _panel-input-alignment:

Aligning Inputs
===============
The rows of ``dependent``, ``exog``, ``weights``, ``other_effects`` and
the ``clusters`` used in covariance estimation are paired by position. To
prevent a value from being attached to the wrong observation, an input that
carries labels (a pandas ``Series`` or ``DataFrame``, an xarray ``DataArray``,
or :class:`~linearmodels.panel.data.PanelData`) must use the same labels, in
the same order, as ``dependent``. A ``ValueError`` is raised if the labels
differ, for example when the values of a regressor have been reversed or sorted
differently from the outcome. Reindex the input to the index of ``dependent``
to correct this.

NumPy arrays do not have labels and are always paired by position. The same
is true of a pandas object that only has a default index, 0, 1, ..., n-1, (and
default columns, if it is a ``DataFrame`` without a MultiIndex) since there are
no labels to compare. Use ``to_numpy()`` to explicitly pair the values of any
other pandas object by position.

Weights can be provided in any of the following shapes, where ``nobs`` is the
number of time periods and ``nentity`` is the number of entities.

* A single weight for each observation, either as a 1-d array with
  ``nobs * nentity`` values or as a Series or DataFrame with the same MultiIndex
  as ``dependent``.
* A single weight for each time period, as a 1-d array or Series with ``nobs``
  values. The weight is used for every entity.
* A single weight for each entity, as a 1-d array or Series with ``nentity``
  values. The weight is used for every time period.
* A weight for each time period and entity, as a 2-d array or a DataFrame with
  ``nobs`` rows and ``nentity`` columns.

Weights that are defined for each time period, for each entity or for each
combination of the two are matched to the time periods and entities in the
order that they first appear in ``dependent``. This is the case even if the
rows of ``dependent`` are not sorted by entity and then time. When these
weights are a pandas object, the index (and columns) must contain the time
periods and entities in this order.

:math:`R^2` definitions
=======================
The :math:`R^2` definitions are all designed so that the reported value will
match the original model using the estimated parameters.  This differs from
other packages, such as Stata, which use a correlation based measure which
ignores the estimated intercept (if included) and allows for affine
adjustments to estimated parameters. The main reported :math:`R^2`
(``rsquared`` in returned results) is always the :math:`R^2` from
the actual model fit, after adjusting the data for:

* weights (all estimators)
* effects (:class:`~linearmodels.panel.model.PanelOLS`)
* re-centering (:class:`~linearmodels.panel.model.RandomEffects`)
* within entity aggregation (:class:`~linearmodels.panel.model.BetweenOLS`)
* differencing (:class:`~linearmodels.panel.model.FirstDifferenceOLS`)
