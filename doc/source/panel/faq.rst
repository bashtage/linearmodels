.. _panel-implementation-choices:

Implementation Choices
----------------------

While the implementation of the panel estimators is similar to Stata, there
are some differenced worth noting.

.. _panel-small-sample:

Small-sample Adjustments
========================
The ``fit`` method of every panel estimator has ``debiased=True`` as its
default. This is the case for :class:`~linearmodels.panel.model.PooledOLS`,
:class:`~linearmodels.panel.model.PanelOLS`,
:class:`~linearmodels.panel.model.BetweenOLS`,
:class:`~linearmodels.panel.model.FirstDifferenceOLS`,
:class:`~linearmodels.panel.model.RandomEffects` and
:class:`~linearmodels.panel.model.FamaMacBeth`. The IV estimators, including
:class:`~linearmodels.iv.absorbing.AbsorbingLS`, have ``debiased=False``.

With ``debiased=True``, the covariance is scaled by :math:`n/(n-k)`, where
:math:`n` is the number of observations and :math:`k` is the number of
regressors. The p-values and the confidence intervals of the parameters use a
Student's t distribution with the residual degrees of freedom, and the joint
tests of significance use an F distribution. With ``debiased=False`` the
covariance is not scaled by this factor, and the normal and :math:`\chi^2`
distributions are used. :class:`~linearmodels.panel.model.PanelOLS` is an
exception, see `Counting the Effects`_.

The clustered covariance, ``cov_type="clustered"``, has a second adjustment,
which is for the number of clusters. ``group_debias=True`` multiplies the
covariance by :math:`g/(g-1) \times (n-1)/n`, where :math:`g` is the number of
distinct clusters. With two-way clustering, it is applied to each of the three
one-way covariances that are combined. The default is ``group_debias=False``.

Matching Stata
**************
Stata's ``regress`` with ``vce(cluster)`` scales the clustered covariance by
:math:`g/(g-1) \times (n-1)/(n-k)`. This is ``debiased=True``, which is the
default, together with ``group_debias=True``.

.. code-block:: python

    from linearmodels import PooledOLS
    from linearmodels.datasets import wage_panel

    data = wage_panel.load().set_index(["nr", "year"])
    mod = PooledOLS.from_formula("lwage ~ 1 + expersq + married + union", data)
    res = mod.fit(
        cov_type="clustered", cluster_entity=True, debiased=True, group_debias=True
    )

The standard errors agree with those of the same regression in R,
``sandwich::vcovCL(model, cluster = ~nr, type = "HC1", cadjust = TRUE)``,
which uses the same formula. The default, ``group_debias=False``, gives smaller
standard errors, which matters when there are few clusters. If the clusters are
defined by a variable with two values, the ratio of the default standard errors
to those of Stata is :math:`\sqrt{n(g-1)/((n-1)g)}`, which is close to
:math:`\sqrt{1/2}`, or 71%, if :math:`g=2` and :math:`n` is large. Few clusters,
such as two, give unreliable standard errors whichever adjustment is used. The
tests of the package that compare clustered covariances with Stata's
``regress`` and ``xtreg`` use ``debiased=True`` and ``group_debias=True``.

``xtreg`` and ``regress`` with a dummy variable for each entity count the effects
differently when the effects are nested in the clusters, so expect differences if
you compare the estimators with both commands.

Counting the Effects
********************
:class:`~linearmodels.panel.model.PanelOLS` counts the effects that it estimates
in its degrees of freedom, which is the ``extra_df`` of its covariance
estimators. If :math:`d` is the number of effects, the covariance is scaled by
:math:`n/(n-d-k)` if ``debiased`` is ``True``, and by :math:`n/(n-d)` if it is
``False``, so that the effects are counted whether or not ``debiased`` is
``True``. With ``debiased=True``, the unadjusted and robust standard errors are
those of the regression that includes a dummy variable for each effect. The
number of effects is the number of entities for entity effects, the number of
time periods for time effects and the number of groups for each of the other
effects. One is subtracted from the number of each of the effects after the first,
and from the number of the first if the model has a constant, because the
effects are not separately identified. Counting the effects is not needed when
they are nested in the clusters, which is described next.

Clustered Covariance with Fixed Effects
=======================================
When using clustered standard errors and entity effects, it is not necessary
to adjust for estimated effects. ``PanelOLS`` attempts to detect when this is
the case and automatically adjust the degree of freedom. This is done when
the model has a single effect and the effect is nested in the clusters. The
effects are counted in all other cases. This can be overridden by setting the
fit option ``auto_df=False`` and then changing the value of ``count_effects``.

An effect is nested in a cluster variable if each of its groups is in one group
of the variable. Entity effects are nested in the clusters of ``cluster_entity=True``,
and in clusters that are the industries of the entities if every entity is in one
industry. The effect is nested in the clusters if it is nested in **every**
cluster variable. This is the rule of :class:`~linearmodels.iv.absorbing.AbsorbingLS`.
It does not depend on the order of the cluster variables, which is why it is not
enough that the effect is nested in one of them. For example:

* Entity effects with ``cluster_entity=True``, or with the clusters (industry,
  entity) when each entity is in one industry, are nested. The effects are not
  counted.
* Entity effects with the clusters (entity, time), the two-way clustering by
  ``cluster_entity=True, cluster_time=True``, are not nested, since an entity
  has observations in several periods. The effects are counted. Time effects with
  these clusters are not nested for the same reason.
* A model with more than one effect, for example entity and time effects, counts
  its effects with every clustered covariance.

.. versionchanged:: 7.1

   The effect has to be nested in every cluster variable. Earlier versions kept
   the result for the last cluster variable only, so that the standard errors of the
   same model depended on the order of the variables, for example entity effects with
   the clusters (time, entity) were treated as nested and with (entity, time)
   as not nested.

.. _panel-ac-lags:

Gaps in the Periods of an Entity
================================
Some estimators use the order of the observations of an entity. They treat an
entity that is not observed in some period between two of its observations (a gap)
in different ways from other packages.

The autocorrelation robust covariance, ``cov_type="ac"``, counts lags as positions
in the observations of the entity, in the order of time. Two observations on both sides of
a gap are one lag apart. The cluster-HAC covariance of statsmodels
(``cov_type="hac-panel"``) does the same. plm (``vcovNW``) and sandwich (``vcovPL``)
count calendar periods, so that the observations are two lags apart if one period is
missing between them. The covariances are the same for an entity without gaps. In a
panel where about a quarter of the periods are missing, the standard errors of plm
were about 3% different from the standard errors of this package and of statsmodels.
See :class:`~linearmodels.panel.covariance.ACCovariance`, which also describes the weights
of the lags of entities that are shorter than the bandwidth.

:class:`~linearmodels.panel.model.FirstDifferenceOLS` differences adjacent periods only.
An observation that follows a gap does not have a difference, and the estimator uses the
differences between adjacent periods of the panel, which are the distinct time labels of
the data. plm differences consecutive observations of an entity, also across a gap, so
that its differences can span several periods. In the panel with the missing periods
above plm had 210 differences and this package 167, and the coefficient of the first
regressor was 0.474 and 0.562. The estimates are the same without gaps.

.. _panel-estimated-effects:

Estimated Effects
=================
``estimated_effects`` in the results is the combined effect of all of the
effects in the model. It is a ``DataFrame`` with one column, the sum of the
entity, time and other effects of each observation, and the effects are not
available separately. It is ``NaN`` for
:class:`~linearmodels.panel.model.PooledOLS`,
:class:`~linearmodels.panel.model.FirstDifferenceOLS` and
:class:`~linearmodels.panel.model.FamaMacBeth`, and 0 for a
:class:`~linearmodels.panel.model.PanelOLS` model without effects. It adds to
``fitted_values`` and ``idiosyncratic`` to give the dependent variable.

The effects do not have standard errors or any other inference, since they are
not estimated consistently when the number of time periods is fixed. The effect
of an entity is estimated from its :math:`T` observations, so the error does not
shrink when the number of entities grows. In a simulation with
:math:`T=5` and 200, 2,000 and 20,000 entities, the standard deviation of the
error of the estimated entity effects was 0.441, 0.452 and 0.445, which is
close to :math:`\sigma/\sqrt{T}=0.447`, where :math:`\sigma=1` is the standard
deviation of the idiosyncratic shocks. If you need inference for the effects of
a small number of entities, include the effects as dummy variables in the
regression, for example in :class:`~linearmodels.panel.model.PooledOLS`, and use
the results for the parameters of the dummy variables.

The estimated effects of :class:`~linearmodels.panel.model.RandomEffects` are
:math:`\hat{\theta}_i` times the entity mean of :math:`y-x^{\prime}\hat{\beta}`,
where :math:`\hat{\theta}_i = 1-\sqrt{\hat{\sigma}^2_\epsilon/(T_i\hat{\sigma}^2_u +
\hat{\sigma}^2_\epsilon)}` and :math:`\hat{\sigma}^2_u` is the estimated variance
of the random effect. The estimate is set to 0 if it is negative. Then
:math:`\hat{\theta}_i=0`, the model is the same as pooled OLS, and all of the
estimated effects are 0. This is a property of the data, not a bug: the
variation between the entities is not larger than the idiosyncratic shocks
imply. The ``Effects`` entry of ``variance_decomposition`` is 0 and ``theta`` is
0 in the results when this happens.

Splitting Entity and Time Effects
*********************************
The sum of an entity effect and a time effect is exactly additive, so the two
can be recovered from the combined effect, up to a constant that can be moved
from one to the other. The function below regresses the estimated effects on a
set of dummy variables for the entities and a set for the time periods, using
sparse matrices. The fit is exact, also in an unbalanced panel. Compare the
differences of the effects, or normalize them, for example by setting the first
time effect to 0.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.sparse import csr_array, hstack
    from scipy.sparse.linalg import lsqr

    from linearmodels import PanelOLS
    from linearmodels.datasets import wage_panel


    def split_effects(res):
        effects = res.estimated_effects.iloc[:, 0]
        entity = pd.Categorical(effects.index.get_level_values(0))
        time = pd.Categorical(effects.index.get_level_values(1))

        def dummies(cat):
            n = len(cat)
            return csr_array(
                (np.ones(n), (np.arange(n), cat.codes)), shape=(n, len(cat.categories))
            )

        design = hstack([dummies(entity), dummies(time)]).tocsr()
        solution = lsqr(
            design, effects.to_numpy(), atol=1e-14, btol=1e-14, iter_lim=10000
        )[0]
        k = len(entity.categories)
        return (
            pd.Series(solution[:k], index=entity.categories),
            pd.Series(solution[k:], index=time.categories),
        )


    data = wage_panel.load().set_index(["nr", "year"])
    exog = sm.add_constant(data[["expersq"]])
    mod = PanelOLS(data.lwage, exog, entity_effects=True, time_effects=True)
    entity_effects, time_effects = split_effects(mod.fit())
    time_effects - time_effects.iloc[0]

.. _panel-absorbing-ls:

AbsorbingLS or PanelOLS
=======================
:class:`~linearmodels.panel.model.PanelOLS` has up to two effects. It has the
statistics of a panel estimator, and inference that matches Stata's ``xtreg``.
:class:`~linearmodels.iv.absorbing.AbsorbingLS` can absorb any number of effects,
and effects interacted with continuous variables, and it is designed for models
with very many groups. It is not a panel estimator, and it reports the
statistics of a regression. See :ref:`iv-absorbing-ls` for more.

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

Dates are compared by value, so the same dates are the same labels when they
are stored at different resolutions, for example ``datetime64[us]`` and
``datetime64[ns]``. This applies to the time periods of the weights that are
given for each time period, and to every other input. See
:ref:`input-alignment`.

NumPy arrays do not have labels and are always paired by position. The same
is true of a pandas object that only has a default index, 0, 1, ..., n-1, (and
default columns, if it is a ``DataFrame`` without a MultiIndex) since there are
no labels to compare. Use ``to_numpy()`` to explicitly pair the values of any
other pandas object by position.

If ``dependent`` does not carry labels, for example if it is a NumPy array, the
first of ``exog``, ``weights`` and ``other_effects`` that carries labels is the
reference instead. Each of the inputs after it that carries labels, and
``clusters``, must use the same labels, in the same order, or a ``ValueError``
that names both inputs is raised. If none of the inputs carries labels, all of
them are paired by position. A list or a tuple is accepted wherever an array is
accepted for ``weights``, ``clusters`` and ``other_effects``. It does not carry
labels.

Weights can be provided in any of the following shapes, where ``nobs`` is the
number of time periods, ``nentity`` is the number of entities and ``nrows`` is the
number of rows of ``dependent``. ``nrows`` is ``nobs * nentity`` if ``dependent`` is
balanced and less if it is not.

* A single weight for each observation, either as a 1-d array with ``nrows``
  values or as a Series or DataFrame with the same MultiIndex as ``dependent``.
* A single weight for each time period, as a 1-d array or Series with ``nobs``
  values. The weight is used for every entity.
* A single weight for each entity, as a 1-d array or Series with ``nentity``
  values. The weight is used for every time period.
* A weight for each time period and entity, as a 2-d array or a DataFrame with
  ``nobs`` rows and ``nentity`` columns.

The number of values of a 1-d array decides how it is read, whether or not
``dependent`` is balanced.

* An array with ``nrows`` values is a weight for each observation. Its values are
  paired with the rows of ``dependent`` by position, in the order of the rows,
  whatever that order is, as a 1-d array of ``clusters`` or ``other_effects`` is. They
  are paired before the observations with missing values are dropped, so the array has
  a value for every row of ``dependent``, including the rows with a missing value of
  ``dependent`` or ``exog``.
* An array with ``nobs`` values is a weight for each time period, and an array with
  ``nentity`` values is a weight for each entity.
* If ``nobs`` and ``nentity`` are equal, an array with that many values could be
  either, and an ``AmbiguityError`` is raised. Give the weights as a 2-d array or for
  each row instead.
* If ``dependent`` is not balanced and has as many rows as time periods or entities,
  which happens if every time period or entity has a single observation, an array with
  that many values is a weight for each observation.
* Any other number of values raises a ``ValueError`` that gives the numbers that are
  accepted. An array with ``nobs * nentity`` values is a weight for each observation
  only if ``dependent`` is balanced. If it is not, use a 2-d array to give a weight to
  each combination of a time period and an entity.

For example, the weights of a panel that is not balanced, whose rows are in any order,
can be given as a column of the data that has a value for each row

.. code-block:: python

    weights = data["weight"].to_numpy()  # a value for each row, in the order of the rows
    mod = PanelOLS(data["y"], data[["x"]], weights=weights, entity_effects=True)

.. versionchanged:: 7.1

   A 1-d array with a weight for each observation can be used if ``dependent`` is not
   balanced. Earlier versions raised "Weights do not have a supported shape" for it,
   although the same weights as a Series with the index of ``dependent`` worked, and
   an array with ``nobs * nentity`` values raised an error from pandas or NumPy. If
   ``dependent`` is not balanced and has as many rows as time periods or entities,
   an array with that many values is a weight for each observation. Earlier versions
   read it as a weight for each time period or entity, or raised an
   ``AmbiguityError`` if the numbers of time periods and entities were equal.

Weights that are defined for each time period, for each entity or for each
combination of the two are matched to the time periods and entities in the
order that they first appear in ``dependent``. This is the case even if the
rows of ``dependent`` are not sorted by entity and then time. When these
weights are a pandas object, the index (and columns) must contain the time
periods and entities in this order.

``clusters`` and ``other_effects`` can be given for each combination of time
period and entity in the same way as weights: a 2-d array or a DataFrame with
``nobs`` rows and ``nentity`` columns, or a 3-d array or a ``DataArray`` with the
variables first, which two-way clusters and two effects need. They are placed on
the rows of ``dependent`` by time period and entity, whatever the order of the
rows is. ``dependent`` does not need to be balanced for any of these inputs. The
values of the combinations of a time period and an entity that ``dependent``
does not have are not used, and can be missing. For example, the industry of each
observation of a panel that is sorted by entity and time can be used for the
clusters of the standard errors in either of the two ways

.. code-block:: python

    clusters = data["industry"].unstack(0)  # time periods by entities
    res = mod.fit(cov_type="clustered", clusters=clusters)
    res = mod.fit(cov_type="clustered", clusters=data[["industry"]])

In an unbalanced panel the order in which the time periods and entities first
appear can differ from the order of the time periods and entities. If the first
entity enters late and the rows are sorted by entity, for example, the first
periods appear last. The time periods are in order of time if the rows of every
entity are in order of time, in either direction, and the entities are in the
order of their labels if the rows of every time period are in order of the
entities. This is how a panel that is sorted by entity and time, or by time and
entity, is matched whether or not it is balanced, and a DataFrame that
``unstack`` makes from such a panel has its time periods and entities in order.
If the rows are in another order, then the time periods and entities are in the
order that they first appear, as in a balanced panel. A pandas object that does not
use this order raises a ``ValueError``. Values for each row, a pandas object with
the index of ``dependent`` or a 1-d array, do not depend on this order.

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

An :math:`R^2` is NaN, with no error or warning, if the total sum of squares that it
divides by is zero, up to rounding error, since it is not defined then. This
happens if the dependent variable does not vary, or if it has no variation left
once the means or the effects of the measure are removed, e.g., the within
:math:`R^2` of a dependent variable that is constant within each entity, or of a
panel with a single time period. See
:attr:`~linearmodels.panel.results.PanelResults.rsquared_within`.
