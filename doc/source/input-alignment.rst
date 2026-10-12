.. _input-alignment:
.. _iv-input-alignment:
.. _system-input-alignment:
.. _asset-pricing-input-alignment:

Aligning Inputs
===============
The rows of the inputs of a model are paired by position: the first row of
``exog`` is attached to the first row of ``dependent``, and so on. To prevent a
value from being attached to the wrong observation, an input that carries
labels must use the same labels, in the same order, as ``dependent``. A
``ValueError`` that names the input is raised if the labels differ, for example
when the rows of a regressor have been sorted differently from the rows of the
dependent variable. The inputs that carry labels are

* a pandas ``Series`` or ``DataFrame`` that has an index other than the default
  index, 0, 1, ..., n-1;
* an xarray ``DataArray`` that has coordinates for its rows;
* :class:`~linearmodels.iv.data.IVData` built from either.

To correct the error, reindex the input to the index of ``dependent``, e.g.,
``exog = exog.reindex(dependent.index)``, or call ``to_numpy()`` to explicitly
pair the values by position.

NumPy arrays, lists, tuples and pandas objects that only have the default index
do not carry labels. They are always paired by position, and only need to have
the right number of rows. This is also the case for a ``DataFrame`` with a
default index and named columns, since the columns are variable names and are
not compared. Index names are ignored.

If ``dependent`` does not carry labels, the first of the other inputs that
carries labels is the reference instead, and each of the inputs after it that
carries labels must use the same labels, in the same order. The error names both
inputs, for example "The row index of instruments does not match exog." If none
of the inputs carries labels, all of them are paired by position.

The order of the inputs, which decides which one is the reference when
``dependent`` does not carry labels, and the inputs that are compared are

* **IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE and OLS**: ``exog``, ``endog``,
  ``instruments`` and ``weights``. ``clusters`` passed to ``fit``, or to the
  weight estimator of IVGMM and IVGMMCUE when ``weight_type`` is
  ``"clustered"``, is compared in the same way.
* **AbsorbingLS**: ``exog``, ``absorb``, ``interactions`` and ``weights``, and
  ``clusters`` passed to ``fit``. If both ``cat`` and ``cont`` of an
  ``Interaction`` carry labels, they must use the same labels.
* **SUR, IV3SLS and IVSystemGMM**: the dependent variables that carry labels
  must use the same labels in every equation, since the residuals in a row of
  all of the equations are the errors of one observation. In each equation,
  ``exog``, ``endog``, ``instruments`` and ``weights`` are compared with the
  ``dependent`` of the equation. If the ``dependent`` of an equation does not
  carry labels, they are compared with the labels of the system. These are the
  labels of the first dependent variable that carries labels or, if none does,
  of the first input that carries labels, taking the equations in order and
  ``exog``, ``endog``, ``instruments`` and ``weights`` within an equation. All of
  the equations must have the same number of observations. ``clusters`` passed
  to ``fit`` of SUR and IV3SLS is compared with the labels of the system.
* **TradedFactorModel, LinearFactorModel and LinearFactorModelGMM**: ``factors``
  is compared with ``portfolios``. Nothing is compared if ``portfolios`` does not
  carry labels, since ``factors`` is the only other input with rows.
* **The panel models** follow the same rule, see :ref:`panel-input-alignment`.

The labels are compared before observations that contain missing values are
dropped. The observations that remain are the ones used in estimation, and so
``clusters`` has to be provided for these observations. For the IV models,
``model.notnull`` is a Boolean array that marks them, e.g.,
``clusters[model.notnull]``. For the other models, drop the same rows from
``clusters`` that were dropped from the data. ``clusters`` is compared with the
labels of the remaining observations. ``clusters`` with the length of the
original data raises the error about the wrong number of observations.

The formula interfaces take all of the variables from one ``DataFrame``, so they
use the same labels. ``weights`` (and ``portfolios`` in the asset pricing
models) that are passed separately to ``from_formula`` are compared in the same
way.

Lists and tuples are accepted wherever a parameter is documented as
``array_like``, for example ``dependent``, ``exog``, ``endog``, ``instruments``,
``weights`` and ``clusters``. They are converted to arrays when the model is
created, and give the same results as arrays. They do not carry labels.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from linearmodels.iv import IV2SLS

   rng = np.random.default_rng(0)
   index = pd.Index([f"obs{i}" for i in range(100)])
   z = pd.Series(rng.standard_normal(100), index=index, name="z")
   x = pd.Series(z + rng.standard_normal(100), index=index, name="x")
   y = pd.Series(x + rng.standard_normal(100), index=index, name="y")
   const = pd.DataFrame({"const": np.ones(100)}, index=index)

   IV2SLS(y, const, x, z)  # same labels, same order
   try:
       IV2SLS(y, const, x, z.iloc[::-1])  # same labels, different order
   except ValueError as exc:
       print(exc)
   IV2SLS(y, const, x, z.reindex(y.index))  # reindexed to the labels of y
   IV2SLS(y, const, x, z.to_numpy())  # paired by position
   try:
       # dependent has no labels, so exog is the reference
       IV2SLS(y.to_numpy(), const, x, z.iloc[::-1])
   except ValueError as exc:
       print(exc)
   IV2SLS(y.to_list(), const.to_numpy(), x.to_list(), z.to_list())  # lists
