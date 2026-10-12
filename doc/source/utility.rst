Utilities
=========

Containers
----------

.. module:: linearmodels.utility
   :synopsis: Utilities used across all model types

.. currentmodule:: linearmodels.shared.utility


.. autosummary::
   :toctree: utility/

   AttrDict


Hypothesis Testing
------------------

.. currentmodule:: linearmodels.shared.hypotheses


.. autosummary::
   :toctree: utility/

   WaldTestStatistic
   NormalTestStatistic
   InvalidTestStatistic
   InapplicableTestStatistic


Covariance Estimation
---------------------

.. currentmodule:: linearmodels.shared.covariance


.. autosummary::
   :toctree: utility/

   cov_cluster
   cov_kernel
   group_debias_coefficient


Missing Values
--------------

.. currentmodule:: linearmodels.shared.exceptions

A model drops the observations that have a missing value, such as ``NaN``, in any of
its inputs when it is created. It issues a :class:`MissingValueWarning` if it drops any
observation, and the results use the remaining observations. This is the same in every
model.

The warning is on by default. It can be turned off in three ways.

* Set ``linearmodels.WARN_ON_MISSING`` to ``False``. It applies to every model that is
  created afterwards.

  .. code-block:: python

     import linearmodels

     linearmodels.WARN_ON_MISSING = False

* Set the environment variable ``LINEARMODELS_WARN_ON_MISSING`` before importing
  linearmodels, for example in the shell that starts Python. The values ``""`` (the
  empty string), ``"0"``, ``"false"`` and ``"False"`` turn the warning off. Any other
  value, or no variable, leaves it on. The variable is read when linearmodels is
  imported, so setting it later has no effect.

  .. code-block:: bash

     export LINEARMODELS_WARN_ON_MISSING=0

* Use a warnings filter. A filter applies only where it is set, so it can turn off the
  warning for part of a program. The warning is issued when the model is created, not
  when it is fit.

  .. code-block:: python

     import warnings

     from linearmodels.iv import IV2SLS
     from linearmodels.shared.exceptions import MissingValueWarning

     with warnings.catch_warnings():
         warnings.simplefilter("ignore", MissingValueWarning)
         mod = IV2SLS(dependent, exog, endog, instruments)
     res = mod.fit()

  The action ``"error"`` instead of ``"ignore"`` raises an exception if a model drops
  any observation.

.. deprecated:: 7.1

   The module attribute ``linearmodels.DROP_MISSING`` and the environment variable
   ``LINEARMODELS_DROP_MISSING`` have never had an effect, because observations with
   missing values are always dropped. Reading ``linearmodels.DROP_MISSING``, and
   importing linearmodels when the environment variable is set, issue a
   ``FutureWarning``. Both will be removed in linearmodels 8.0. Use
   ``WARN_ON_MISSING`` to turn off the warning.

.. autoclass:: linearmodels.shared.exceptions.MissingValueWarning
