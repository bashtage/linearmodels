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


Errors
------

.. currentmodule:: linearmodels.shared.exceptions

The name of a covariance estimator (``cov_type``), of a weight matrix estimator
(``weight_type``) or of a kernel (``kernel``) that is not supported raises an
:class:`UnknownOptionError`, which lists the supported names. It is a ``ValueError``
and a ``KeyError``.

.. autoclass:: linearmodels.shared.exceptions.UnknownOptionError
   :show-inheritance:


Covariance Estimation
---------------------

.. currentmodule:: linearmodels.shared.covariance


.. autosummary::
   :toctree: utility/

   cov_cluster
   cov_kernel
   group_debias_coefficient
