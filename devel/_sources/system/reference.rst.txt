.. _system_models-module-reference:

================
Module Reference
================

.. _system_models-module-reference-models:

System Regression Estimators
----------------------------

.. module:: linearmodels.system.model
   :synopsis: System regression estimators

.. currentmodule:: linearmodels.system.model

.. autosummary::
   :toctree: system/

   SUR
   IV3SLS
   IVSystemGMM

.. _system_models-module-reference-results:

Estimation Results
------------------

.. module:: linearmodels.system.results
   :synopsis: Results for system regression estimators

.. currentmodule:: linearmodels.system.results

.. autosummary::
   :toctree: system/

   SystemResults
   GMMSystemResults

.. _system_models-module-reference-covariance:

System Regression Estimator Covariance Estimation
-------------------------------------------------

.. module:: linearmodels.system.covariance
   :synopsis: Covariance estimators for system regression models

.. currentmodule:: linearmodels.system.covariance

SUR and 3SLS
~~~~~~~~~~~~

.. autosummary::
   :toctree: system/

   HomoskedasticCovariance
   HeteroskedasticCovariance
   KernelCovariance
   ClusteredCovariance

Generalized Method of Moments (GMM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: system/

   GMMHomoskedasticCovariance
   GMMHeteroskedasticCovariance
   GMMKernelCovariance

GMM Moment Weighting
--------------------

.. module:: linearmodels.system.gmm
   :synopsis: Moment weight estimators for GMM IV estimation

.. currentmodule:: linearmodels.system.gmm

.. autosummary::
   :toctree: system/

   HomoskedasticWeightMatrix
   HeteroskedasticWeightMatrix
   KernelWeightMatrix

