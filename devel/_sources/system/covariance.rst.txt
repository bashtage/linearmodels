.. _system_models-module-reference-covariance:

System Regression Estimator Covariance Estimation
-------------------------------------------------

.. module:: linearmodels.system.covariance
   :synopsis: Covariance estimators for system regression models

SUR and 3SLS
============

Homoskedastic Covariance Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HomoskedasticCovariance
   :members:
   :inherited-members:

Heteroskedasticity Robust Covariance Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HeteroskedasticCovariance
   :members:
   :inherited-members:

Heteroskedasticity-Autocorrelation (HAC) Robust Covariance Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: KernelCovariance
   :members:
   :inherited-members:

Generalized Method of Moments (GMM)
===================================

GMM Homoskedastic Covariance Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GMMHomoskedasticCovariance
   :members:
   :inherited-members:

GMM Heteroskedasticity Robust Covariance Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GMMHeteroskedasticCovariance
   :members:
   :inherited-members:

GMM Heteroskedasticity-Autocorrelation (HAC) Robust Covariance Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GMMKernelCovariance
   :members:
   :inherited-members:
