.. _asset_pricing-module-reference:

================
Module Reference
================

.. _asset_pricing-module-reference-models:

Linear Asset Pricing Models
---------------------------

.. module:: linearmodels.asset_pricing.model
   :synopsis: Models for asset prices

.. currentmodule:: linearmodels.asset_pricing.model

.. autosummary::
   :toctree: asset-pricing/

   LinearFactorModel
   LinearFactorModelGMM
   TradedFactorModel


.. _asset_pricing-module-reference-results:

Estimation Results
------------------

.. module:: linearmodels.asset_pricing.results
   :synopsis: Results for asset pricing models

.. currentmodule:: linearmodels.asset_pricing.results

.. autosummary::
   :toctree: asset-pricing/

   LinearFactorModelResults

.. _asset_pricing-module-reference-covariance:

Linear Asset Pricing Model Covariance Estimators
------------------------------------------------

.. module:: linearmodels.asset_pricing.covariance
   :synopsis: Covariance estimators for asset pricing models

.. currentmodule:: linearmodels.asset_pricing.covariance

.. autosummary::
   :toctree: asset-pricing/

   HeteroskedasticCovariance
   KernelCovariance
