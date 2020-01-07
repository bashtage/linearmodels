.. _iv-module-reference:

================
Module Reference
================

.. _iv-module-reference-models:

Instrumental Variable Estimation
--------------------------------

.. module:: linearmodels.iv.model
   :synopsis: Instrumental variable estimation

.. currentmodule:: linearmodels.iv.model

.. autosummary::
   :toctree: iv/

   IV2SLS
   IVLIML
   IVGMM
   IVGMMCUE

Absorbing Least Squares
-----------------------
OLS and WLS with high-dimensional effects.

.. _absorbing-module-reference-models:

.. currentmodule:: linearmodels.iv.absorbing

.. module:: linearmodels.iv.absorbing
   :synopsis: Regression with high-dimensional effects


.. autosummary::
   :toctree: absorbing/

   AbsorbingLS
   AbsorbingLSResults
   Interaction


.. _iv-module-reference-results:

Estimation Results
------------------

.. module:: linearmodels.iv.results
   :synopsis: Estimation results for instrumental variable estimators

.. currentmodule:: linearmodels.iv.results

.. autosummary::
   :toctree: iv/

   IVResults
   IVGMMResults
   OLSResults
   IVModelComparison
   FirstStageResults
   compare

.. _iv-module-reference-covariance:

Instrumental Variable Covariance Estimation
-------------------------------------------

.. module:: linearmodels.iv.covariance
   :synopsis: Covariance estimators for linear IV models

.. currentmodule:: linearmodels.iv.covariance

.. autosummary::
   :toctree: iv/

   HomoskedasticCovariance
   HeteroskedasticCovariance
   ClusteredCovariance
   KernelCovariance


Kernel Weight Generators
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: iv/
   
   kernel_weight_bartlett
   kernel_weight_parzen
   kernel_weight_quadratic_spectral


GMM Weight and Covariance Estimation
------------------------------------

.. module:: linearmodels.iv.gmm
   :synopsis: GMM estimation of linear IV models

.. currentmodule:: linearmodels.iv.gmm

.. autosummary::
   :toctree: iv/

   IVGMMCovariance
   HomoskedasticWeightMatrix
   HeteroskedasticWeightMatrix
   KernelWeightMatrix
   OneWayClusteredWeightMatrix


IV Data Structures
------------------

.. module:: linearmodels.iv.data
   :synopsis: Data structured used in instrumental variables estimators

.. currentmodule:: linearmodels.iv.data

.. autosummary::
   :toctree: iv/

   IVData
