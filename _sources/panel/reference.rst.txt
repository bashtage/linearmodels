.. _panel-module-reference:

================
Module Reference
================

.. _panel-module-reference-models:

Models for Panel Data
---------------------

.. module:: linearmodels.panel.model
   :synopsis: Models for panel data

.. currentmodule:: linearmodels.panel.model

.. autosummary::
   :toctree: panel/

   PanelOLS
   RandomEffects
   BetweenOLS
   FirstDifferenceOLS
   PooledOLS
   FamaMacBeth

.. _panel-module-reference-results:

Estimation Results
------------------

.. module:: linearmodels.panel.results
   :synopsis: Estimation results for panel data models

.. currentmodule:: linearmodels.panel.results

.. autosummary::
   :toctree: panel/

   FamaMacBethResults
   PanelResults
   PanelEffectsResults
   RandomEffectsResults
   PanelModelComparison

.. _panel-module-reference-covariance:

Panel Model Covariance Estimators
---------------------------------

.. module:: linearmodels.panel.covariance
   :synopsis: Covariance estimators for panel data models

.. currentmodule:: linearmodels.panel.covariance

.. autosummary::
   :toctree: panel/

   HomoskedasticCovariance
   HeteroskedasticCovariance
   ClusteredCovariance
   DriscollKraay
   ACCovariance
   FamaMacBethCovariance

.. _panel-module-reference-data:

Panel Data Structures
---------------------

.. module:: linearmodels.panel.data
   :synopsis: Data structured used in panel data models

.. autosummary::
   :toctree: panel/

   PanelData

Helper Functions
----------------
.. module:: linearmodels.panel.utility
   :synopsis: Utilities for working with panel data

.. autosummary::
   :toctree: panel/

   generate_panel_data
