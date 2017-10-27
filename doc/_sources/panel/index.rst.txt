Panel Data Model Estimation
---------------------------
The :ref:`panel-introduction` provides a brief overview of the available panel
model estimators. :ref:`panel-module-reference-models` contains a complete
reference to available estimation methods.
:ref:`panel-module-reference-results` describes the results classes returned
after estimating a model as well as model specification tests. The covariance
estimators are presented in :ref:`panel-module-reference-covariance`.

:ref:`Basic Examples <panel/examples/examples.ipynb>` shows basic usage
of the models using examples from Wooldridge's text books.
:ref:`Using Formulas <panel/examples/using-formulas.ipynb>` shows how models
can be specified using R-like formulas using patsy.
:ref:`Data Formats <panel/examples/data-formats.ipynb>` described the
alternative formats that can be used when specifying models.


:ref:`panel-mathematical-notation` contains a concise explanation of the formulas
used in estimating parameters, estimating covariances and conducting
hypothesis tests.

:ref:`panel-implementation-choices` describes some differences between the
estimators in this package and other packages commonly used to estimate panel
data models. Finally, :ref:`panel-pandas-differences` describes important differences
to the now deprecated `PanelOLS` that was in the `pandas.stats.plm` modules
until release 0.20.

.. toctree::
   :maxdepth: 1
   :glob:

   introduction
   examples/data-formats.ipynb
   examples/examples.ipynb
   examples/using-formulas.ipynb
   pandas
   reference
   mathematical-formula
   faq