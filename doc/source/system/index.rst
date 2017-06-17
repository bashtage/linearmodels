System Regression Models
------------------------

System regression estimates multiple regressions simulteneously.  There are
three reasons to consider system estimation in stread of equation by
equation estimation

* Joint inference on parameters across models
* Efficiency gains in some circumstances using cross-model GLS
* Imposing restrictions on parameters across models

The main model is the Seemingly Unrelated Regression (:class:`~linearmodels.system.model.SUR`)
Estimator.  This estimator uses a modified syntax since the class allow
multiple models to be specified, each with it own dependent and exogenous
variablces. The preferred syntax uses an (:class:`dict`) or preferrable
an (:class:`~collections.OrderedDict`) where each entry is a complete model
with a dependent and exogenous (stored as a :class:`dict` with keys
``dependent`` and ``exog``.

.. code-block:: python

  from collections import OrderedDict
  import statsmodels.api as sm
  from linearmodels.datasets import fringe
  from linearmodels.system import SUR
  data = sm.add_constant(fringe.load())
  equations = OrderedDict()
  equations['earnings'] = {'dependent': data.hrearn,
                           'exog': data[['const', 'exper', 'tenure']]}
  equations['benefits'] = {'dependent': data.hrbens,
                           'exog': data[['const', 'exper', 'tenure']]}


::

                           System GLS Estimation Summary
   ==============================================================================
   Estimator:                        GLS   Overall R-squared:              0.0757
   No. Equations.:                     2   Cov. Estimator:             unadjusted
   No. Observations:                 616   Num. Constraints:                 None
   Date:                Sat, Jun 17 2017
   Time:                        23:21:18


                   Equation: earnings, Dependent Variable: hrearn
   ==============================================================================
               Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
   ------------------------------------------------------------------------------
   const          4.2839     0.3407     12.573     0.0000      3.6161      4.9517
   exper          0.1163     0.0186     6.2478     0.0000      0.0798      0.1528
   tenure        -0.0283     0.0295    -0.9598     0.3372     -0.0862      0.0295

                   Equation: benefits, Dependent Variable: hrbens
   ==============================================================================
               Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
   ------------------------------------------------------------------------------
   const          0.6390     0.0449     14.220     0.0000      0.5509      0.7270
   exper          0.0014     0.0025     0.5617     0.5743     -0.0034      0.0062
   tenure         0.0316     0.0039     8.1254     0.0000      0.0240      0.0393

   ==============================================================================
   SURResults, id: 0x282ca8f7b70


.. toctree::
   :maxdepth: 1
   :glob:

   examples/examples.ipynb
   examples/formulas.ipynb
   reference
   mathematical-formula
