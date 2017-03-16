Linear Models
=============

|Build Status|

.. code:: diff

    + The instrumental variable code is complete and tested
    - This package is still under heavy development and should not be considered stable

Plan and status
~~~~~~~~~~~~~~~

Should eventually add some useful linear model estimators such as
instrumental variable estimators and panel regression. Currently very
unpolished.

-  Linear Instrumental variable estimation - *complete*
-  Linear Panel model estimation - *incomplete*
-  Linear IV Panel model estimation - *not started*
-  System regression - *not started*

Requirements
~~~~~~~~~~~~

-  Python 3.5+: extensive use of ``@`` operator
-  NumPy
-  SciPy
-  Pandas
-  xarray (optional)
-  Statsmodels

Testing
~~~~~~~

-  py.test
-  xarray

Documentation
~~~~~~~~~~~~~

-  sphinx
-  sphinx\_rtd\_theme
-  nbsphinx
-  nbconvert
-  nbformat
-  ipython
-  jupyter
-  matplotlib

.. |Build Status| image:: https://travis-ci.org/bashtage/linearmodels.svg?branch=master
   :target: https://travis-ci.org/bashtage/linearmodels
