Contributing
------------

Contributions are welcome.  There are many areas that can be improved that do
not require extensive experience in programming Python:

* Documentation
* Examples
* Docstrings
* Typos in docstrings, examples or documentation

For more experienced programmers, there are some clear opportunities to
refactor the code to share more between the difference classes of models.
These types of simplifications are welcome.

Changes that simplify releasing, documentation or other CI tasks are also
welcome.

Of course, entire models the expand the package to cover other important
linear models are always welcome.

**Github**

Fork the repo, make a branch with your new feature/documentation/typos
and submit a pull request.

.. _docstring-conventions:

Docstring conventions
~~~~~~~~~~~~~~~~~~~~~

Docstrings use the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_
format. Every module, class, function, method and property needs one, unless its name
starts with an underscore. Apply these nine rules to every docstring that you write or
change.

1. Put ``, optional`` after the type of every parameter that has a default. The default
   value goes in the description, never in the type line.
2. Write a union of types as ``int or str``. Use curly braces only for a set of literal
   values, written with plain quotes: ``{"robust", "unadjusted"}``. List every value that
   the code accepts, aliases included.
3. Do not subscript types. Write ``list of str`` and ``dict of str to DataFrame``, not
   ``list[str]`` and ``dict[str, DataFrame]``.
4. ``array_like`` means that the value is converted with ``np.asarray`` and then used as
   an array. ``ndarray`` means that it must already be an array. ``sequence of X`` means
   that it is only iterated over or indexed.
5. Use ``bool``, ``int``, ``str`` and ``callable``, not ``boolean``, ``integer``,
   ``string`` or ``function``. Document ``**cov_config`` and ``**weight_config`` with
   their stars, and list their keys.
6. Use the standard section names only, with underlines exactly as long as the header.
   Write ``See Also``, not ``See also``. In order, the sections are ``Parameters``,
   ``Attributes``, ``Methods``, ``Returns``, ``Yields``, ``Other Parameters``,
   ``Raises``, ``Warns``, ``Warnings``, ``See Also``, ``Notes``, ``References`` and
   ``Examples``.
7. A ``Returns`` section gives concrete types, such as ``Series`` or ``IVResults``. It
   names what the function returns, not what it accepts.
8. Every documented parameter exists, and every parameter is documented, in the order of
   the signature.
9. A real bug that you find while editing a docstring goes in its own ``BUG`` commit.

For example:

.. code-block:: text

    Parameters
    ----------
    cats : DataFrame or array_like
        Category codes of pandas categoricals, with shape (nobs, ncats).
    output_format : {"csc", "csr", "coo"}, optional
        Format of the sparse output. The default is ``"csc"``.
    drop : {"first", "last"}, optional
        Whether to exclude the first or the last category. The default is ``"first"``.

Checking docstrings
^^^^^^^^^^^^^^^^^^^

``numpydoc lint`` checks docstrings. It needs numpydoc 1.10 or later. Run it from the
root of the repository. It prints one line for each violation, and exits with status 1 if
there is any violation:

.. code-block:: bash

    python -m pip install "numpydoc>=1.10"
    python -m numpydoc lint $(git ls-files "linearmodels/*.py") --config .

In PowerShell, write ``(git ls-files "linearmodels/*.py")`` for the file list. To check
only some files, pass their paths instead. On Windows, also run Python in UTF-8 mode, for
example with ``python -X utf8``, because numpydoc reads the files with the system
encoding and some docstrings contain non-ASCII characters.

The ``[tool.numpydoc_validation]`` table in ``pyproject.toml`` sets what is checked.

* Every numpydoc check is enforced except ``ES01`` (no extended summary), ``SA01`` (no
  ``See Also`` section) and ``EX01`` (no ``Examples`` section). The enforced checks
  include the punctuation of summaries and of parameter and return descriptions
  (``SS03``, ``PR08``, ``PR09``, ``RT04`` and ``RT05``), the description of every
  ``See Also`` entry (``SA04``), parameters that match the signature (``PR01``, ``PR02``
  and ``PR03``), and the names and order of the sections (``GL06`` and ``GL07``).
  Numpydoc does not check rules 1 to 4 and 9, the keys listed under rule 5 or the types
  named under rule 7. Check them in review.
* Module docstrings are required (``GL08`` applies to modules), including the docstrings
  of ``__init__.py`` files. One line is enough.
* The linter skips tests, ``conftest.py``, ``_version.py``, special methods such as
  ``__init__`` (document the constructor in the class docstring), functions defined
  inside other functions, and every name with a leading underscore. The exceptions are
  the public members of private base classes, such as ``_IVModelBase.predict``, which the
  reference pages show on the public subclasses, and ``_OLS``, ``_Panel`` and
  ``LinearConstraint``, which the reference pages list.
* The linter cannot see decorators, so it asks for a ``Returns`` section and an infinitive
  verb in the summary of every property. The configuration ignores both checks for a
  docstring without a ``Parameters`` section, so a property needs only a summary. A
  method without parameters that returns a value, such as ``PanelData.copy``, still needs
  a ``Returns`` section, but the linter does not check it.
  ``python -m numpydoc validate linearmodels.panel.data.PanelData.copy`` does; ignore
  ``ES01``, ``SA01`` and ``EX01`` in its output.
* Some declarations cannot satisfy the linter. A property setter and an ``@overload``
  stub cannot have a docstring that is shown in the documentation: write
  ``# numpydoc ignore=GL08`` on its ``def`` line. A ``NamedTuple`` has no ``__init__``, so
  the linter reports its fields as unknown parameters: document them under ``Parameters``
  and write ``# numpydoc ignore=PR02`` on its ``class`` line.
