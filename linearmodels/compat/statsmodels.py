"""
Summary class built on the summary of statsmodels.
"""

from statsmodels.iolib import summary

_LATEX_ESCAPES = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


class Summary(summary.Summary):
    """
    Summary of the estimation results of a model.

    The summary contains tables of results and notes below the tables. It is
    returned by the ``summary`` property of the results. It can be converted
    to text, HTML, LaTeX and CSV using ``as_text``, ``as_html``, ``as_latex``
    and ``as_csv``.

    Attributes
    ----------
    tables : list
        The tables of the summary.
    extra_txt : str or None
        The notes that are written below the tables in every format, with one
        line of the notes on each line of the string. Set it to None to remove
        the notes from ``as_text``, ``as_html``, ``as_latex`` and ``as_csv``, or
        to a string to replace them. ``add_extra_txt`` replaces the notes with
        the lines of text that it is given.

    Notes
    -----
    The ``summary`` property of the results creates a new summary each time that
    it is used. Keep the summary in a variable before changing ``extra_txt``.

    Examples
    --------
    Export the tables without the notes

    >>> smry = res.summary
    >>> smry.extra_txt = None
    >>> latex = smry.as_latex()
    """

    def as_latex(self) -> str:
        """
        Return the tables as a string in LaTeX format.

        Returns
        -------
        str
            The concatenated summary tables in LaTeX format. Special LaTeX
            characters in the extra text, such as the underscores in
            variable names, are escaped.
        """
        latex = summary.summary_return(self.tables, return_fmt="latex")
        if self.extra_txt is not None:
            extra_txt = self.extra_txt.translate(_LATEX_ESCAPES)
            latex = latex + "\n\n" + extra_txt.replace("\n", " \\newline\n ")
        return latex

    def as_html(self) -> str:
        """
        Return the tables as a string in HTML format.

        Returns
        -------
        str
            The concatenated summary tables in HTML format.
        """
        html = summary.summary_return(self.tables, return_fmt="html")
        if self.extra_txt is not None:
            html = html + "<br/><br/>" + self.extra_txt.replace("\n", "<br/>")
        return html
