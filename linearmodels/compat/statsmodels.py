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
    def as_latex(self) -> str:
        """
        Return tables as string

        Returns
        -------
        str
            concatenated summary tables in LaTeX format. Special LaTeX
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
        Return tables as string

        Returns
        -------
        str
            concatenated summary tables in HTML format
        """
        html = summary.summary_return(self.tables, return_fmt="html")
        if self.extra_txt is not None:
            html = html + "<br/><br/>" + self.extra_txt.replace("\n", "<br/>")
        return html
