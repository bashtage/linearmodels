from __future__ import annotations

from statsmodels.iolib import summary, summary2


class Summary(summary.Summary):
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

class Summary2(summary2.Summary):
    def as_html(self) -> str:
        """
        Return tables as string

        Returns
        -------
        str
            concatenated summary tables in HTML format
        """
        html = summary2.summary_return(self.tables, return_fmt="html")
        if self.extra_txt is not None:
            html = html + "<br/><br/>" + self.extra_txt.replace("\n", "<br/>")
        return html

