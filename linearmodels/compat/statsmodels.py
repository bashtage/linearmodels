from statsmodels.iolib import summary


class Summary(summary.Summary):
    def as_html(self):
        '''return tables as string
        Returns
        -------
        html : string
            concatenated summary tables in HTML format
        '''
        html = summary.summary_return(self.tables, return_fmt='html')
        if self.extra_txt is not None:
            html = html + '<br/><br/>' + self.extra_txt.replace('\n', '<br/>')
        return html
