from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl
import plotly
import plotly.graph_objs as go
import tempfile


def test():
    x1 = [10, 3, 4, 5, 20, 4, 3]

    trace1 = go.Box(x = x1)

    layout = go.Layout(
        showlegend = True
    )


    data = [trace1]
    fig = go.Figure(data=data, layout = layout)

    # fn = '/Users/Chao/plot.html'
    fn = tempfile.NamedTemporaryFile(suffix='.html').name
    plotly.offline.plot(fig, output_type = 'file', filename=fn, auto_open = False, include_plotlyjs=True)
    app = QApplication([])
    view = QWebEngineView()
    view.load(QUrl.fromLocalFile(fn))

    # view.setHtml(raw_html)
    view.show()

    app.exec()