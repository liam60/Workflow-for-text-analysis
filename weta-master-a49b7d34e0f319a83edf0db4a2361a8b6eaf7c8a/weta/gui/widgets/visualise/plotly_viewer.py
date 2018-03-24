import pyspark
import pyspark.sql
from PyQt5 import QtCore
from Orange.widgets import widget, gui
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import plotly
import plotly.graph_objs as go
import tempfile

from weta.gui.spark_environment import SparkEnvironment


class OWPlotlyViewer(SparkEnvironment, widget.OWWidget):
    # --------------- Widget metadata protocol ---------------
    priority = 2

    name = "Plotly"
    description = "A plotly canvas"
    icon = "../assets/DataFrameViewer.svg"

    input_figure: go.Figure = None
    class Inputs:
        figure = widget.Input("Figure", go.Figure)

    class Outputs:
        pass
        #data_frame = widget.Output("DataFrame", pyspark.sql.DataFrame)

    want_control_area = True

    def __init__(self):
        super().__init__()
        self.controlArea.setMinimumWidth(250)
        self.v_info_box = gui.vBox(self.controlArea, 'Info')
        self.v_info = gui.label(self.v_info_box, self, '')
        self.v_info.setAlignment(QtCore.Qt.AlignTop)

        self.mainArea.setMinimumWidth(800)
        self.mainArea.setMinimumHeight(600)

        self.v_webview = QWebEngineView(self.mainArea)
        self.v_webview.setMinimumWidth(800)
        self.v_webview.setMinimumHeight(600)


    @Inputs.figure
    def set_input_figure(self, figure):
        self.input_figure = figure

    # called after received all inputs
    def handleNewSignals(self):
        self.apply()

    def _check_input(self):
        if self.input_figure is None:
            self.warning('Input figure does not exist')
            return False
        else:
            return True

    # this is the logic: computation, update UI, send outputs. ..
    def apply(self):
        if not self._check_input():
            return

        self.clear_messages()

        filename = tempfile.NamedTemporaryFile(suffix='.html').name
        print('Plot file path: %s' % filename)
        plotly.offline.plot(self.input_figure, output_type='file', filename=filename,
                            auto_open=False, show_link=False)  # , include_plotlyjs=True)

        self.v_webview.load(QUrl.fromLocalFile(filename))