from functools import partial

import pyspark
import pyspark.sql
from PyQt5 import QtCore
from Orange.widgets import widget, gui
from pyspark.ml.linalg import Vector
from weta.gui.async_task import AsyncTask

from weta.gui.spark_environment import SparkEnvironment


class OWDataFrameViewer(SparkEnvironment, AsyncTask, widget.OWWidget):
    # --------------- Widget metadata protocol ---------------
    priority = 2

    name = "Data Viewer"
    description = "View Spark Data frame"
    icon = "../assets/DataFrameViewer.svg"

    # --------------- Input/Output signals ---------------
    DataFrame: pyspark.sql.DataFrame = None
    class Inputs:
        DataFrame = widget.Input("DataFrame", pyspark.sql.DataFrame)

    class Outputs:
        DataFrame = widget.Output("DataFrame", pyspark.sql.DataFrame)

    # --------------- UI layout settings ---------------
    want_control_area = True

    # --------------- Settings ---------------

    def __init__(self):
        super().__init__()
        self.controlArea.setMinimumWidth(250)
        self.v_info_box = gui.vBox(self.controlArea, 'Info')
        self.v_info = gui.label(self.v_info_box, self, '')
        self.v_info.setAlignment(QtCore.Qt.AlignTop)

        self.mainArea.setMinimumWidth(600)
        self.mainArea.setMinimumHeight(600)

        self.v_table = gui.table(self.mainArea, 0, 0)

    @Inputs.DataFrame
    def set_data_frame(self, df):
        self.DataFrame = df

    # called after received all inputs
    def handleNewSignals(self):
        self.apply()

    def _validate_input(self):
        if self.DataFrame is None:
            self.warning('Input data does not exist')
            return False
        else:
            return True

    # this is the logic: computation, update UI, send outputs. ..
    def apply(self):
        if not self._validate_input():
            return

        self.clear_messages()

        # show data
        columns = self.DataFrame.columns

        self.v_info.setText(self.get_info())

        self.v_table.setRowCount(0)
        self.v_table.setColumnCount(len(columns))
        self.v_table.setHorizontalHeaderLabels(columns)

        self.async_call(partial(collect, self.DataFrame, 100))

        self.Outputs.DataFrame.send(self.DataFrame)

    def on_finish(self, results):
        for i, row in enumerate(results):  # show top 100 rows
            self.v_table.insertRow(i)
            for j, column in enumerate(self.DataFrame.columns):
                value = row[column]
                if isinstance(value, Vector):
                    value = str(list(value.toArray()[:10])) + '...' # to dense array
                else:
                    value = str(value)
                gui.tableItem(self.v_table, i, j, value)


    def get_info(self):
        df = self.DataFrame
        columns = df.columns
        return '''
Rows: %d 
Columns: %d
Types: 
    %s
       ''' % (df.count(), len(columns), '\n    '.join([t[0] + ' - ' + t[1] for t in df.dtypes]))

def collect(df, N):
    return df.head(n=N)