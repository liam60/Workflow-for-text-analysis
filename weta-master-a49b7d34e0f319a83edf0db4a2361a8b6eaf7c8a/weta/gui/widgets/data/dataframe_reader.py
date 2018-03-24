from collections import OrderedDict
from pyspark.sql.functions import monotonically_increasing_id

import pyspark.sql
from PyQt5 import QtWidgets, QtCore
from Orange.widgets import widget, gui, settings

from weta.gui.spark_environment import SparkEnvironment


class Parameter:
    def __init__(self, name, default_value='', type='str', widget_type='text_edit', data=None):
        self.name = name
        self.default_value = default_value
        self.type = type
        self.widget_type = widget_type
        self.data = data


class OWDataFrameReader(SparkEnvironment, widget.OWWidget):
    priority = 1

    name = 'Data Reader'
    description = 'Read supported file format to a DataFrame'
    icon = "../assets/FileReader.svg"

    class Inputs:
        pass

    class Outputs:
        data_frame = widget.Output('DataFrame', pyspark.sql.DataFrame)

    FORMAT_LIST = [
        'csv',
        'tsv'
    ]
    OPTIONS_LIST = [
        Parameter('header', 'true', 'Include Header?', 'str')
    ]
    format = settings.Setting('csv')
    file_path = settings.Setting('')

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.controlArea.setMinimumWidth(400)
        gui.comboBox(self.controlArea, self, 'format', items=OWDataFrameReader.FORMAT_LIST, label='File format', sendSelectedValue=True)
        file_browser_box = gui.hBox(self.controlArea, 'File path')
        gui.lineEdit(file_browser_box, self, 'file_path', orientation=QtCore.Qt.Horizontal)
        gui.toolButton(file_browser_box, self, 'Browse...', callback=self.browse_file)
        gui.button(self.controlArea, self, 'Apply', callback=self.apply)

    def browse_file(self):
        file = QtWidgets.QFileDialog.getOpenFileName()[0]
        if file:
            self.controls.file_path.setText(file)

    def apply(self):
        # OWDataFrameReader.FORMAT_LIST[self.format][1]) \

        df = self.sqlContext.read.format('csv')\
            .options(header='true', inferschema='true', delimiter='\t' if self.format == 'tsv' else ',')\
            .load(self.file_path)

        # add a id column
        if '_id' not in df.columns:
            df = df.withColumn("_id", monotonically_increasing_id())

        self.Outputs.data_frame.send(df)
        self.hide()
