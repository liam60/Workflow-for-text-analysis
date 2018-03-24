from Orange.widgets import widget
from pyspark.sql import DataFrame
from weta.gui.spark_base import SparkBase, Parameter
from collections import OrderedDict


class OWDataFrameGroup(SparkBase, widget.OWWidget):

    priority = 10

    name = 'Column Group'
    description = 'Specify a column to group, other columns into List'
    icon = '../assets/DataFrameSplitter.svg'

    DataFrame = None
    class Inputs:
        DataFrame = widget.Input('DataFrame', DataFrame)

    class Outputs:
        DataFrame = widget.Output('DataFrame', DataFrame)

    class Parameters:
        groupCol = Parameter(str, 'id', 'Group column', input_column=True)
        outputCol = Parameter(str, 'list', 'List column after grouping', output_column=True)

    @Inputs.DataFrame
    def set_data_frame(self, data_frame):
        self.DataFrame = data_frame

    def _validate_input(self):
        if not super(OWDataFrameGroup, self)._validate_input():
            return False

        if self.DataFrame is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame does not exist')
        else:
            self.v_apply_button.setEnabled(True)
            return True
