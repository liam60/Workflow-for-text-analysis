from Orange.widgets import widget
from pyspark.sql import DataFrame
from weta.gui.spark_base import SparkBase, Parameter
from collections import OrderedDict


class OWDataFrameSplitter(SparkBase, widget.OWWidget):

    priority = 6

    name = 'Row Splitter'
    description = 'Split a DataFrame into train and test datasets with a fixed ratio'
    icon = '../assets/DataFrameSplitter.svg'

    DataFrame = None
    class Inputs:
        DataFrame = widget.Input('DataFrame', DataFrame)

    class Outputs:
        DataFrame1 = widget.Output('DataFrame1', DataFrame, id='train')
        DataFrame2 = widget.Output('DataFrame2', DataFrame, id='test')

    class Parameters:
        train_weight =  Parameter(float, 0.9, 'Train weight of split ratio')
        test_weight =  Parameter(float, 0.1, 'Test weight of split ratio')

    @Inputs.DataFrame
    def set_data_frame(self, data_frame):
        self.DataFrame = data_frame

    def _validate_input(self):
        if not super(OWDataFrameSplitter, self)._validate_input():
            return False

        if self.DataFrame is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame does not exist')
        else:
            self.v_apply_button.setEnabled(True)
            return True
