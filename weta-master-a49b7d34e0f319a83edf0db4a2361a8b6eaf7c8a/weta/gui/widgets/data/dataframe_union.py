from Orange.widgets import widget
from pyspark.sql import DataFrame
from weta.gui.spark_base import SparkBase, Parameter


class OWDataFrameUnion(SparkBase, widget.OWWidget):

    priority = 11

    name = 'Row Union'
    description = 'Union two DataFrame into one by rows'
    icon = '../assets/DataFrameUnion.svg'

    DataFrame1 = None  # type: DataFrame
    DataFrame2 = None  # type: DataFrame

    class Inputs:
        DataFrame1 = widget.Input('DataFrame1', DataFrame, id='df1')
        DataFrame2 = widget.Input('DataFrame2', DataFrame, id='df2')

    class Outputs:
        DataFrame = widget.Output('DataFrame', DataFrame)

    class Parameters:
        pass

    @Inputs.DataFrame1
    def set_data_frame1(self, data_frame):
        self.DataFrame1 = data_frame

    @Inputs.DataFrame2
    def set_data_frame2(self, data_frame):
        self.DataFrame2 = data_frame

    def _validate_input(self):
        if not super(OWDataFrameUnion, self)._validate_input():
            return False

        if self.DataFrame1 is None or self.DataFrame2 is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame does not exist')
            return False
        else:
            self.v_apply_button.setEnabled(True)
            return True
