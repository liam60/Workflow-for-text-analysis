from Orange.widgets import widget
import pyspark.sql

from weta.gui.spark_base import Parameter
from weta.gui.spark_transformer import SparkBase


class OWNZPoliceLinkage(SparkBase, widget.OWWidget):
    priority = 2
    name = 'Linkage for testing'
    description = 'link reports within a group and random selected reports in other groups'
    icon = "../assets/Linkage.svg"

    DataFrame1 = None
    DataFrame2 = None

    class Inputs:
        DataFrame1 = widget.Input('DataFrame1', pyspark.sql.DataFrame) # existing crimes
        DataFrame2 = widget.Input('DataFrame2', pyspark.sql.DataFrame) # new crimes

    class Outputs:
        DataFrame = widget.Output('DataFrame', pyspark.sql.DataFrame)
        RawDataFrame = widget.Output('RawDataFrame', pyspark.sql.DataFrame)

    class Parameters:
        pass

    @Inputs.DataFrame1
    def set_data_frame1(self, df):
        self.DataFrame1 = df

    @Inputs.DataFrame2
    def set_data_frame2(self, df):
        self.DataFrame2 = df

    def _validate_input(self):
        if self.DataFrame1 is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame1 does not exist')
            return False

        if self.DataFrame2 is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame2 does not exist')
            return False

        return True
