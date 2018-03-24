from Orange.widgets import widget
import pyspark.sql

from weta.gui.spark_base import Parameter
from weta.gui.spark_transformer import SparkBase


class OWTestCollo(SparkBase, widget.OWWidget):
    priority = 1
    name = 'Collocation'
    description = 'Collocation scores using Chi and t scores'
    icon = "../assets/Linkage.svg"

    DataFrame = None

    class Inputs:
        DataFrame = widget.Input('DataFrame', pyspark.sql.DataFrame)

    class Outputs:
        DataFrame = widget.Output('DataFrame', pyspark.sql.DataFrame)

    class Parameters:
        cutoff = Parameter(int, 50, 'Top scores to display:')
        inputCol = Parameter(str, 'tokens', 'Input column', input_column=True, input_dtype=Parameter.T_STRING)
        sortingType = Parameter(str, 'Term_Freq', 'Choose a sorting type', items=['Term Frequency','Chi','t-score','Non perfect Chi'])

    @Inputs.DataFrame
    def set_data_frame(self, df):
        self.DataFrame = df

    def _validate_input(self):
        if self.DataFrame is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame does not exist')
            return False

        return True
    

    