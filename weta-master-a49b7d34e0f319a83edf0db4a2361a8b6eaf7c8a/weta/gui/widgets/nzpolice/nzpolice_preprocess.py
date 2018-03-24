from Orange.widgets import widget
from Orange.widgets.widget import OWWidget
import pyspark.sql
from weta.gui.spark_transformer import SparkBase


class OWNZPolicePreprocess(SparkBase, OWWidget):
    priority = 1
    name = 'Crime Preprocess'
    description = 'transform original reports to quantity or category features'
    icon = "../assets/LinearRegression.svg"

    DataFrame = None

    class Inputs:
        DataFrame = widget.Input('DataFrame', pyspark.sql.DataFrame)

    class Outputs:
        DataFrame = widget.Output('DataFrame', pyspark.sql.DataFrame)

    @Inputs.DataFrame
    def set_data_frame(self, df):
        self.DataFrame = df

    def _validate_input(self):
        if self.DataFrame is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame does not exist')
            return False

        return True
