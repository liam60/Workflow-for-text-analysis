from Orange.widgets import widget
from Orange.widgets.widget import OWWidget
import pyspark.sql
import pyspark.ml
import plotly.graph_objs as go

from weta.gui.spark_base import SparkBase


class OWNZPoliceEvaluate(SparkBase, OWWidget):
    priority = 3
    name = 'Evaluation'
    description = 'Evaluation'
    icon = "../assets/LinearRegression.svg"

    DataFrame1 = None
    DataFrame2 = None
    Model1 = None
    Model2 = None

    class Inputs:
        DataFrame1 = widget.Input('DataFrame1', pyspark.sql.DataFrame)
        Model1 = widget.Input('Model1', pyspark.ml.Model)
        DataFrame2 = widget.Input('DataFrame2', pyspark.sql.DataFrame)
        Model2 = widget.Input('Model2', pyspark.ml.Model)

    class Outputs:
        Figure = widget.Output('Figure', go.Figure)

    @Inputs.DataFrame1
    def set_data_frame1(self, df):
        self.DataFrame1 = df

    @Inputs.DataFrame2
    def set_data_frame2(self, df):
        self.DataFrame2 = df

    @Inputs.Model1
    def set_model1(self, model):
        self.Model1 = model

    @Inputs.Model2
    def set_model2(self, model):
        self.Model2 = model

    def _validate_input(self):
        if self.DataFrame1 is None or self.DataFrame2 is None or self.Model1 is None or self.Model2 is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frames / models does not exist')
            return False

        return True