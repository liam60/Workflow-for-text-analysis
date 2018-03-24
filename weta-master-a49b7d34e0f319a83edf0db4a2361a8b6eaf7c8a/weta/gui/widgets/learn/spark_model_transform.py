from Orange.widgets import widget
from pyspark.ml import Model
import pyspark
from collections import OrderedDict

from weta.gui.spark_base import Parameter
from weta.gui.spark_transformer import SparkTransformer


class OWModelTransformation(SparkTransformer, widget.OWWidget):
    name = "Model Transformation"
    description = "A Model Transformer of the Spark ml api"
    icon = "../assets/ModelTransformation.svg"

    Model = None

    class Inputs(SparkTransformer.Inputs):
        Model = widget.Input("Model", Model)

    class Outputs:
        DataFrame = widget.Output("DataFrame", pyspark.sql.DataFrame)
        Model = widget.Output("Model", Model)

    class Parameters:
        inputCol = Parameter(str, 'input', 'Input column', input_column=True)
        # outputCol = Parameter(str, 'output', 'Output column', output_column=True)

    @Inputs.Model
    def set_model(self, model):
        self.Model = model

    def _validate_input(self):
        if not super(OWModelTransformation, self)._validate_input():
            return False

        if self.Model is None:
            self.error('Input Model does not exist')
            return False

        # if self.inputCol not in self.input_data_frame.columns:
        #     self.inputCol = self.input_transformer.inputCol
        # self.input_dtype = self.Model.input_dtype
        return True


