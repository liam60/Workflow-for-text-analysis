from Orange.widgets import widget
from weta.gui.spark_base import Parameter
from weta.gui.spark_transformer import SparkTransformer
from collections import OrderedDict
from pyspark.ml import Transformer


class OWGenericTransformation(SparkTransformer, widget.OWWidget):
    name = "Transformation"
    description = "A Generic Transformer of the Spark ml api"
    icon = "../assets/Transformation.svg"

    Transformer = None

    class Inputs(SparkTransformer.Inputs):
        Transformer = widget.Input("Transformer", Transformer)

    class Parameters:
        inputCol = Parameter(str, 'input', 'Input column', input_column=True)
        outputCol = Parameter(str, 'output', 'Output column', output_column=True)

    @Inputs.Transformer
    def set_transformer(self, transformer):
        self.Transformer = transformer

    def _validate_input(self):
        if not super(OWGenericTransformation, self)._validate_input():
            return False

        if self.Transformer is None:
            self.error('Input Transformer does not exist')
            return False

        self.input_dtype = self.Transformer.input_dtype
        return True

    def _apply(self, params):
        transformer = self.Transformer
        output_data_frame = transformer.transform(self.DataFrame)
        self.Outputs.DataFrame.send(output_data_frame)
        self.Outputs.Transformer.send(self.Transformer)