from collections import OrderedDict

from pyspark.ml import feature
from Orange.widgets import widget

from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkTransformer


class OWVectorAssembler(SparkTransformer, widget.OWWidget):
    priority = 16
    name = "Vector Assembler"
    description = "VectorAssembler"
    icon = "../assets/VectorAssembler.svg"

    learner = feature.VectorAssembler

    class Parameters:
        inputCols = Parameter(list, [], 'Input columns', input_column=True, input_multiple=True)
        outputCol = Parameter(str, 'features', 'Output column', output_column=True)