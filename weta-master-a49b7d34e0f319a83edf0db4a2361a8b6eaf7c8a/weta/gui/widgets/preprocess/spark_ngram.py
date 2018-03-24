from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkTransformer


class OWNGram(SparkTransformer, widget.OWWidget):
    priority = 3
    name = "NGram"
    description = "NGram"
    icon = "../assets/NGram.svg"

    learner = feature.NGram

    class Parameters:
        n = Parameter(int, 2, 'N')
        inputCol = Parameter(str, 'text', 'Input column', input_column=True, input_dtype=Parameter.T_ARRAY_STRING)
        outputCol = Parameter(str, 'tokens', 'Output column', output_column=True)
