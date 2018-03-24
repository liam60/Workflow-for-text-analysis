from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkTransformer


class OWTokenizer(SparkTransformer, widget.OWWidget):
    priority = 2
    name = "Tokenizer"
    description = "Simple Tokenizer (mainly for sentence tokenization)"
    icon = "../assets/Tokenizer.svg"

    learner = feature.Tokenizer

    class Parameters:
        inputCol = Parameter(str, 'text', 'Input column', input_column=True, input_dtype=Parameter.T_STRING)
        outputCol = Parameter(str, 'tokens', 'Output column', output_column=True)
