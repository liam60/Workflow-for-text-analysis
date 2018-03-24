from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkTransformer


class OWOneHotEncoder(SparkTransformer, widget.OWWidget):
    priority = 2
    name = "One-Hot Encoder"
    description = "One-Hot Encoder"
    icon = "../assets/OneHotEncoder.svg"

    learner = feature.OneHotEncoder

    class Parameters:
        dropLast = Parameter(bool, True, 'Drop the last category')
        inputCol = Parameter(str, 'tokens', 'Input column (%s)', input_column=True, input_dtype=Parameter.T_ARRAY_STRING)
        outputCol = Parameter(str, 'features', 'Output column', output_column=True)