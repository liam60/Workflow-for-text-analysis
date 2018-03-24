from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWStringIndexer(SparkEstimator, widget.OWWidget):
    priority = 12
    name = "StringIndexer"
    description = "StringIndexer"
    icon = "../assets/OneHotEncoder.svg"

    learner = feature.StringIndexer

    class Parameters:
        inputCol = Parameter(str, 'category', 'Input column', input_column=True, input_dtype=Parameter.T_STRING)
        outputCol = Parameter(str, 'category_index', 'Output column', output_column=True)