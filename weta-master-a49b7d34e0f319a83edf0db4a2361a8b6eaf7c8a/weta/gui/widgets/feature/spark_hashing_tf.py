from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkTransformer


class OWHashingTF(SparkTransformer, widget.OWWidget):
    priority = 1
    name = "Hashing TF"
    description = "Hashing TF"
    icon = "../assets/CountVectorizer.svg"

    box_text = 'Hashing TF'

    learner = feature.HashingTF

    class Parameters:
        inputCol = Parameter(str, 'tokens', 'Input column', input_column=True, input_dtype=Parameter.T_ARRAY_STRING)
        outputCol = Parameter(str, 'tf', 'Output column', output_column=True)
        numFeatures = Parameter(int, 1 << 18, 'Number of features')
        binary = Parameter(bool, False, 'Binary')