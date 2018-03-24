from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWIDF(SparkEstimator, widget.OWWidget):
    priority = 11
    name = "IDF"
    description = "Document IDF transformer"
    icon = "../assets/IDF.svg"

    learner = feature.IDF

    class Parameters:
        inputCol = Parameter(str, 'tf', 'Input column', input_column=True, input_dtype=Parameter.T_VECTOR)
        outputCol = Parameter(str, 'tfidf', 'Output column', output_column=True)
        minDocFreq = Parameter(int, 0, 'Minimum document frequency')
