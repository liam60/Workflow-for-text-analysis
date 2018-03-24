from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWCountVectorizer(SparkEstimator, widget.OWWidget):
    priority = 1
    name = "Count Vectorizer"
    description = "Count Vectorizer"
    icon = "../assets/CountVectorizer.svg"

    learner = feature.CountVectorizer
    class Parameters:
        inputCol = Parameter(str, 'tokens', 'Input column', input_column=True, input_dtype=Parameter.T_ARRAY_STRING)
        outputCol = Parameter(str, 'vector', 'Output column', output_column=True)
        minTF = Parameter(float, 1.0, 'Minimum term frequency'),
        minDF = Parameter(float, 1.0, 'Minimum document frequency')
        vocabSize = Parameter(int, 1 << 18, 'Vocabulary size')
        binary = Parameter(bool, False, 'Binary')