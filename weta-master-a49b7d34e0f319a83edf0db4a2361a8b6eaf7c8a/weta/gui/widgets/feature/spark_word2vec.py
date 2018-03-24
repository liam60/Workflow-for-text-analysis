from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWWord2Vec(SparkEstimator, widget.OWWidget):
    priority = 12
    name = "Word2Vec"
    description = "Word2Vec"
    icon = "../assets/Word2Vec.svg"

    learner = feature.Word2Vec

    class Parameters:
        inputCol = Parameter(str, 'tokens', 'Input column', input_column=True, input_dtype=Parameter.T_ARRAY_STRING)
        outputCol = Parameter(str, 'vector', 'Output1 column', output_column=True)
        vectorSize = Parameter(int, 100, 'Vector size')
        minCount = Parameter(int, 5, 'Minimum count')
        numPartitions = Parameter(int, 1, 'Number of partitions ')
        stepSize = Parameter(float, 0.025, 'Step size')
        maxIter = Parameter(int, 1, 'Maximum Iteration')
        seed = Parameter(int, None, 'Seed')
        windowSize = Parameter(int, 5, 'Window size')
        maxSentenceLength = Parameter(int, 1000, 'Maximum sentence length')
