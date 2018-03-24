from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import classification
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWDecisionTreeClassifier(SparkEstimator, widget.OWWidget):
    priority = 201
    name = "Decision Tree"
    description = "Decision Tree Classifier Algorithm"
    icon = "../assets/DecisionTree.svg"

    learner = classification.DecisionTreeClassifier
    
    class Parameters:
        featuresCol = Parameter(str, 'features', 'Features column', input_column=True)
        labelCol = Parameter(str, 'label', 'Label column', input_column=True)
        predictionCol = Parameter(str, 'prediction', 'Prediction column', input_column=True)
        probabilityCol = Parameter(str, "probability", 'Probability column', input_column=True)
        rawPredictionCol = Parameter(str, 'rawPrediction', 'Raw prediction column', input_column=True)
        maxDepth = Parameter(int, 5, 'Maximal depth')
        maxBins = Parameter(int, 32, 'Maximal bins')
        minInstancesPerNode = Parameter(int, 1, 'Minimum instance per node')
        minInfoGain = Parameter(float, 0.0, 'Minimum Information gain')
        maxMemoryInMB = Parameter(int, 256, 'Maximal Memory (MB)')
        cacheNodeIds = Parameter(bool, False, 'Cache node ids')
        checkpointInterval = Parameter(int, 10, 'Checkpoint interval')
        impurity = Parameter(str, 'gini', 'Impurity')
        seed = Parameter(int, None, 'Seed')
