from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import classification
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWLogisticRegression(SparkEstimator, widget.OWWidget):
    priority = 101
    name = "Logistic Regression"
    description = "Logistic Regression Algorithm"
    icon = "../assets/LinearRegression.svg"

    box_text = "Linear Regression"

    learner = classification.LogisticRegression

    class Parameters:
        featuresCol = Parameter(str, 'features', 'Features column', input_column=True)
        labelCol = Parameter(str, 'label', 'Label column', input_column=True)
        # predictionCol = Parameter(str, 'prediction', 'Prediction column')
        # probabilityCol = Parameter(str, "probability", 'Probability column')
        # rawPredictionCol = Parameter(str, 'rawPrediction', 'Raw probability column')
        weightCol = Parameter(str, 'weight', 'Weight column')

        maxIter = Parameter(int, 100, 'Maximum iteration')
        regParam = Parameter(float, 0.0, 'Regression Parameter')
        elasticNetParam = Parameter(float, 0.0, 'Elastic Net Parameter')
        tol =  Parameter(float, 0.000001, 'tol')
        fitIntercept = Parameter(bool, True, 'Fit intercept')
        threshold = Parameter(float, 0.5, 'Threshold')
        # thresholds = Parameter(list, None, 'Thresholds')  # list[float]
        standardization = Parameter(bool, True, 'Standardization')
        aggregationDepth = Parameter(int, 2, 'Aggregation depth')
        family = Parameter(str, 'auto', 'Family')
