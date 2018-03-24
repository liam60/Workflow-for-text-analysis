from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import regression
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWLinearRegression(SparkEstimator, widget.OWWidget):
    priority = 101
    name = "Linear Regression"
    description = "Linear Regression Algorithm"
    icon = "../assets/LinearRegression.svg"

    learner = regression.LinearRegression
    
    class Parameters:
        featuresCol = Parameter(str, 'features', 'Feature column', input_column=True)
        labelCol = Parameter(str, 'label', 'Label column', input_column=True)
        # weightCol =  Parameter(str, 'weight', 'Weight Column')

        maxIter = Parameter(int, 100, 'Maximal iteration')
        regParam = Parameter(float, 0.0, 'Regression Parameter')
        elasticNetParam = Parameter(float, 0.0, 'Elastic Net Parameter')
        tol = Parameter(float, 0.000001, 'tol')
        fitIntercept = Parameter(bool, True, 'Fit intercept')
        standardization = Parameter(bool, False, 'Standardization')
        solver = Parameter(str, 'auto', 'Solver')
        aggregationDepth = Parameter(int, 2, 'Aggregation depth')
