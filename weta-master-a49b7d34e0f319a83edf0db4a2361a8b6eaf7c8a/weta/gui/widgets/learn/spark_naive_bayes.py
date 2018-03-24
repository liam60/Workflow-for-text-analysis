from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import classification
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkEstimator


class OWNaiveBayes(SparkEstimator, widget.OWWidget):
    priority = 201
    name = "NaiveBayes"
    description = "NaiveBayes Algorithm"
    icon = "../assets/LinearRegression.svg"
    
    learner = classification.NaiveBayes
    
    class Parameters:
        featuresCol =  Parameter(str, 'features', 'Feature column', input_column=True)
        labelCol =  Parameter(str, 'label', 'Label column', input_column=True)
        # predictionCol =  Parameter(str, 'prediction', 'Prediction column')
        # probabilityCol =  Parameter(str, 'probability', 'Probability column')
        # rawPredictionCol =  Parameter(str, 'rawPrediction', 'Raw probability column')
        weightCol =  Parameter(str, 'weight', 'Weight Column')

        smoothing =  Parameter(float, 1.0, 'Smoothing')
        # modelType =  Parameter(str, 'multinomial', 'Model Type')
        # 'thresholds =  Parameter(list, None, 'Thresholds') # list[float]