from collections import OrderedDict

from Orange.widgets import widget
from pyspark.ml import feature
from weta.gui.spark_base import Parameter

from weta.gui.spark_estimator import SparkTransformer


class OWStopWordsRemover(SparkTransformer, widget.OWWidget):
    priority = 4
    name = "Stopwords Remover"
    description = "StopWords Remover"
    icon = "../assets/StopWordsRemover.svg"

    learner = feature.StopWordsRemover

    class Parameters:
        inputCol = Parameter(str, 'text', 'Input column', input_column=True, input_dtype=Parameter.T_ARRAY_STRING)
        outputCol = Parameter(str, 'tokens', 'Output column', output_column=True)
        # 'stopWords': Parameter(list, None, 'Stopwords list')
        caseSensitive = Parameter(bool, False, 'Case sensitive')