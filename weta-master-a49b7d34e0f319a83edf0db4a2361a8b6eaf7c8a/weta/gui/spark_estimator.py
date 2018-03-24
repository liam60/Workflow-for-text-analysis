from Orange.widgets import widget
from pyspark.ml import Model
import pyspark

from .spark_transformer import SparkTransformer


class SparkEstimator(SparkTransformer):

    class Outputs:
        DataFrame = widget.Output("DataFrame", pyspark.sql.DataFrame)
        Model = widget.Output('Model', Model)


    def _apply(self, params):
        estimator = self.learner()  # estimator
        estimator.setParams(**params)
        model = estimator.fit(self.DataFrame)  # model

        output_model = model
        output_model.input_dtype = self.input_dtype  # attach a required input dtype
        output_data_frame = model.transform(self.DataFrame)

        self.Outputs.DataFrame.send(output_data_frame)
        self.Outputs.Model.send(output_model)