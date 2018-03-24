from collections import OrderedDict
from pyspark import SparkConf, SparkContext, SQLContext, HiveContext

class SparkEnvironment:
    _conf = SparkConf()
    _sc = None
    _hc = None
    _sqlContext = None

    @property
    def sc(self):
        if self._sc is None:
            self.create_context()

        return SparkEnvironment._sc

    @property
    def sqlContext(self):
        if self._sqlContext is None:
            self.create_context()
        return SparkEnvironment._sqlContext

    @property
    def hc(self):
        if self._hc is None:
            self.create_context()
        return SparkEnvironment._hc

    @staticmethod
    def create_context(parameters=None):
        if parameters is None:
            parameters = OrderedDict()
            parameters['spark.app.name'] = 'weta_workflow'
            parameters['spark.master'] = 'local'  # 'yarn'
            parameters["spark.executor.instances"] = "8"
            parameters["spark.executor.cores"] = "8"
            parameters["spark.executor.memory"] = "2g"
            parameters["spark.driver.cores"] = "4"
            parameters["spark.driver.memory"] = "1g"
            parameters["spark.logConf"] = "false"
            parameters["spark.app.id"] = "dummy"
            # parameters['spark.debug.maxToStringFields'] = 100

        cls = SparkEnvironment
        if cls._sc:
            cls._sc.stop()

        for key, parameter in parameters.items():
            cls._conf.set(key, parameter)

        cls._sc = SparkContext(conf=cls._conf)
        cls._sqlContext = SQLContext(cls._sc)
        cls._hc = HiveContext(cls._sc)