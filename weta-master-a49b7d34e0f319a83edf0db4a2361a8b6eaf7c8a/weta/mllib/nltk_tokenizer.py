import string as string_module

import os
import nltk
import pyspark.ml
from pyspark import keyword_only
from pyspark.ml.param.shared import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType

CURRENT_PATH = os.path.realpath(__file__)
nltk.data.path.append(os.path.realpath(CURRENT_PATH + '/../../../extras/nltk_data'))

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string_module.punctuation)


# For vectorizing text
def stem_tokens(tokens):
    tokens1 = []
    for token in tokens:
        if len(token) > 0:
            tokens1.append(token)
    return [stemmer.stem(item) for item in tokens1]


# Normalizes text (i.e, tokenizes and then stems words)
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


udf_tokenize_and_normalize = udf(normalize, ArrayType(StringType()))


class NLTKTokenizer(pyspark.ml.Transformer, HasInputCol, HasOutputCol):
    """
    Use NLTK tokenizer: lower case, remove punctuation, tokenize, stem, filter empty token
    """

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        """
        __init__(self, inputCol=None, outputCol=None)
        """
        super(NLTKTokenizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        """
        setParams(self, inputCol=None, outputCol=None)
        Sets params for this Tokenizer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        return dataset.withColumn(self.getOutputCol(), udf_tokenize_and_normalize(dataset[self.getInputCol()]))
