import weta.mllib.nltk_tokenizer
from Orange.widgets import widget
from weta.gui.spark_base import Parameter
from weta.gui.spark_estimator import SparkTransformer


class OWNLTKTokenizer(SparkTransformer, widget.OWWidget):
    priority = 1
    name = "NLTK Tokenizer"
    description = "NLTK Tokenizer"
    icon = "../assets/NLTKTokenizer.svg"

    learner = weta.mllib.nltk_tokenizer.NLTKTokenizer

    class Parameters:
        inputCol = Parameter(str, 'text', 'Input column', input_column=True, input_dtype=Parameter.T_STRING)
        outputCol = Parameter(str, 'tokens', 'Output column', output_column=True)
        # 'minTokenLength': Parameter(int, 1, 'Minimum token length'),
        # 'removePunctuation': Parameter(bool, True, 'Remove punctuation?'),
        # 'stem': Parameter(bool, True, 'Stem?'),
        # 'toLowercase': Parameter(bool, True, 'Convert to lower case?')


