import weta.mllib.nltk_tokenizer
from pyspark.ml import classification
from pyspark.ml import feature
from pyspark.ml import regression
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, struct

def spark_transformer(env, transformer_cls, inputs, settings):
    assert 'DataFrame' in inputs
    input_data_frame = inputs['DataFrame']
    params = settings
    transformer = transformer_cls()
    transformer.setParams(**params)
    output_data_frame = transformer.transform(input_data_frame)

    return {
        'DataFrame': output_data_frame,
        'Transformer': transformer
    }


def spark_estimator(env, estimator_cls, inputs, settings):
    assert 'DataFrame' in inputs
    input_data_frame = inputs['DataFrame']
    params = settings
    estimator = estimator_cls()
    estimator.setParams(**params)
    model = estimator.fit(input_data_frame)  # model
    output_dataframe = model.transform(input_data_frame)

    return {
        'DataFrame': output_dataframe,
        'Model': model
    }


# ----------------------- preprocess -----------------------

def spark_ngram(env, inputs, settings):
    return spark_transformer(env, feature.NGram, inputs, settings)


def spark_nltk_tokenizer(env, inputs, settings):
    return spark_transformer(env, weta.mllib.nltk_tokenizer.NLTKTokenizer, inputs, settings)


def spark_regex_tokenizer(env, inputs, settings):
    return spark_transformer(env, feature.RegexTokenizer, inputs, settings)


def spark_stopwords_remover(env, inputs, settings):
    feature.StopWordsRemover.loadDefaultStopWords('english')
    return spark_transformer(env, feature.StopWordsRemover, inputs, settings)


def spark_tokenizer(env, inputs, settings):
    return spark_transformer(env, feature.Tokenizer, inputs, settings)


# ----------------------- feature ---------------------
def spark_hashing_tf(env, inputs, settings):
    return spark_transformer(env, feature.HashingTF, inputs, settings)


def spark_idf(env, inputs, settings):
    return spark_estimator(env, feature.IDF, inputs, settings)


def spark_string_indexer(env, inputs, settings):
    return spark_estimator(env, feature.StringIndexer, inputs, settings)

def spark_word2vec(env, inputs, settings):
    return spark_estimator(env, feature.Word2Vec, inputs, settings)

def spark_vector_assembler(env, inputs, settings):
    df = inputs['DataFrame']
    settings['inputCols'] = [df.columns[i] for i in settings['inputCols']]
    return spark_transformer(env, feature.VectorAssembler, inputs, settings)


# ----------------------- learn -----------------------

def spark_decision_tree_classifier(env, inputs, settings):
    return spark_estimator(env, classification.DecisionTreeClassifier, inputs, settings)


def spark_linear_regression(env, inputs, settings):
    return spark_estimator(env, regression.LinearRegression, inputs, settings)


def spark_logistic_regression(env, inputs, settings):
    return spark_estimator(env, classification.LogisticRegression, inputs, settings)


def spark_naive_bayes(env, inputs, settings):
    return spark_estimator(env, classification.NaiveBayes, inputs, settings)


def spark_model_transform(env, inputs, settings):
    model = inputs['Model']
    df = inputs['DataFrame']
    output_data_frame = model.transform(df)

    return {
        'Model': model,
        'DataFrame': output_data_frame
    }

# ----------------------- data -----------------------

def dataframe_reader(env, inputs, settings):
    df = env['sqlContext'].read.format(settings['format']) \
        .options(header='true', inferschema='true') \
        .load(settings['file_path'])

    return {'DataFrame': df}


def dataframe_viewer(env, inputs, settings):
    df = inputs['DataFrame']
    return {'DataFrame': df}


def dataframe_joiner(env, inputs, settings):
    df1 = inputs['DataFrame1']
    df2 = inputs['DataFrame2']

    id_col = settings['id']
    df = df1.join(df2, [id_col])
    return {'DataFrame': df}

def dataframe_splitter(env, inputs, settings):
    df = inputs['DataFrame']
    train, test = df.randomSplit([settings['train_weight'], settings['test_weight']], seed=12345)

    return {
        'DataFrame1': train,
        'DataFrame2': test
    }

def dataframe_group(env, inputs, settings):
    sqlContext = env['sqlContext']
    df = inputs['DataFrame']
    group_col = settings['groupCol']
    list_col = settings['outputCol']
    rows = df.groupby(group_col).agg(F.collect_list(struct(*df.columns))).collect()
    out_df = sqlContext.createDataFrame(rows, [group_col, list_col])
    return {
        'DataFrame': out_df
    }

def dataframe_union(env, inputs, settings):
    df1 = inputs['DataFrame1']
    df2 = inputs['DataFrame2']
    return {
        'DataFrame': df1.union(df2)
    }

