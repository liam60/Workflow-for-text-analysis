import datetime
import hashlib
import math
import random
import re
from itertools import combinations

import numpy
# import scipy.misc
# from scipy import spatial
import scipy
from Orange.widgets.widget import OWWidget
from nltk.corpus import wordnet
from nltk.wsd import lesk
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, struct
from pyspark.sql.types import IntegerType, StringType, ArrayType, Row, DoubleType, StructField, StructType

from scipy.optimize import linear_sum_assignment


@udf(returnType=StringType())
def p_locationType(string):
    if "Farm" in string:
        return 1
    if "Residenti" in string:
        return 2
    if "Commerci" in string:
        return 3
    if "Publi" in string:
        return 4
    return "UNKNOWN"


@udf(returnType=IntegerType())
def p_ordinalDate(string):
    start = datetime.datetime.strptime(string.strip(), '%d/%m/%Y')
    return start.toordinal()


@udf(returnType=IntegerType())
def p_time(string):
    hours = int(string.split(":")[0])
    if "PM" in string: hours += 12
    return hours


@udf(returnType=StringType())
def p_entryLocation(string):
    vectors1 = ['PREMISES-REAR', 'PREMISES-FRONT', 'PREMISES-SIDE']
    for x in vectors1:
        if x in string: return x
    return "UNKNOWN"


@udf(returnType=StringType())
def p_entryPoint(string):
    vectors2 = ['POINT OF ENTRY-DOOR', 'POINT OF ENTRY-WINDOW', \
                'POINT OF ENTRY-FENCE', 'POINT OF ENTRY-DOOR: GARAGE']
    vectors3 = ['POE - DOOR', 'POE - WINDOW', 'POE - FENCE', 'POE - GARAGE']
    for x, y in list(zip(vectors2, vectors3)):
        if x in string or y in string: return x
    return "UNKNOWN"


@udf(returnType=IntegerType())
def p_dayOfWeek(string):
    start = datetime.datetime.strptime(string, '%d/%m/%Y')
    return start.weekday()


@udf(returnType=StringType())
def p_northingEasting(string, string2):
    return "%s-%s" % (string, string2)


@udf(returnType=StringType())
def p_methodOfEntry(string):
    if string is None:
        return ''

    narrative = string.split("__________________________________ CREATED BY")[-1]
    if 'NARRATIVE' in narrative or 'CIRCUMSTANCES' in narrative:
        narrative = re.split('NARRATIVE|CIRCUMSTANCES', narrative)[-1]
        narrative = re.split("\*|:", narrative[1:])[0]
    return narrative


# Classifies if the search was messy
@udf(returnType=IntegerType())
def p_messy(string):
    negations = ["NOT ", "NO ", "HAVEN'T", "DIDN'T", 'DIDNT', "HAVENT"]
    messywords = ['MESSY', 'MESSIL', 'RUMMAG', 'TIPPED']
    sentences = [sentence + '.' for sentence in string.split(".") if any(word in sentence for word in messywords)]
    c = 0
    for x in sentences:
        if any(word in x for word in negations):
            c -= 1
        else:
            c += 1
    return 1 if c > 0 else 0


@udf(returnType=StringType())
def p_signature(string):
    if "DEFECA" in string:
        return 1
    if "URINAT" in string:
        return 2
    if "MASTURB" in string:
        return 3
    if "GRAFFIT" in string:
        return 4
    return "UNKNOWN"


@udf(returnType=IntegerType())
def p_propertySecure(string):
    verbs = ['LOCKED', 'FENCED', 'GATED', 'SECURED', 'BOLTED']
    negations = ["NOT ", "NO ", "HAVEN'T", "DIDN'T", 'DIDNT', "HAVENT"]
    c = 0
    sentences = [sentence + '.' for sentence in string.split(".") if any(word in sentence for word in verbs)]
    for x in sentences:
        if any(word in x for word in negations):
            c -= 1
        else:
            c += 1
    return 1 if c > 0 else 0

import os
import nltk
from nltk.parse.stanford import StanfordDependencyParser
import string as string_module

CURRENT_PATH = os.path.realpath(__file__)
nltk.data.path.append(os.path.realpath(CURRENT_PATH + '/../../../extras/nltk_data'))
STANFORD_PARSER_PATH = os.path.realpath(CURRENT_PATH + '/../../../extras/stanford-parser')
stemmer = nltk.stem.porter.PorterStemmer()
parser = StanfordDependencyParser(
    path_to_models_jar=STANFORD_PARSER_PATH+'/stanford-parser-3.8.0-models.jar',
    model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
    path_to_jar=STANFORD_PARSER_PATH + '/stanford-parser.jar',
    java_options='-Xmx1000M',
    verbose=False)
remove_punctuation_map = dict((ord(char), None) for char in string_module.punctuation)
unigram_tagger = nltk.tag.UnigramTagger(nltk.corpus.brown.tagged_sents())
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# For vectorizing text
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


# Normalizes text (i.e, tokenizes and then stems words)
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


@udf(returnType=ArrayType(StringType()))
def p_propertyStolenList(string):
    if "PROPERTY" not in string:
        return []
    property_list = " ".join(
        [re.split(':|_', listing)[0] for listing in re.split("PROPERTY LIST SUMMARY:|PROPERTY STOLEN:", string)])
    text = normalize(property_list)
    tagged = unigram_tagger.tag(text)
    removable = ['modus', 'operandi', 'call', 'with', 'list', 'of', 'location', 'point', 'entry', 'value', 'property'
                                                                                                           'police',
                 'stage', 'name', 'details', 'insured', 'victim', 'address']
    o = []
    for x in tagged:
        if (not (x[1] in ["NN", "NNS"])) or (x[0] in removable):
            pass
        else:
            if not len(x[0]) < 3:
                o.append(x[0])
    return o


@udf(returnType=ArrayType(StringType()))
def p_pullMOTags(string):
    sentences = sent_tokenizer.tokenize(string)
    sentences = [sent.lower().capitalize() for sent in sentences]
    x_relations = []
    for sent in sentences:
        if len(sent.split(" ")) > 100: continue
        try:
            parsed = parser.raw_parse(sent)
            triples = [parse.triples() for parse in parsed]
            selected = [triple for triple in triples[0] if (triple[1] in ("dobj", "nsubjpass"))]
        except:
            continue
        for x in selected:
            x_relations.append(x)
    return x_relations


@udf(returnType=StringType())
def p_narrative_hash(narrative):
    return hashlib.sha224(narrative.encode('utf-8')).hexdigest()


def d_straightLineDistance(x, y):
    xsplit = x.split('-')
    ysplit = y.split('-')
    xN = int(xsplit[0])
    xE = int(xsplit[1])
    yN = int(ysplit[0])
    yE = int(ysplit[1])

    return int(math.sqrt(abs(xE - yE) ** 2 + abs(xN - yN) ** 2))


# 24 hour time difference
def d_timeDifference(x, y):
    diff = abs(int(x) - int(y))
    return min(diff, 24 - diff)


# 7 day week
def d_dayDifference(x, y):
    diff = abs(int(x) - int(y))
    return min(diff, 7 - diff)


# absolute
def d_distance(x, y):
    return abs(int(x) - int(y))


# 1 if same else 0
def d_nominalDistance(x, y):
    if x == 'UNKNOWN' or y == 'UNKNOWN': return None  # '?'
    return 1 if x == y else 0


def d_cosineTFIDF(xvec, yvec):
    if len(xvec) == 0 or len(yvec) == 0: return None  # '?'

    # Return cosine similarity
    return (xvec.dot(yvec) / (xvec.norm(2) * yvec.norm(2))).item()  # 1 - spatial.distance.cosine(xvec, yvec)


# As above, but for 2 grams
def d_cosineTFIDF2(xvec, yvec):
    if len(xvec) == 0 or len(yvec) == 0: return None  # '?'

    if xvec.norm(2) == 0 or yvec.norm(2) == 0:
        return None
    # Return cosine similarity
    return (xvec.dot(yvec) / (xvec.norm(2) * yvec.norm(2))).item()  # 1 - spatial.distance.cosine(xvec, yvec)


# As above, but the two grams are parsed verb-noun pairs
def d_cosineMO(xvec, yvec):
    if len(xvec) == 0 or len(yvec) == 0: return None  # '?'

    # Return cosine similarity
    return (xvec.dot(yvec) / (xvec.norm(2) * yvec.norm(2))).item()  # 1 - spatial.distance.cosine(xvec, yvec)

# def d_moSim(x, y):
#     if len(x) == 0 or len(y) == 0: return None#'?'
#
#     similarity = 0
#     for word in x:
#         for wordy in y:
#             if word == wordy:
#                 similarity += 1 * IDFSPAIRS[word]
#
#     return similarity / ((len(x) + 1) * (len(y) + 1))


# Uses wordnet to discover path similarity between lists
def d_wordNet(x, y):
    if len(x) < 1 or len(y) < 1:
        return None  # '?'

    def getUnique(sent, word):
        return lesk(sent, word, pos=wordnet.NOUN)

    # Get word sets
    sensesx = []
    for word in x:
        try:
            sensesx.append(getUnique(x, word))
        except IndexError:
            continue
    sensesy = []
    for word in y:
        try:
            sensesy.append(getUnique(y, word))
        except IndexError:
            continue

    # Form matrix of similarities
    matrix = []
    for wordx in sensesx:
        current = []
        if wordx is None: continue
        for wordy in sensesy:
            if wordy is None: continue
            current.append(wordx.lch_similarity(wordy))
        matrix.append(current)

    # Inverse costs
    max = 0
    for m in matrix:
        for mm in m:
            if mm > max: max = mm
    for m in matrix:
        for mm in m:
            mm = max - mm

    # Find max matches
    cost = numpy.array(matrix)
    row, col = linear_sum_assignment(cost)

    return (cost[row, col].sum() / len(matrix)).item()


# Uses wordnet to discover path similarity between lists
def d_wordNetNormalizedAdditive(x, y):
    if len(x) < 1 or len(y) < 1:
        return None  # '?'

    def getUnique(sent, word):
        return lesk(sent, word, pos=wordnet.NOUN)

    # Get word sets
    sensesx = []
    for word in x:
        try:
            sensesx.append(getUnique(x, word))
        except IndexError:
            continue
    sensesy = []
    for word in y:
        try:
            sensesy.append(getUnique(y, word))
        except IndexError:
            continue

    score = 0
    for sx in sensesx:
        for sy in sensesy:
            try:
                score += sx.lch_similarity(sy)
            except AttributeError:
                continue
    return score / (len(sensesx) * len(sensesy))


def d_listSimilarity(x, y):
    if len(x) == 0 or len(y) == 0: return None  # '?'

    similarity = 0
    for word in x:
        for wordy in y:
            if word == wordy:
                similarity += 1

    return similarity / ((len(x) + 1) * (len(y) + 1))


###################################################
from collections import OrderedDict

FEATURES_TO_USE = OrderedDict({
    "locationType": ('location_type', p_locationType, d_nominalDistance, IntegerType()),
    'ordinalDate': ('occurrence_start_date', p_ordinalDate, d_distance, IntegerType()),
    'time': ('occurrence_start_time', p_time, d_timeDifference, IntegerType()),
    'entryLocation': ('narrative', p_entryLocation, d_nominalDistance, IntegerType()),
    'entryPoint': ('narrative', p_entryPoint, d_nominalDistance, IntegerType()),
    'dayOfWeek': ('occurrence_start_date', p_dayOfWeek, d_dayDifference, IntegerType()),
    'northingEasting': (
        ('nztm_location_northing', 'nztm_location_easting'), p_northingEasting, d_straightLineDistance, IntegerType()),

    'methodOfEntry': ('narrative', p_methodOfEntry, None),  # non final feature
    'messy': ('methodOfEntry', p_messy, d_nominalDistance, IntegerType()),
    'propertySecure': ('narrative', p_propertySecure, d_nominalDistance, IntegerType()),
    'propertyStolenWordnet': ('narrative', p_propertyStolenList, d_wordNet, DoubleType()),
    'cosineTFIDF': (None, None, d_cosineTFIDF, DoubleType()),
    'cosineTFIDF2': (None, None, d_cosineTFIDF2, DoubleType()),
})


def nzpolice_preprocess(env, inputs, settings):
    df = inputs['DataFrame']

    df = df.na.fill({'Narrative': ''})
    # df.na.drop(subset=["Narrative"])

    features = []

    for feature in FEATURES_TO_USE:
        t = FEATURES_TO_USE[feature]
        in_cols = t[0]
        udf_func = t[1]
        if udf_func is not None:
            params = (df[c] for c in in_cols) if isinstance(in_cols, tuple) else [df[in_cols]]
            df = df.withColumn(feature, udf_func(*params))
            features.append(feature)

    features = ['crime_id']
    features.extend([column for column in df.columns if column in FEATURES_TO_USE.keys()])
    df = df.select(*features)

    return {'DataFrame': df}

def _group_crimes():
    import psycopg2

    try:
        conn = psycopg2.connect("dbname='nzpolice' user='postgres' host='localhost' password='postgres'")
    except:
        print("I am unable to connect to the database")

    cur = conn.cursor()
    cur.execute("""
      select "crime_id", array_agg(DISTINCT "person_id" ORDER BY "person_id" ASC) as offenders from "actions" 
      group by "crime_id" having crime_id != '9703eba752f3094f0d862f6d7c18c2c9798cb4960096a68b8cfcbc4d8f0cbab4'
    """)

    rows = cur.fetchall()

    groups = {}

    for row1 in rows:
        for row2 in rows:
            if row1[0] == row2[0]:
                continue
            common = list(set(row1[1]) & set(row2[1]))
            if len(common) == 0:
                continue
            common.sort()
            crimes = groups.setdefault('+'.join(common), set())
            crimes.add(row1[0])
            crimes.add(row2[0])
    return {group: list(groups[group]) for group in groups.keys()}

def set_progress(env, percentage):
    if env['ui'] is not None:
        widget: OWWidget = env['ui']
        widget.progressBarSet(percentage)

def nzpolice_link_test(env, inputs, settings):
    crimes1 = {row['crime_id']: row for row in inputs['DataFrame1'].collect()}
    crimes2 = {row['crime_id']: row for row in inputs['DataFrame2'].collect()}

    links = []
    for crime_id2 in crimes2.keys():
        for crime_id1 in crimes1.keys():
            links.append((crime_id2, crime_id1, 1.0, 1))
    set_progress(env, 10)

    return _linkage(env, {**crimes1, **crimes2}, links)

def nzpolice_link_train(env, inputs, settings):

    # convert dataframe to a dict
    crimes = {row['crime_id']: row for row in inputs['DataFrame'].collect()}
    set_progress(env, 10)

    # group crimes with common offender(s)
    groups = _group_crimes()
    set_progress(env, 15)

    # stat for selection
    print('Start statistics for selection...')

    NUM_GROUPS = len(groups)
    NUM_LINKED = 0
    for group in groups:
        length = len(groups[group])
        if length == 1:
            continue
        NUM_LINKED += scipy.misc.comb(length, 2, exact=True)  # combination length*(length-1)-1

    NUM_TO_SELECT = int(math.ceil(NUM_LINKED / NUM_GROUPS)) * settings['select_ratio']

    print('%d groups, %d linked, %d unlinked with %d select/r on average' % (
        NUM_GROUPS, NUM_LINKED, NUM_TO_SELECT * len(crimes), NUM_TO_SELECT))
    set_progress(env, 20)

    # balancing_ratio = NUM_TO_SELECT * len(reports) / (NUM_LINKED + NUM_TO_SELECT * len(reports))

    # combination to get pairwise crimes
    print('Start links combination...')
    links = []
    for group in groups:
        group_weight = 1 / len(groups[group])
        internal_group_links = [t + (group_weight, 1) for t in combinations(groups[group], 2)]
        external_group_links = []

        for report in groups[group]:
            random_groups = random.sample([g for g in groups if g != group], NUM_TO_SELECT)
            external_group_links += [(report, random.choice(groups[g]), 1.0, 0) for g in random_groups]

        links.extend(internal_group_links)
        links.extend(external_group_links)

    print('Links combination finished: %d' % len(links))
    set_progress(env, 30)

    return _linkage(env, crimes, links)

def _linkage(env, crimes, links):
    # linkage
    print('Start links with distance transformation...')
    linked_rows = []

    progress = 0
    for link in links:
        crime_id1 = link[0]
        crime_id2 = link[1]
        crime1 = crimes[crime_id1]
        crime2 = crimes[crime_id2]
        row = {}
        for feature in FEATURES_TO_USE:
            distance_func = FEATURES_TO_USE[feature][2]
            if distance_func is not None:
                row[feature] = distance_func(crime1[feature], crime2[feature])
        row['weight'] = link[2]
        row['class'] = link[3]
        row['crime_pair'] = '%s+%s' % (crime_id1, crime_id2)
        linked_rows.append(Row(**row))
        progress += 1
        set_progress(env, 30 + progress * 60 / len(links))

    # type schema
    fields = [StructField(feature, FEATURES_TO_USE[feature][3], True) for feature in FEATURES_TO_USE if
              FEATURES_TO_USE[feature][2] is not None]
    fields.append(StructField('weight', DoubleType(), False))
    fields.append(StructField('class', IntegerType(), False))
    fields.append(StructField('crime_pair', StringType(), False))

    df = env['sqlContext'].createDataFrame(linked_rows, schema=StructType(fields))
    attributes = df.columns

    raw_df = _handle_missing(df)
    df = _vector_assembly(raw_df)
    df = _normalize(df)

    _write_arff(attributes, df)

    return {'DataFrame': df, 'RawDataFrame': raw_df}

def _handle_missing(df):
    from pyspark.ml.feature import Imputer
    from pyspark.sql import functions as F

    # handle missing values
    columns = list(filter(lambda col: col not in ('class', 'weight', 'crime_pair'), df.columns))
    dtypes = dict(df.dtypes)

    # for int columns
    int_columns = list(filter(lambda col: dtypes[col] not in ('float', 'double'), columns))
    stats = df.agg(*(
        F.avg(c).alias(c) for c in int_columns
    ))
    fillers = {k: round(v) for k, v in stats.first().asDict().items() if v is not None}
    df = df.na.fill(fillers)

    # for float columns
    float_columns = list(filter(lambda col: dtypes[col] in ('float', 'double'), columns))
    print(float_columns)
    imputer = Imputer(
        inputCols=float_columns,
        outputCols=["{}_imputed".format(c) for c in float_columns]
    )
    df = imputer.fit(df).transform(df)
    df = df.drop(*float_columns)

    return df

def _normalize(df):
    scaler = MinMaxScaler(inputCol="raw_feature_vec", outputCol="feature_vec")
    # withStd=True, withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(df)

    # Normalize each feature to have unit standard deviation.
    df = scalerModel.transform(df)
    return df.select(['feature_vec', 'weight', 'class', 'crime_pair'])

def _vector_assembly(df):
    # vector
    columns = list(filter(lambda col: col not in ('class', 'weight', 'crime_pair'), df.columns))
    # columns = ['time', 'northingEasting']
    assembler = VectorAssembler(inputCols=columns, outputCol='raw_feature_vec')
    df = assembler.transform(df)

    return df.select(['raw_feature_vec', 'weight', 'class', 'crime_pair'])

def _write_arff(attributes, df):
    # write ARFF header
    arff = open("/Users/Chao/reports.arff", "w")
    arff.write("@RELATION reports\n\n")
    for col in attributes:
        if col != 'class':
            arff.write("@ATTRIBUTE %s  NUMERIC\n" % col)
        else:
            arff.write("@ATTRIBUTE class {0,1}\n")
    arff.write("\n@DATA\n")

    # write ARFF data
    def write_line(r):
        l = [str(v) for v in r['feature_vec']]
        l.append(str(r['weight']))
        l.append(str(r['class']))
        arff.write(','.join(l) + "\n")

    for r in df.collect():
        write_line(r)
    arff.close()

def nzpolice_evaluate(env, inputs, settings):
    predictions1 = inputs['DataFrame1']
    predictions2 = inputs['DataFrame2']
    model1 = inputs['Model1'] # type: pyspark.ml.Model
    model2 = inputs['Model2']

    import plotly.graph_objs as go


    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(labelCol="class", metricName="areaUnderROC")
    metric1 = evaluator.evaluate(predictions1)
    metric2 = evaluator.evaluate(predictions2)

    print('train metrics: %f, test metrics: %f %f' % (model1.summary.areaUnderROC, metric1, metric2))

    roc1 = model1.summary.roc.toPandas()
    # f1 = model1.summary.fMeasureByThreshold.toPandas()

    # roc2 = model2.summary.roc.toPandas()
    trace_roc1 = go.Scatter(
        x=roc1['FPR'],
        y=roc1['TPR'],
        name='ROC',
        mode='lines'
    )
    # trace_f1 = go.Scatter(
    #     x=f1['threshold'],
    #     y=f1['F-Measure'],
    #     name='F-Measure1',
    #     mode='line'
    # )

    # trace_roc2 = go.Scatter(
    #     x=roc2['TPR'],
    #     y=roc2['FPR'],
    #     name='ROC Curve2',
    #     mode='markers',
    #     marker=dict(
    #         size=10,
    #         color='rgba(255, 90, 100, .9)',
    #         line=dict(
    #             width=2,
    #         )
    #     )
    # )

    data = [trace_roc1, ]#trace_roc2]

    layout = dict(title='LogisticRegression ROC Curve (areaUnderROC=%f)' % metric1,
                  xaxis=dict(title='False Positive Rate'),
                  yaxis=dict(title='True Positive Rate'),
                  )

    figure = go.Figure(data=data, layout=layout)

    return {'Figure': figure}