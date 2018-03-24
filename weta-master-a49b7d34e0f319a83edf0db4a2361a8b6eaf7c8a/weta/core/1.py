from weta.core.weta_lib import *


def pyspark_script_console11(inputs, settings):
    data = inputs.get('data', None)
    df = inputs.get('df', None)
    df1 = inputs.get('df1', None)
    df2 = inputs.get('df2', None)
    df3 = inputs.get('df3', None)
    transformer = inputs.get('transformer', None)
    estimator = inputs.get('estimator', None)
    model = inputs.get('model', None)

    import re
    import datetime
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType, StringType, ArrayType

    def p_ordinalDate(string):
        start = datetime.datetime.strptime(string.strip(), '%d/%m/%Y')
        return start.toordinal()

    def p_time(string):
        hours = int(string.split(":")[0])
        if "PM" in string: hours += 12
        return hours

    def p_entryLocation(string):
        vectors1 = ['PREMISES-REAR', 'PREMISES-FRONT', 'PREMISES-SIDE']
        for x in vectors1:
            if x in string: return x
        return "UNKNOWN"

    def p_entryPoint(string):
        vectors2 = ['POINT OF ENTRY-DOOR', 'POINT OF ENTRY-WINDOW', \
                    'POINT OF ENTRY-FENCE', 'POINT OF ENTRY-DOOR: GARAGE']
        vectors3 = ['POE - DOOR', 'POE - WINDOW', 'POE - FENCE', 'POE - GARAGE']
        for x, y in list(zip(vectors2, vectors3)):
            if x in string or y in string: return x
        return "UNKNOWN"

    def p_dayOfWeek(string):
        start = datetime.datetime.strptime(string, '%d/%m/%Y')
        return start.weekday()

    def p_northingEasting(string, string2):
        return "%s-%s" % (string, string2)

    def p_methodOfEntry(string):
        if string is None:
            return ''

        narrative = string.split("__________________________________ CREATED BY")[-1]
        if 'NARRATIVE' in narrative or 'CIRCUMSTANCES' in narrative:
            narrative = re.split('NARRATIVE|CIRCUMSTANCES', narrative)[-1]
            narrative = re.split("\*|:", narrative[1:])[0]
        return narrative


        # Classifies if the search was messy

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

    import nltk
    from nltk.parse.stanford import StanfordDependencyParser
    import string as string_module

    stemmer = nltk.stem.porter.PorterStemmer()
    parser = StanfordDependencyParser(
        path_to_models_jar='/Users/Chao/nzpolice/summer/stanford-parser/stanford-parser-3.8.0-models.jar',
        model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
        path_to_jar='/Users/Chao/nzpolice/summer/stanford-parser/stanford-parser.jar',
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

    def p_propertyStolenList(string):
        if "PROPERTY" not in string:
            return []
        property_list = " ".join(
            [re.split(':|_', listing)[0] for listing in re.split("PROPERTY LIST SUMMARY:|PROPERTY STOLEN:", string)])
        text = normalize(property_list)
        tagged = unigram_tagger.tag(text)
        removable = ['modus', 'operandi', 'call', 'with', 'list', 'of', 'location', 'point', 'entry', 'value',
                     'property'
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


        # def stem_tokens(tokens):

    # 	return [stemmer.stem(item) for item in tokens]
    #
    #
    # # Normalizes text (i.e, tokenizes and then stems words)
    # def normalize(text):
    # 	if text is None:
    # 		return []
    # 	return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

    udf_ordinal_date = udf(p_ordinalDate, IntegerType())
    udf_time = udf(p_time, IntegerType())
    udf_entry_location = udf(p_entryLocation, StringType())
    udf_entry_point = udf(p_entryPoint, StringType())
    udf_day_of_week = udf(p_dayOfWeek, IntegerType())
    udf_northing_easting = udf(p_northingEasting, StringType())
    udf_method_of_entry = udf(p_methodOfEntry, StringType())  # *
    udf_messy = udf(p_messy, IntegerType())
    udf_signature = udf(p_signature, IntegerType())
    udf_property_secure = udf(p_propertySecure, IntegerType())
    udf_property_stolen_list = udf(p_propertyStolenList, ArrayType(StringType()))
    udf_pull_mo_tags = udf(p_pullMOTags, ArrayType(StringType()))

    # udf_normalize = udf(normalize, ArrayType(StringType()))

    FEATURES_TO_USE = [
        ('ordinalDate', 'Occurrence Start Date', udf_ordinal_date),
        ('time', 'Occurrence Start Time', udf_time),
        ('entryLocation', 'Narrative', udf_entry_location),
        ('entryPoint', 'Narrative', udf_entry_point),
        ('dayOfWeek', 'Occurrence Start Date', udf_day_of_week),
        ('northingEasting', ('NZTM Location Northing', 'NZTM Location Easting'), udf_northing_easting),

        ('methodOfEntry', 'Narrative', udf_method_of_entry),
        ('messy', 'methodOfEntry', udf_messy),
        ('signature', 'Narrative', udf_signature),
        ('propertySecure', 'Narrative', udf_property_secure),
        ('propertyStolenWordnet', 'Narrative', udf_property_stolen_list),
        # ('cosineTFIDF', 'Narrative', udf_method_of_entry),
        # ('cosineTFIDF2', 'Narrative', udf_method_of_entry),
        ('cosineMO', 'methodOfEntry', udf_pull_mo_tags),
        # ('propertyStolenWordNetNA', 'Narrative', udf_property_stolen_list),
        # ('listSimilarity', 'Narrative', udf_property_stolen_list),
        # ('moSim', 'methodOfEntry', udf_pull_mo_tags),
    ]

    df = df.na.fill({'Narrative': ''})
    # df.na.drop(subset=["Narrative"])

    for t in FEATURES_TO_USE:
        new_col = t[0]
        func = t[2]
        in_cols = t[1]
        params = (df[c] for c in t[1]) if isinstance(in_cols, tuple) else [df[in_cols]]
        df = df.withColumn(new_col, func(*params))

    return {'data': data, 'df': df, 'df1': df1, 'df2': df2, 'df3': df3, 'transformer': transformer,
            'estimator': estimator, 'model': model}


def pyspark_script_console12(inputs, settings):
    data = inputs.get('data', None)
    df = inputs.get('df', None)
    df1 = inputs.get('df1', None)
    df2 = inputs.get('df2', None)
    df3 = inputs.get('df3', None)
    transformer = inputs.get('transformer', None)
    estimator = inputs.get('estimator', None)
    model = inputs.get('model', None)

    select_cols = ['_id', 'ordinalDate', 'time', 'entryLocation', 'entryPoint', 'dayOfWeek', 'northingEasting', 'messy',
                   'signature', 'propertySecure', 'propertyStolenWordnet', 'cosineTF', 'cosineIDF']

    df = df.select(*select_cols)

    df1 = df1.select('_id', 'cosineTF2', 'cosineIDF2')

    df2 = df2.select('_id', 'cosineMOTF', 'cosineMOIDF')

    df = df.join(df1.join(df2, ['_id']), ['_id'])

    return {'data': data, 'df': df, 'df1': df1, 'df2': df2, 'df3': df3, 'transformer': transformer,
            'estimator': estimator, 'model': model}


# Data Reader
inputs0 = {}
settings0 = {'file_path': '/Users/Chao/nzpolice/summer/newreports1000.csv', 'format': 'csv'}
outputs0 = dataframe_reader(inputs0, settings0)

# PySpark Script (1)
inputs11 = {'df': outputs0['DataFrame']}
settings11 = {}
outputs11 = pyspark_script_console11(inputs11, settings11)

# NLTK Tokenizer
inputs2 = {'DataFrame': outputs11['df']}
settings2 = {'inputCol': 'methodOfEntry', 'outputCol': 'methodOfEntry_tokens'}
outputs2 = spark_nltk_tokenizer(inputs2, settings2)

# Stopwords Remover
inputs3 = {'DataFrame': outputs2['DataFrame']}
settings3 = {'caseSensitive': False, 'inputCol': 'methodOfEntry_tokens', 'outputCol': 'methodOfEntry_cleaned_tokens'}
outputs3 = spark_stopwords_remover(inputs3, settings3)

# Hashing TF
inputs8 = {'DataFrame': outputs3['DataFrame']}
settings8 = {'binary': False, 'inputCol': 'methodOfEntry_cleaned_tokens', 'numFeatures': 262144,
             'outputCol': 'cosineTF'}
outputs8 = spark_hashing_tf(inputs8, settings8)

# IDF
inputs1 = {'DataFrame': outputs8['DataFrame']}
settings1 = {'inputCol': 'cosineTF', 'minDocFreq': 0, 'outputCol': 'cosineIDF'}
outputs1 = spark_idf(inputs1, settings1)

# NGram
inputs4 = {'DataFrame': outputs3['DataFrame']}
settings4 = {'inputCol': 'methodOfEntry_cleaned_tokens', 'n': 2, 'outputCol': 'twogram'}
outputs4 = spark_ngram(inputs4, settings4)

# Hashing TF (1)
inputs9 = {'DataFrame': outputs4['DataFrame']}
settings9 = {'binary': False, 'inputCol': 'twogram', 'numFeatures': 262144, 'outputCol': 'cosineTF2'}
outputs9 = spark_hashing_tf(inputs9, settings9)

# IDF (1)
inputs5 = {'DataFrame': outputs9['DataFrame']}
settings5 = {'inputCol': 'cosineTF2', 'minDocFreq': 0, 'outputCol': 'cosineIDF2'}
outputs5 = spark_idf(inputs5, settings5)

# Hashing TF (2)
inputs10 = {}
settings10 = {'binary': False, 'inputCol': 'cosineMO', 'numFeatures': 262144, 'outputCol': 'cosineMOTF'}
outputs10 = spark_hashing_tf(inputs10, settings10)

# IDF (2)
inputs7 = {'DataFrame': outputs10['DataFrame']}
settings7 = {'inputCol': 'cosineMOTF', 'minDocFreq': 0, 'outputCol': 'cosineMOIDF'}
outputs7 = spark_idf(inputs7, settings7)

# PySpark Script (2)
inputs12 = {'df': outputs1['DataFrame'], 'df1': outputs5['DataFrame'], 'df2': outputs7['DataFrame']}
settings12 = {}
outputs12 = pyspark_script_console12(inputs12, settings12)

# Data Viewer
inputs6 = {'DataFrame': outputs12['df']}
settings6 = {}
outputs6 = dataframe_viewer(inputs6, settings6)