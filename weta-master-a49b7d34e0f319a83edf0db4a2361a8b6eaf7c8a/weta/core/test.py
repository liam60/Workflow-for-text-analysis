import math
from Orange.widgets.widget import OWWidget
from pyspark.sql.functions import desc
from pyspark.sql.functions import col
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import NGram

#Finds the frequency of a given corpus
#returns a dataframe containing two columns
#first column contains word or phrase, second column contains count of that word or phrase.
def frequencyCorpus(df, inputCol):
    column = df.select(inputCol)
    keys = column.rdd.flatMap(lambda row: [(w,1) for w in row[inputCol]])
    corpus = keys.reduceByKey(lambda a,b: a+b)
    corpus = corpus.toDF(['key','count'])
    return corpus.sort(desc('count'))

#Returns the given dataframe with a column contain n-grams of the text
def ngram(df, _n, _inputCol):
    ngram = NGram(n=_n, inputCol=_inputCol, outputCol="ngrams")
    bigrams = ngram.transform(df)
    return bigrams


def collo(df,inputCol):

    #Returns the probablity of a term, given the count of the term and total terms in document
    #Returns int
    def prob(count, total):
        return (int(count)/int(total))

    #Returns the tscore of a given bigram
    def tScore(bigram, w1, w2, totalBigrams, totalWords):
        y = prob(bigram, totalBigrams)
        x = prob(w1,totalWords) * prob(w2,totalWords)
        t = (y-x)/(math.sqrt(y/totalBigrams))
        #t = round(t,1)
        return t

    #returns the chi score of a given bigram
    def chiProb(bCount, w1, w2, totalBigrams, totalWords):
        #x^2 = (N(O11 * O22 - O12 * O21)^2) / (O11 + O12)(O11 + O21)(O12 + O22)(O21 + O22)
        O11 = bCount
        O12 = w1 - O11
        O21 = w2 - O11
        O22 = totalBigrams - O11 - O12 - O21
        xSqr = (totalWords*(math.pow(((O11*O22)-(O12*O21)),2)))
        xSqr = xSqr/((O11 + O12)*(O11 + O21)*(O12 + O22)*(O21 + O22))
        return xSqr

    set_progress(20)

    #gets the frequency corpus for both the documents 2-grams and single words
    freqBiGrams = frequencyCorpus(ngram(df, 2, inputCol), 'ngrams')
    freqMap = frequencyCorpus(df, inputCol)
    set_progress(30)

    #Splits each column so we have a better formatted dataframe
    #format is not [bigram, bigram count, 1st word from bigram, 2nd word from bigram]
    result = freqBiGrams.rdd.map(lambda x: (x[0],
                                            x[1],
                                            x[0].split()[0],
                                            x[0].split()[1]))
    result = result.toDF(['Bigram','bCount','word1','word2'])
    set_progress(40)

    #Joins the bigram dataframe word1 and word2 column with each words corresponding frequency
    result = result.join(freqMap, result.word1 == freqMap.key, 'inner').select(col("Bigram"), col("bCount"), col("count").alias('c1'), col("word2"))
    result = result.join(freqMap, result.word2 == freqMap.key, 'inner').select(col("Bigram"), col("bCount"), col("c1"), col("count").alias('c2'))
    set_progress(50)

    #gets the total number of bigrams by adding each count.
    global BiCount
    BiCount = freqBiGrams.rdd.map(lambda x: float(x["count"])).reduce(lambda x, y: x+y)
    set_progress(55)
    #gets the total number of tokens by added all counts.
    OneCount = freqMap.rdd.map(lambda x: float(x["count"])).reduce(lambda x, y: x+y)
    set_progress(60)
    #gets the t and chi score for each row using the data in each column and the total counts.
    df = result.rdd.map(lambda x, btotal=BiCount, total=OneCount: (x[0],
                            x[1],
                            tScore(x[1],x[2],x[3],btotal,total),
                            chiProb(x[1], x[2], x[3], btotal, total)))
    set_progress(80)
    #Returns the completed dataframe with the named columns
    return df.toDF(['Bigram','Count','tScore','chiScore'])

def test_collo(env, inputs, settings):
    global set_progress
    def set_progress(percentage):
        if env['ui'] is not None:
            widget: OWWidget = env['ui']
            widget.progressBarSet(percentage)
    set_progress(0)

    #Retreives the input parameters from the UI
    df = inputs['DataFrame']
    cutoff = settings['cutoff']
    inputCol = settings['inputCol']
    sortType = settings['sortingType']
    sqlContext = env['sqlContext']
    set_progress(10)

    df = collo(df,inputCol)

    #Sorts the dataframe by selected score
    if sortType == 0:
        df = df.sort(desc('Count'))
    elif sortType == 1:
        df = df.sort(desc('chiScore'))
    elif sortType == 2:
        df = df.sort(desc('tScore'))
    elif sortType == 3:
        df = df.sort(desc('chiScore'))
        #to remove chi scores where word1, word2 and bigram count = 1
        df = df.filter(df.chiScore < BiCount)

    set_progress(90)

    #only displays the top number of rows required
    df = df.limit(cutoff)

    set_progress(100)
    #returns the completed dataframe
    return {'DataFrame': df}