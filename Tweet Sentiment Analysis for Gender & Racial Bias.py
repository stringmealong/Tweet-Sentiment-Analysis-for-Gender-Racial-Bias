#!/usr/bin/env python
# coding: utf-8

# # Tweet Sentiment Analysis for Gender & Racial Bias

# ### Importing libraries 

# In[69]:


from pyspark.sql import SparkSession
from pyspark.ml import feature, regression, evaluation, Pipeline
from pyspark.sql import functions as fn, Row
import matplotlib.pyplot as plt
import pandas as pd 
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


# In[70]:


import numpy as np


# In[71]:


# dataframe functions
from pyspark.sql import functions as fn
from __future__ import division


# In[72]:


from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer
from pyspark.ml import Pipeline


# ### Loading in data

# In[73]:


get_ipython().system('head -5 ..//00-Project/initialDataset.csv')


# **i tried to upload the file with Spark, without succces**

# In[74]:


#not working
# StructType to define the schema 
#from pyspark.sql.types import StructType, StructField,StringType,IntegerType,FloatType,LongType


# In[75]:


#not working
# Create schema  dataset
'''schema = StructType([StructField("tweet_id",StringType(),True),
                     StructField("text",StringType(),True),
                    StructField("place",StringType(),True),
                    StructField(" polarity",FloatType(),True)])'''


# In[76]:


#tweets_df = spark.read.schema(schema).option('sep',',').csv('..//00-Project/initialDataset.csv')
#tweets_df.show(5)


# #### upload the file using `pandas`  *read_csv*

# In[77]:


tweets_df=pd.read_csv('initialDataset.csv')
tweets_df.head()


# In[78]:


#tweets_df.withColumnRenamed("polarity","label").printSchema()


# #### creating the dataframe 

# In[79]:


tweets_df=spark.createDataFrame(tweets_df)
tweets_df.show(5)


# In[80]:


tweets_df.printSchema()


# In[81]:


tweets_df.count()


# #### top 10 values in `polarity` column

# In[82]:


from pyspark.sql.functions import col
tweets_df.groupBy("polarity")     .count()     .orderBy(col("count").desc())     .show(10)


# #### top 10 values in `place` column

# In[83]:


from pyspark.sql.functions import col
tweets_df.groupBy("place")     .count()     .orderBy(col("count").desc())     .show(10)


# #### Remove the columns we do not need and have a look the first five rows

# In[84]:


#tweets_text_df = tweets_df.drop("tweet_id", "place", "polarity")
tweets_text_df = tweets_df.drop("tweet_id", "place")


# In[85]:


tweets_text_df.show(5)


# In[86]:


tweets_text_df.show(5)


# #### dropping the rows that contain any null or NaN values 
# 
# https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35

# In[87]:


tweets_text_df = tweets_text_df.na.drop()


# In[88]:


tweets_text_df.count()


# In[89]:


# the tokenizer object
tokenizer = Tokenizer().setInputCol('text').setOutputCol('words')


# In[90]:


#tweet_tokenizer= tokenizer.transform(tweets_text_df).show()


# In[91]:


count_vectorizer_estimator = CountVectorizer().setInputCol('words').setOutputCol('features1') ##features1 changed right now so that we can use tfidf as new "features" for experiment


# In[92]:


count_vectorizer_transformer = count_vectorizer_estimator.fit(tokenizer.transform(tweets_text_df))


# In[93]:


#count_vectorizer_transformer.transform(tokenizer.transform(tweets_text_df)).show(truncate= False)


# In[94]:


# list of words in the vocabulary
count_vectorizer_transformer.vocabulary


# #### Removing stop words

# In[95]:


# changed setsetInputCol("filtered") - to features
import requests
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()
stop_words[0:10]
from pyspark.ml.feature import StopWordsRemover
sw_filter = StopWordsRemover()  .setStopWords(stop_words)  .setCaseSensitive(False)  .setInputCol("words")  .setOutputCol("features1") #features1 changed right now so that we can use tfidf as new "features" for experiment


# Defining min document frequency

# In[96]:


# changed setsetInputCol("filtered") - to features
# we will remove words that appear in 5 docs or less
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)  .setInputCol("features1").setOutputCol("tf")
##features1 changed right now so that we can use tfidf as new "features" for experiment


# #### Pipeline to tokenize, remove stopwords and account for min document frequency 

# In[97]:


cv_pipeline = Pipeline(stages=[tokenizer, sw_filter, cv]).fit(tweets_text_df)


# In[98]:


# now we can make the transformation between the raw text and the counts
cv_pipeline.transform(tweets_text_df).show(5)


# In[99]:


len(cv_pipeline.stages[-1].vocabulary)


# #### Pipeline with Tfidf tokenizer

# In[100]:


from pyspark.ml.feature import IDF
idf = IDF().    setInputCol('tf').    setOutputCol('features') #changed tfidf to features


# In[101]:


idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(tweets_text_df)


# In[102]:


idf_pipeline.transform(tweets_text_df).show(5)


# In[103]:


tfidf_df = idf_pipeline.transform(tweets_text_df)
tfidf_df.limit(5).toPandas()


# In[ ]:





# # Logistic regression

# #### training, validation, and test set split

# In[104]:


from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


# In[105]:


(train_set, val_set, test_set) = tfidf_df.randomSplit([0.98, 0.01, 0.01], seed = 2000)


# In[106]:


train_set.show(10)


# In[107]:


train_set = train_set.withColumnRenamed("polarity","label")


# In[ ]:





# In[108]:


train_set.show(5)


# In[109]:


val_set.show()


# In[110]:


val_set = val_set.withColumnRenamed("polarity","label")


# In[111]:


val_set.show()


# In[112]:


######### round off

from pyspark.sql.functions import round, col

train_set = train_set.select("*", round(col('label'),1))


train_set.show()


# In[113]:


#val_set.drop('label').collect()

#val_set = train_set.withColumnRenamed("label","labelold")
train_set = train_set.withColumnRenamed("label","labelold")


# In[114]:


train_set = train_set.withColumnRenamed("round(label, 1)","label")


# In[115]:


train_set.show(10)


# In[116]:


#######MINMAXSCALER#########
#issue is we need integer values and this automatically scales in range from [0,1]


# In[117]:


#from pyspark.ml.feature import MinMaxScaler


# In[118]:


#scaler = MinMaxScaler(inputCol="features", outputCol="scaledLabel")


# In[119]:


#Compute summary statistics and generate MinMaxScalerModel
#scalerModel = scaler.fit(train_set)


# In[120]:


#train_set_lr = scalerModel.transform(train_set)


# In[121]:


#train_set_lr.show(10)


# In[122]:


#train_set_lr = train_set_lr.withColumnRenamed("label","labelold2")


# In[123]:


#train_set_lr = train_set_lr.withColumnRenamed("scaledLabel","label")


# In[124]:


#train_set_lr.show(10)


# In[125]:


######multiply by 10
#train_set_lr.withColumn('label', 
     #(col('label') * 10))


# In[126]:


#train_set_lr.show(20)


# In[ ]:





# In[127]:


####BINARIZER####


# In[128]:


from pyspark.ml.feature import Binarizer


# In[129]:


binarizer = Binarizer(threshold=0.05, inputCol="label", outputCol="binarized_label")


# In[130]:


train_set_lr = binarizer.transform(train_set)


# In[131]:


print("Binarizer output with Threshold = %f" % binarizer.getThreshold())


# In[132]:


train_set_lr.show(10)


# In[133]:


train_set_lr = train_set_lr.withColumnRenamed("label","labelold")


# In[134]:


train_set_lr = train_set_lr.withColumnRenamed("binarized_label","label")


# In[135]:


train_set_lr.show(10)


# In[136]:


#####END OF BINARIZER#########


# In[137]:


#val_set = train_set.withColumnRenamed("text","label1")


# In[138]:


#val_set.show()


# In[139]:


#test_set.show(3)
#type(test_set['features'])
#int(test_set['features'])


# In[140]:


#val_set = test_set.withColumnRenamed("polarity","label")


# In[141]:


#import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
#import org.apache.spark.mllib.evaluation.MulticlassMetrics
#new LogisticRegressionWithLBFGS().setNumClasses(10)


# In[142]:


from pyspark.ml.classification import LogisticRegression


# In[143]:


lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_set_lr)
lrModel = lr.fit(train_set_lr)

predictions = lrModel.transform(test_set)


# In[144]:


predictions.show()


# ## Random Forest 

# In[145]:


from pyspark.ml.classification import RandomForestClassifier


# In[146]:


#from pyspark.ml.feature import MinMaxScaler


# In[147]:


#scaler = MinMaxScaler(min=0, max=2147483647, inputCol='label', outputCol='label_minmax')


# In[148]:


#scaler_model = scaler.fit(train_set)
#train_set_rf = scaler_model.transform(train_set_rf)
#train_set_rf.show(5)


# In[149]:


#val_set = val_set.withColumnRenamed("label","labelold")
#val_set = val_set.withColumnRenamed("round(label, 1)","label")
#val_set.show()


# In[150]:


test_set = test_set.withColumnRenamed("polarity","label")


# In[151]:


test_set.show(5)


# In[152]:


rf = RandomForestClassifier(labelCol="label",                             featuresCol="features",                             numTrees = 100,                             maxDepth = 4,                             maxBins = 32)
# Train model with train_set
rfModel = rf.fit(train_set_lr)
predictions = rfModel.transform(test_set)
predictions.filter(predictions['prediction'] == 0)     .select("features","probability","label","prediction")     .orderBy("probability", ascending=False)     .show(n = 10, truncate = 30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




