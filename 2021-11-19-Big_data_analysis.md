---
layout: single
title: "Big Data Analysis"
date: 2021-11-19 12:00:00 -0000
categories: Big_Data Spark NLP   
excerpt: Perform data analysis and machine learning on large messy data sets.
---

## Summary
In this project have parsed, cleaned, and processed a 10 GB set of XML files of user actions on the Stack Overflow website.
By performing SQL-like queries on Spark RDDs and DataFrames, we have answered questions about user behavior to predict the long-term behavior of new users. I have trained a word2vec model and a classification model on tags associated with questions. These machine learning pipelines were implemented using Spark ML.


## Data format & parsing
The anonymized data are download from aws s3 bucket with the following format:

```xml
<row Body="&lt;p&gt;Have you considered using the Joachims\' &lt;a href=&quot;http://svmlight.joachims.org/svm_multiclass.html&quot;   
           rel=&quot;nofollow&quot;&gt;SVM Light\'s MultiClass classifier&lt;/a&gt;? &#10;&lt;a href=&quot;http://svmlight.joachims.   
           org/svm_multiclass.html&quot;rel=&quot;nofollow&quot;&gt;http://svmlight.joachims.org/svm_multiclass.html&lt;/a&gt;&lt;    
           /p&gt;&#10;"CommentCount="0" CreationDate="2012-08-02T19:15:04.647" Id="33580" LastActivityDate="2012-08-02T19:15:04.647".  
            OwnerUserId="12060" ParentId="21465" PostTypeId="2" Score="1" /> 
```
we used `lxml.etree` to parse XML files

```python
from lxml import etree
file_posts_stats = "spark-stats-data/allPosts/*"

def get_posts_data(fpath):
    rdd_xmls = sc.textFile(fpath)\
             .filter(lambda row: '<row' in row) \
             .map(parse_row)\
             .filter(lambda x: x is not None)

    # rdd of extracted elements
    return rdd_clean_xmls.map(lambda el: get_attributes(el, atrrib_keys))
```

## Data analysis
By performing SQL-like queries on Spark DataFrames, we can answer following questions about user behaviours to predict the *long-term* behaviour of new users

<ul> 
  <li>Relationship between the number of times a post was favorited (the `FavoriteCount`) and the `Score`.</li>
  <li>Correlation between a user's reputation and the kind of posts they make.</li>    
  <li>Identify "veterans" (user to remain active on the site over a long period of time) and compare their characterstics with  "brief   
      users".</li>  
</ul>


## Building ML model
We'd intend to predict the tags of a question from its body text. We tackled this problem in three phases:
1. Find the ten most common tags for questions in the training data set (the tags have been removed from the test set). 
2. Then train a learner to predict from the text of the question (the Body attribute) if it should have one of those ten tags in it - additional NLP techniques were used to process the question text.
3. Finally, we built a `pipeline` that takes in multiple *stages* of `Transformer` and `Estimator`. The hyperparameters tuning is done by creating a cross validator object that takes the `pipeline` as the estimator and its parameters. 

```python
paramGrid = (ParamGridBuilder() 
    .addGrid(hashingTF.numFeatures, [2000, 3000, 5000]) 
    .addGrid(logreg.regParam, [0.1, 0.01, 0.5])
    .addGrid(logreg.elasticNetParam, [0.1, 0.5, 0.8])         
    .build())


crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=6, # 3 or 4 ?
                          seed=17)

cvModel = crossval.fit(train)
predictions = cvModel.transform(test)
```
