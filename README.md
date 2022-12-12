# South-African-Language-Identification-Hack-2022

South Africa is a multicultural society that is characterised by its rich linguistic diversity. Language is an indispensable
tool that can be used to deepen democracy and also contribute to the social, cultural, intellectual, economic and
political life of the South African society. With such a multilingual population, it is only obvious that our systems and
devices also communicate in multi-languages.

In this challenge,we will text which is in any of South Africa's 11 Official languages and identify which
language the text is in.

# 1. Dataset Description

* `train_set.csv` - the training set

* `test_set.csv` - the test set

### Language IDs

* `afr` - Afrikaans

* `eng` - English

* `nbl` - isiNdebele

* `nso` - Sepedi

* `sot` - Sesotho

* `ssw` - siSwati

* `tsn` - Setswana

* `tso` - Xitsonga

* `ven` - Tshivenda

* `xho` - isiXhosa

* `zul` - isiZulu

# 2. Import Packages

```
import string 
import re
import codecs
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import itertools
```

# 3. Loading the dataset

Using the pandas library we were able to read the dataset of the respective languages

# 4. Feature Engineering

## 4.1 Text Cleaning

Removing noise (i.e. unneccesary information) is a key part of getting the data into a usable format.  For this dataset, we will be carrying out the following cleaning techniques:

* removing the web urls(if any)

* converting all text into lowercase

* removing punctuation marks

* removing stopwords from tweets

## 4.2 Label Encoding
Our output variable, `lang_id` is a categorical variable. For training the model we should have to convert it into a numerical form, so we are performing label encoding on that output variable. For this process, we are importing LabelEncoder from sklearn

## 4.3 Bag of Words
The output feature and also the input feature should be of the numerical form. So we are converting text into numerical form by creating a Bag of Words model using `CountVectorizer`.

## 4.4 Train Test Splitting
The next step is to create the training set, for training the model and test set, for evaluating the test set. For this process, we are using a train test split.

# 5. Modelling
we will be using the naive_bayes algorithm for our model creation. 

