# -*- coding: utf-8 -*-

from __future__ import print_function

import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.cross_validation import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from scipy.stats import sem

#import data 
categories = ['Env', 'Fac', 'Med', 'Saf']
arims_data = sklearn.datasets.load_files('/Users/catherineordun/Documents/ARIMS/arimsdata', categories=categories, load_content=True, shuffle=True, encoding='utf-8', decode_error='ignore', random_state=42)
X, y = arims_data.data, arims_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Function to perform and evaluate a cross validation
def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score:{0:.3f}(+/-{1:.3f})".format(np.mean(scores),sem(scores)))

#Creating multiple models through pipelines
clf_1 = Pipeline([
    ('vect', CountVectorizer(decode_error='ignore', strip_accents='unicode', stop_words='english')),
    ('clf', MultinomialNB()),
])

clf_2 = Pipeline([
    ('vect', HashingVectorizer(decode_error='ignore', strip_accents='unicode', stop_words='english', non_negative=True)),
    ('clf', MultinomialNB()),
])

clf_3 = Pipeline([
    ('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode', stop_words='english')),
    ('clf', MultinomialNB()),
])

#Evaluate each model using the K-fold cross-validation with 5 folds
clfs = [clf_1, clf_2, clf_3]
for clf in clfs:
    evaluate_cross_validation(clf, arims_data.data, arims_data.target, 5)

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names = arims_data.target_names))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

#I chose to use model clf_3
train_and_evaluate(clf_3, X_train, X_test, y_train, y_test)

#This function generates all the features for clf_3
feature_names = clf_3.named_steps['vect'].get_feature_names()

#Returns the most informative top 20 features for each category
def most_informative(num):
    classnum = num
    inds = np.argsort(clf_3.named_steps['clf'].coef_[classnum, :])[-20:]
    print("The top 20 most informative words for category:",classnum)    
    for i in inds: 
        f = feature_names[i]
        c = clf_3.named_steps['clf'].coef_[classnum, [i]]
        print(f,c)
                
most_informative(3)
most_informative(2)
most_informative(1)
most_informative(0)
