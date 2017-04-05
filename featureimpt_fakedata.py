#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:26:43 2017
Generating fake dataset of employee names, answers to 50 survey questions
from 1 to 5, and fake end-of-year bonuses

Feature Importance using GINI and Extra Trees Classifier
@author: catherineordun
"""
from sklearn import preprocessing
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from faker import Factory
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from tabulate import tabulate


#Generated Fake Names
fake = Factory.create()
fake.name()

#Print names to a list of 200 fake people
names=[]
for _ in range(0, 200):
  print (fake.name())
  names.append(fake.name())

#Create a fake dataframe of 30 questions with responses from 0 to 3
columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
           'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 
           'q20', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27', 'q28', 
           'q29', 'q30', 'q31', 'q32', 'q33', 'q34', 'q35', 'q36', 'q37', 
           'q38', 'q39', 'q40', 'q41', 'q42', 'q43', 'q44', 'q45', 'q46',
           'q47', 'q48', 'q49', 'q50']
df = pd.DataFrame(np.random.uniform(1,5,size=(200, 50)), columns=columns)
df['names'] = names
df.set_index('names', inplace=True)
#predictor variable, A is bonuses
df2 = pd.DataFrame(np.random.randint(1000,5000,size=(200, 1)), columns=list('A'))
df['Bonus'] = df2['A'].values
print(df.head())

#features and target
X =  (df.drop(['Bonus'], axis=1)).values
y = (df['Bonus'].values)

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

#frame of feature names and feature importances
names = ((df.drop(['Bonus'], axis=1)).columns).tolist()
flist = (model.feature_importances_).tolist()
feature_list = pd.DataFrame(
    {'names': names,
     'feature_impt': flist})

#sort most important features in descending order
print(tabulate(feature_list.sort_values(by='feature_impt', ascending=False), headers='keys', tablefmt='psql'))

fframe = pd.DataFrame(feature_list)

# SNS DotPlot Make the PairGrid
"""
Ref: http://seaborn.pydata.org/examples/pairgrid_dotplot.html

"""
sns.set_context("notebook", font_scale=1.5)
sns.set(style="whitegrid")
g = sns.PairGrid(fframe,
                 x_vars=["feature_impt"], y_vars=["names"],
                 size=15, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="Reds_r", edgecolor="gray")

sns.despine(left=True, bottom=True)
















