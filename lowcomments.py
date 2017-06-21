#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 19:28:27 2017

@author: catherineordun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:38:55 2017

@author: vacoordunc
"""

from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk.collocations
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import textblob
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from tabulate import tabulate
import warnings
import os
from colorama import init
from colorama import Fore, Back, Style

init()
basedir = "/Users/catherineordun/Documents/data/experiments/"
warnings.filterwarnings("ignore")


print("""
      

 /\  /(_)   / __\ __(_) ___ _ __   __| |
 / /_/ / |  / _\| '__| |/ _ \ '_ \ / _` |
/ __  /| | / /  | |  | |  __/ | | | (_| |
\/ /_/ |_| \/   |_|  |_|\___|_| |_|\__,_|
                                                                                       
                                                                                                                                                 
                                                                                                                                                      
                                                                                                                                                      
""" )

print(Fore.CYAN + "Reading in and formatting time.  This will take a minute...," + Style.RESET_ALL)
filename = raw_input(str('Enter a filename: '))
data = pd.read_csv("{}{}".format(basedir, filename))

print("We currently have" + Fore.RED + str(len(data)) + Style.RESET_ALL + "records.")

data_fin = data[['User', 'Text', 'Score', 'Help_Numer', 'Help_Denom']].copy()

plt.figure(1,figsize=(8,8)) 
data_fin[['Score', 'Help_Numer', 'Help_Denom']].hist(bins=25)
plt.savefig("/Users/catherineordun/Documents/data/experiments/histogram.pdf", bbox_inches='tight')
print(Fore.CYAN + "Saved histogram in your plots folder." + Style.RESET_ALL)

print("The mean Score for the first 500 users is:" + Fore.RED + str(data_fin['Score'].mean()) + Style.RESET_ALL)
print("The mean Helpfulness Numerator for the first 500 users is:" + Fore.RED + str(data_fin['Help_Numer'].mean()) +Style.RESET_ALL)
print("The mean Helpfulness Denominator for the first 500 users is:" + Fore.RED + str(data_fin['Help_Denom'].mean()) + Style.RESET_ALL)

plt.figure(1,figsize=(6,6)) 
x = data_fin[['Score', 'Help_Numer', 'Help_Denom']].mean()
x.plot(marker="o")
plt.savefig('/Users/catherineordun/Documents/data/experiments/averages.pdf', bbox_inches='tight')
print(Fore.CYAN + "Saved plot with average scores for each metric in the plots folder."+ Style.RESET_ALL)


data_fin['low'] = np.where(((data_fin['Score'] == 1) & (data_fin['Help_Numer'] ==1) & (data_fin['Help_Denom'] ==1)), 1, 0)
low_names = data_fin.loc[(data_fin['low'] == 1)]
print("There are" + Fore.CYAN + str(len(low_names)) + Style.RESET_ALL + "users where scores and helpfulness indicators are 1.")
print(low_names['User'])

#Trigrams per person
#tokenize the corpus
print(Fore.CYAN + "Please wait this may take a minute, cleaning text data." + Style.RESET_ALL)
low_names.fillna(value=0, inplace=True)
low_text = low_names.loc[(low_names['Text'] !=0)]
low_text['text_cln'] = low_text['Text'].map(lambda x: BeautifulSoup(x, "lxml").get_text())
tokenizer = RegexpTokenizer(r'\w+')
low_text['tokens'] = low_text['text_cln'].map(lambda x: tokenizer.tokenize(x))
#remove stopwords
cachedstopwords = stopwords.words('english')
stw_set = set(cachedstopwords)
low_text['tokens_cln'] = low_text.apply(lambda row: [item for item in row['tokens'] if item not in stw_set], axis=1)
#for each station, put trigrams in a separate column
tgm = nltk.collocations.TrigramAssocMeasures()
low_text['tgms'] = low_text['tokens_cln'].map(lambda x: (nltk.collocations.TrigramCollocationFinder.from_words(x)).score_ngrams(tgm.pmi))

request = str(raw_input("Wanna read comments? Enter the user's name exactly as displayed above. Copy and paste it in here."))
print(Fore.YELLOW + str(list(low_text['text_cln'].loc[(low_text['User'] ==request)])) + Style.RESET_ALL)

while True:
    try:
        lookupanother = str(raw_input("Would you like to look up another name? Enter 'yes' or 'no'"))
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    if lookupanother == 'yes':
        request = str(raw_input("To show you the comments made about a low performing staff member, enter the person's name exactly as displayed above. Copy and paste it in here."))
        print(Fore.YELLOW + str(list(low_text['text_cln'].loc[(low_text['User'] ==request)])) + Style.RESET_ALL)
        continue
    elif lookupanother == 'no':
        print(Fore.CYAN + "OK, moving on.") 
        break


print(Fore.CYAN + "Now let's see the Top 100 negative phrases." + Style.RESET_ALL)
low_corpus = (low_text.reset_index()).text_cln.sum()
tokenizer = RegexpTokenizer(r'\w+')
low_tokens = tokenizer.tokenize(low_corpus)
stw_set = set(stopwords.words('english'))
low_filtered = [word for word in low_tokens if word not in stw_set]
V = set(low_filtered)
finder = nltk.collocations.TrigramCollocationFinder.from_words(low_filtered)
phrases = pd.DataFrame(finder.score_ngrams(tgm.pmi))
print(phrases[:100])

print(Fore.CYAN + "Sentiment analysis on comments from low performing staff. Warning: Sentiment analysis algorithm can sometimes have errors. It is important to always check the comment using the tool above."+ 
      Style.RESET_ALL)
low_text['sent']= low_text['text_cln'].map(lambda x: TextBlob(x).sentiment.polarity)
print(tabulate(low_text[['sent', 'User']].sort_values(by='sent', ascending=False), headers='keys', tablefmt='psql'))

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}
plt.rc('font', **font)
plt.rcParams.update({'font.size': 5})
#plot the first 100 users
(low_text[['sent', 'User']].sort_values(by='sent', ascending=False))[:100].plot(x='User', kind='bar',figsize=(25,8))
plt.savefig("/Users/catherineordun/Documents/data/experiments/sentimentlowscorers.pdf", bbox_inches='tight')
print(Fore.CYAN + "Saved plot distribution of staff by their sentiment score.")

