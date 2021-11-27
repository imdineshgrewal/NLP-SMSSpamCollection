# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:57:31 2021

@author: imdineshgrewal@gmail.com
"""

#SpamClassifier

#import the Dataset
import pandas as pd
 
messages = pd.read_csv('dataset/smsspamcollection/SMSSpamCollection', sep='\t',
                       names=["label", "message"])

#clean data and preprocessing 
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

ps = PorterStemmer()
lm = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    #Stemming 
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    
    #lemmatization
    #review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    
    review = ' '.join(review)
    corpus.append(review)
    
# creating the bag of words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features=5000)

# Creating the TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()

x = cv.fit_transform(corpus).toarray()
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


#test train split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state = 0)


# Training model using Navie bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)


#Random forest
#rf = RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=1)
#spam_detect_model = rf.fit(x_train,y_train)
# Now, let's see the predictions. I would be using predict function and calculating Precision, Recall , f- score, and Accuracy measure also.

y_pred = spam_detect_model.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

#Accuracy and ROC curve
from sklearn.metrics import accuracy_score, plot_roc_curve
#print('Accuracy :', accuracy_score(y_test, y_pred) * 100)

plot_roc_curve(spam_detect_model, x_test, y_test)

precision,recall,fscore,support =score(y_test,y_pred,pos_label=1, average ='binary')
print('Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision,3),round(recall,3),round(fscore,3),round((y_pred==y_test).sum()/len(y_test) * 100,8)))
