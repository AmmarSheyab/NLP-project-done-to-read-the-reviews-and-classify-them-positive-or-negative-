# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:30:10 2022

@author: ASUS
"""

import numpy as np
import pandas as pd
 
import pyarabic.araby as ar

df=pd.read_csv('Dev.csv',usecols=(['sent2','label']))

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re ,string , functools, operator
from nltk.stem.porter import PorterStemmer
import pyarabic.araby as ar

stemmer = nltk.ISRIStemmer()#stem arabic in python
'''
unique = [j.split() for j in df['sent2']]
unique = pd.DataFrame(unique)
print(unique.nunique().sum())
'''
df['char_count'] = df['sent2'].apply(lambda x: len(x.split()))
print('count word befor claen',df['char_count'].sum())

#stopword=open('stopword egypt.txt','r',encoding=('utf-8')).read()# read stopsword in text
corpus=[]
for i in range(0,len(df)):
    
 review=df['sent2'][i] 

 review = re.sub(r'^https?:\/\/.*[\r\n]*', '', review, flags=re.MULTILINE)
 review= re.sub(r'^http?:\/\/.*[\r\n]*', '', review, flags=re.MULTILINE)
 review = re.sub(r"http\S+", "", review)
 review = re.sub(r"https\S+", "", review)
 review = re.sub(r'\s+', ' ',review)
 review = re.sub("(\s\d+)","",review) 
 review = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", review)
 review = re.sub("\d+", " ", review)
 review= ar.strip_tashkeel(review)#حذف الحركات كلها بما فيها الشدة
 review = ar.strip_tatweel(review)#حذف التطويل
 review= review.replace("#", " ")
 review = review.replace("@", " ")
 review = review.replace("_", " ")
 translator = str.maketrans('', '', string.punctuation)
 review =review.translate(translator)
 review=re.sub('([@A-Za-z0-9]+)|[^\w\s]|#|http\S+', '',review)
 review = review.lower()
 review = review.split()
 ps = PorterStemmer()
 review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('arabic'))]
 review = ' '.join(review)
 corpus.append(review)

unique = [j.split() for j in corpus]
unique = pd.DataFrame(unique)
print('count word after clean',unique.nunique().sum())
'''
# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features =5000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values
'''
'''
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features =5000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

'''

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features =5000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred)
#print(cm)
print('naive_bayes = 50',accuracy_score(y_test, y_pred))



from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred2=classifier.predict(X_test)
cm2=confusion_matrix(y_test,y_pred2)
#print(cm)
print('DecisionTree = 56',accuracy_score(y_test, y_pred2))

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred3=classifier.predict(X_test)
cm3=confusion_matrix(y_test,y_pred3)
#print(cm)
print('randomforest =54',accuracy_score(y_test, y_pred3))





