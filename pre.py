import os
import re
import string
from textblob import Word
from SupportVectorClassifier import *
from NaiveBayesClassifier import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('wordnet')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# WE HAVE TWO CLASSES
category=['pos','neg']
# print(os.listdir(path))
courpos=[]
sentiment=[]
stopwords = set(stopwords.words('english'))
for cat in category:
    for i in os.listdir(cat):
        file = os.path.join(cat, i)
        # print(file)
        f = open(file, 'r')
        text = f.read()
        #1- convert to lowerCase
        text = text.lower()
        #2-REMOVE ANY SPEACIAL CHARACTER  AND NUMBERS
        text = re.sub("[^a-zA-Z]", " ", text)

        #3-TOKANIZATION
        word_tokinzed = word_tokenize(text)

        #4-LEMMATIZATION
        lemmatizer = WordNetLemmatizer()
        words=[Word(word).lemmatize() for word in word_tokinzed]


        # 5-Stop Words Removel
        filtered = [word for word in words if word.casefold() not in stopwords]

        # # Remove Punctuation strings
        # no_of_punc_words = [''.join(char for char in word if char not in string.punctuation)
        #                     for word in filtered]

        # #6-Remove the empty string
        # no_of_punc_words = [word for word in filtered if word]
        # print(len(no_of_punc_words))
        # word_counter = Counter(no_of_punc_words)



        #6-JOIN THE TEXT
        no_of_punc_words = " ".join(filtered)

        #LIST OF TEXTS
        courpos.append(no_of_punc_words)
        #LIST OF CLASSES
        sentiment.append(cat)


courposArray=np.array(courpos)
sentimentArray = np.array(sentiment)

# DataFrame text coloum with sentiment coloulm
data=pd.DataFrame(courposArray)
data['sentiment']=sentimentArray
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

# Label Encodeing to sentiment coloum
one_hot_encoded_data = pd.get_dummies(data, columns=['sentiment'])
print(one_hot_encoded_data)

# y=data.iloc[:,-1]
# print((x.shape))
# print((y))


X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=.8,shuffle=True,random_state=100)
print(X_train)

#FEATURE EXTRACTION
vectorizer= TfidfVectorizer(min_df=100,use_idf=True)
vectors_train = vectorizer.fit_transform(X_train).toarray()
vectors_test = vectorizer.transform(X_test).toarray()
# print()

print("vectors_train : ",vectors_train.shape)
print(pd.DataFrame(vectors_train,columns=vectorizer.get_feature_names_out()))
print("vectors_test : ",vectors_train.shape)


acc= SVClassifier().S_V_C(vectors_train,vectors_test,Y_train,Y_test)
acc1= Naive_Bayes().naivebayes(vectors_train,vectors_test,Y_train,Y_test)
plt.show()

