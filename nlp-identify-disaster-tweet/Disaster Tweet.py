#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

#Importing the dataset - Training dataset
dataset = pd.read_csv('D:/coding/nlp-getting-started/train.csv')
X = dataset.iloc[:,[0,3]]
y = dataset.iloc[:,-1]

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from urllib.parse import urlparse
from nltk.stem.porter import PorterStemmer

def TextCleaning(df_input):
    corpus = []
    for i in range(0,df_input.shape[0]):
        text = re.sub('[^a-zA-Z]',' ',df_input.iloc[i,-1])          #all characters that is not a letter in tweet text will be put as a space
        text = text.lower()                                         #everything into lower case
        text = re.compile(r'https?://\S+|www\.\S+').sub(r'',text)   #remove url in text
        text = re.compile(r'<.*?>').sub(r'',text)                   #remove html tags
        text = re.compile("["                                       #remove emojis
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE).sub(r'',text)
        text = text.split()                                         #split cleaned text into list of words comprised in the text
        
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
        #for word in text:
        #    if not word in set(all_stopword):
        #        ps.stem(word)
        text = ' '.join(text)
    
        corpus.append(text)
    return corpus

corpus = TextCleaning(X)

# Spell Checking
  #The cleaning and stemming process above result in some word being cropped off, e.g. forgiv instead of forgive
  #So we are doing a spell check here and correct wrong spellings
from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(list_input):
    corrected_text = []
    misspelled_words = spell.unknown(list_input)
    for word in list_input:
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

corpus = correct_spellings(corpus)

#Creating the Bag of Words model/sparse matrix (Transforming X_train into a sparse matrix)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

#Split dataset X_Train into actual Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Train the Naive Bayes Model on Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# %time classifier.fit(X_train, y_train)

#Multinomial Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
%time classifier.fit(X_train, y_train)


#Predict the Test set result
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)), 1))

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score = accuracy_score(y_test,y_pred)
print(accuracy_score)
