#importing Librarys
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
#importing spacy nlp and loading english 
nlp = spacy.load("en_core_web_sm")
#stopwords
stopwords = list(STOP_WORDS)
print(stopwords)
#importing Librarys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#importing datasets
data_yelp = pd.read_csv('yelp_labelled.txt', sep='\t', header = None)
#renameing columns
columns_name = ['Review', 'Sentiment']
data_yelp.columns = columns_name
#importing datasetsand renaming columns
data_amazon = pd.read_csv('amazon_cells_labelled.txt', sep = '\t', header = None)
data_amazon.columns = columns_name
#importing datasets
data_imdb = pd.read_csv('imdb_labelled.txt', sep = '\t', header = None)
data_imdb.columns = columns_name
#appending dataset into one
data = data_yelp.append([data_amazon, data_imdb], ignore_index=True)
#checking sentiment counts
data['Sentiment'].value_counts()
#checking is there null value
data.isnull().sum()

import string
punct = string.punctuation
print(punct)
#define fuction for limitizing and tokenizing
def text_data_cleaning(sentence):
    doc = nlp(sentence)
    
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens
#makig classifier
from sklearn.svm import SVC
tfidf = TfidfVectorizer(tokenizer = text_data_cleaning)
classifier = SVC()
#splitting dataset
X = data['Review']
y = data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#running pipline
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
clf.predict(['Wow, this is amzing lesson'])
clf.predict(['you are awesome'])
