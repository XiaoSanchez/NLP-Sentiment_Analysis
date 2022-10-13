# Twitter Sentiment Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
from wordcloud import WordCloud
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# Load the data
df = pd.read_csv('data.csv')
df
df.info()
df.describe()
df['tweet']
# Drop the 'id' column
df = df.drop(['id'], axis=1)
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
df.hist(bins = 30, figsize = (13,5), color = 'r')
sns.countplot(df['label'], label = "Count") 
# Let's get the length of the messages
df['length'] = df['tweet'].apply(len)
df
df.describe()
# Let's see the shortest message 
df[df['length'] == 11]['tweet'].iloc[0]
# Let's view the message with mean length 
df[df['length'] == 84]['tweet'].iloc[0]
# Plot the histogram of the length column
df['length'].plot(bins=100, kind='hist') 
positive = df[df['label']==0]
positive
negative = df[df['label']==1]
negative
sentences = df['tweet'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
negative_list = negative['tweet'].tolist()
negative_list
negative_sentences_as_one_string = " ".join(negative_list)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))
import string
string.punctuation
Test = '$I love AI & Machine learning!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join
Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed
# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join
import nltk # Natural Language tool kit 
nltk.download('stopwords')

# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')
Test_punc_removed_join = 'I enjoy coding, programming and Artificial intelligence'
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
Test_punc_removed_join_clean # Only important (no so common) words are left
Test_punc_removed_join
mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'
# Remove punctuations
challege = [ char     for char in mini_challenge  if char not in string.punctuation ]
challenge = ''.join(challege)
challenge
challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 
challenge
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

print(vectorizer.get_feature_names())

print(X.toarray())  
mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

# mini_challenge = ['Hello World', 'Hello Hello Hello World world', 'Hello Hello World world world World']

vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean
# Let's test the newly added function
df_clean = df['tweet'].apply(message_cleaning)
print(df_clean[5]) # show the cleaned up version
print(df['tweet'][5]) # show the original version
from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(df['tweet'])

print(tweets_countvectorizer.toarray())  
tweets_countvectorizer.shape
X = pd.DataFrame(tweets_countvectorizer.toarray())

X
y = df['label']
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix
# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))