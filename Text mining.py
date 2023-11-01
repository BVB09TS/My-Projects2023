#!/usr/bin/env python
# coding: utf-8

#  # Project 1: Manuplation of Text 
#  

# In[4]:


import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[7]:


Doc1 = "La collecte de ces données a plusieurs objectifs. Elle permet de constituer en temps réel une cohorte de patients infectés, qui inclut aujourd’hui plus de 500 000 patients et qui est devenue une base de données de référence dans le monde"
Doc2 = "Cette base de données alimente près de 60 travaux de recherche des Hôpitaux de Paris portant sur les caractéristiques cliniques de la maladie, sur l’impact de certains médicaments ou encore sur l’étude des facteurs aggravants"
Doc3 = "En y injectant de l’intelligence artificielle, l’AP-HP et ses partenaires mènent également des études de prédiction et testent virtuellement des solutions capables de stopper le virus. Par exemple, ils développent des algorithmes d’apprentissage profond (deep learning) utilisant les données de radiographie et de scanner pour créer des outils fiables de prédiction de formes sévères de Covid-19"
Doc4 = "L’exploitation de ces données permet également d’établir des statistiques et des visualisations, et de fournir des informations réutilisables. Elles sont mises à disposition des équipes d’encadrement des unités de soin sur un portail dédié pour les aider à mieux comprendre le virus, ses évolutions et sa diffusion"


# In[8]:


stopwords.words("french")


# In[9]:


len(stopwords.words("french"))


# In[10]:


Doc1 = "La collecte de ces données a plusieurs objectifs. Elle permet de constituer en temps réel une cohorte de patients infectés, qui inclut aujourd’hui plus de 500 000 patients et qui est devenue une base de données de référence dans le monde"
Doc1 = re.sub("\d+|\.|,|Md$|%|:|$|\)|\(" , '', Doc1).lower()
Doc1             


# In[11]:


Doc1 = "La collecte de ces données a plusieurs objectifs. Elle permet de constituer en temps réel une cohorte de patients infectés, qui inclut aujourd’hui plus de 500 000 patients et qui est devenue une base de données de référence dans le monde"
Doc1 = re.sub("\d+|\.|,|Md$|%|:|$|\)|\(" , '1234', Doc1).lower()
Doc1             


# In[12]:


Doc1 = "La collecte de ces données a plusieurs objectifs. Elle permet de constituer en temps réel une cohorte de patients infectés, qui inclut aujourd’hui plus de 500 000 patients et qui est devenue une base de données de référence dans le monde"
Doc2 = "Cette base de données alimente près de 60 travaux de recherche des Hôpitaux de Paris portant sur les caractéristiques cliniques de la maladie, sur l’impact de certains médicaments ou encore sur l’étude des facteurs aggravants"
Doc3 = "En y injectant de l’intelligence artificielle, l’AP-HP et ses partenaires mènent également des études de prédiction et testent virtuellement des solutions capables de stopper le virus. Par exemple, ils développent des algorithmes d’apprentissage profond (deep learning) utilisant les données de radiographie et de scanner pour créer des outils fiables de prédiction de formes sévères de Covid-19"
Doc4 = "L’exploitation de ces données permet également d’établir des statistiques et des visualisations, et de fournir des informations réutilisables. Elles sont mises à disposition des équipes d’encadrement des unités de soin sur un portail dédié pour les aider à mieux comprendre le virus, ses évolutions et sa diffusion"


# In[13]:


def clean_text(text):
    text = re.sub("\d+|\.|,|Md$|%|:|$|\)|\(|\n", '', text).lower()
    return text


# In[14]:


clean_text(Doc1)


# In[15]:


Doc1 = "La collecte de ces données a plusieurs objectifs. Elle permet de constituer en temps réel une cohorte de patients infectés, qui inclut aujourd’hui plus de 500 000 patients et qui est devenue une base de données de référence dans le monde"


# In[59]:


import nltk


# In[16]:


Stopwords = stopwords.words("french")
Stopwords.append('a')
Stopwords.append("”")


# In[17]:


Doclist = word_tokenize(clean_text(Doc1).lower())
DocFil = [w for w in Doclist if not w in Stopwords]
DocFil


# In[18]:


import nltk
nltk.download('wordnet')


# In[19]:


from nltk.stem import WordNetLemmatizer
import collections, itertools, seaborn
import matplotlib.pyplot as plt


# In[20]:


lemmatizer = WordNetLemmatizer()
DocStem = [lemmatizer.lemmatize(w) for w in DocFil]


# In[21]:


CountWord = collections.Counter(DocStem)
DfCountWord = pd. DataFrame(CountWord.most_common(15), columns = ["Mots", "Occurence"])


# In[71]:


fig, ax = plt.subplots(figsize=(8, 8))
#DfCountWord.sort_values(by="Occurence" )

#fig, ax = plt.subplots(figsize=(8, 8))

DfCountWord.sort_values(by='Occurence').plot.barh(x='Mots',
                      y='Occurence',
                      ax=ax,
                      color="red")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
# 
# # Project 2:  Analysis + Visualization
# 
# You have a paragraph of text as follows:
# 
# "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It employs techniques and theories drawn from many fields within the broad areas of mathematics, statistics, information science, and computer science, in particular from the subdomains of machine learning, classification, cluster analysis, data mining, databases, and visualization."
# 
# Perform the following tasks:
# 
# 1. Preprocess the text to remove any non-alphabetic characters and convert the text to lowercase.
# 2. Tokenize the preprocessed text.
# 3. Remove the English stop words from the tokens.
# 4. Lemmatize the remaining words.
# 5. Count the occurrences of each word.
# 6. Create a DataFrame that shows the 10 most common words and their respective occurrences.
# 
# Implement the above steps and display the DataFrame as the final output.

# In[22]:


string =  "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It employs techniques and theories drawn from many fields within the broad areas of mathematics, statistics, information science, and computer science, in particular from the subdomains of machine learning, classification, cluster analysis, data mining, databases, and visualization."


# ### 1. Preprocess the text to remove any non-alphabetic characters and convert the text to lowercase.
# 

# In[23]:


import re
import pandas as pd
from nltk.corpus import stopwords


# In[24]:


def clean_text (text):
    text = re.sub("\d+|\.|,|Md$|%|:|$|\)|\(" , '', text).lower()
    return text


# In[25]:


clean_text(string)


# ### 2. Tokenize the preprocessed text.
# 

# In[26]:


from nltk.tokenize import word_tokenize


# In[27]:


tokens = word_tokenize(string)


# In[28]:


print(word_tokenize(string))


# ###  3. Remove the English stop words from the tokens.

# In[29]:


stop_words = set(stopwords.words("english"))

flitered_tokens = [w for w in tokens if w.lower() not in stop_words]

print (flitered_tokens)


# ### 4. Lemmatize the remaining words.

# In[30]:


from nltk.stem import WordNetLemmatizer


# In[31]:


lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(w)for w in flitered_tokens ]
print (lemmatized_tokens)


# ### 5. Count the occurrences of each word.
# 

# In[32]:


import collections


# In[33]:


word_counter = collections.Counter(lemmatized_tokens)
print(word_counter)


# ### 6. Create a DataFrame that shows the 10 most common words and their respective occurrences.
# 

# In[34]:


df = pd.DataFrame(word_counter.most_common(10), columns = ["words", "Occurrence"])


# In[35]:


print(df)


# ### 7.Make a bar chart

# In[36]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(df['words'], df['Occurrence'], color='skyblue')
plt.title('Top 10 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Occurrence')
plt.xticks(rotation=45)
plt.show()



# ### 8. Generate a word cloud image in Python

# In[37]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(lemmatized_tokens))

# Display the word cloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Project 3:  Analysis + Visualization + TF IDF 
# 

# ### Consider the following corpus containing three documents:
# 
# Document 1: "Machine learning is fun and interesting."
# Document 2: "Machine learning techniques are used in data science."
# Document 3: "Data science is fun and challenging."
# 
# 
# #### Compute the Term Frequency (TF) for each term in each document:
# Term Frequency (TF) is calculated as the number of times a term appears in a document divided by the total number of terms in that document.
# Compute the Inverse Document Frequency (IDF) for each term:
# 
# #### Inverse Document Frequency (IDF) 
# is calculated as the logarithm of the total number of documents divided by the number of documents containing the term, and then taking the inverse of that.
# 
# #### Calculate the TF-IDF score for each term in each document:
# 
# Multiply the TF and IDF scores for each term in each document to get the TF-IDF score.

# In[38]:


Doc1 = "Machine learning is fun and interesting." 
Doc2 = "Machine learning techniques are used in data science." 
Doc3= "Data science is fun and challenging."


# In[39]:


import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import collections
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text (text):
    text = re.sub("\d+|\.|,|Md$|%|:|$|\)|\(" , '', text).lower()
    return text


# In[40]:


tokens = word_tokenize(Doc1,Doc2,Doc3)
print(word_tokenize(Doc1,Doc2,Doc3))
stop_words = set(stopwords.words("english"))



# In[41]:


flited_tokens = [w for w in tokens if w.lower() not in stop_words]
print (flited_tokens)



# In[42]:


lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(w)for w in flited_tokens ]
print (lemmatized_tokens)


# In[43]:


word_counter = collections.Counter(lemmatized_tokens)
print(word_counter)


# In[44]:


df = pd.DataFrame(word_counter.most_common(10), columns = ["words", "Occurrence"])

plt.figure(figsize=(10, 6))
plt.bar(df['words'], df['Occurrence'], color='skyblue')
plt.title('Top 10 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Occurrence')
plt.xticks(rotation=45)
plt.show()


# In[45]:


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(lemmatized_tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Given corpus
corpus = [
    "Machine learning is fun and interesting.",
    "Machine learning techniques are used in data science.",
    "Data science is fun and challenging."
]

# Calculate TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame for TF-IDF scores
rows, cols = tfidf_matrix.nonzero()
data = [(rows[i], feature_names[cols[i]], tfidf_matrix[rows[i], cols[i]]) for i in range(len(rows))]
df = pd.DataFrame(data, columns=['Document', 'Term', 'TF-IDF'])

# Display the DataFrame
print(df)


# In[ ]:




