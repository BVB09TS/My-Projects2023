#!/usr/bin/env python
# coding: utf-8

# #       Text mining Search engine Project 1

# ## 1. **Importing Libraries:**  + **Reading JSON Data:**
# 
# The code begins by importing necessary libraries, including pandas, re (regular expressions), TfidfVectorizer from sklearn, and various modules from the nltk library.
# 
# **Reading JSON Data:** The script reads a JSON file named 'AbstractBook.json' using the pandas library's read_json method and stores it in the variable `Data`.
# 

# In[21]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[22]:


Data = pd.read_json('DataSetes/AbstractBook.json', encoding='utf-8')
stop_words = stopwords.words('english')
Data


# ## 2. **Defining the clean_sentence function:** This function takes a sentence as input and performs the following operations:
# 
#  a. Removes digits, dots, commas, 'Md', '%', colons, dollar signs, parentheses, and newlines from the sentence using regular expressions.
#    
#  b. Converts the sentence to lowercase.
#    
#  c. Tokenizes the sentence into words.
#    
#  d. Filters out the stop words using the NLTK library.
#    
#  e. Joins the remaining words back into a sentence and returns it.

# In[23]:


def clean_sentence(sen):
    sen = re.sub("\d+|\.|,|Md$|%|:|$|\)|\(|\n", '', sen).lower()
    senList = word_tokenize(sen.lower())
    DocFil = [w for w in senList if not w in stop_words]
    senFil = ' '.join(DocFil)  # Ensure a space between words
    return senFil


# ### 3. **Creating a TfidfVectorizer instance:** The script creates a TfidfVectorizer object from the sklearn library, which is used to convert a collection of raw documents to a matrix of TF-IDF features.
# 
# ### 4. **Processing the Abstracts:** The script cleans each abstract in the 'Data' using the clean_sentence function, stores the cleaned abstracts in the list `abstractsList`, and transforms them into vectors using the TfidfVectorizer.
# 
# ### 5. **Converting vectors to DataFrame:** The script converts the vectors to a dense matrix, then to a list, and finally to a DataFrame with the feature names as columns. The DataFrame is transposed and stored in the variable `engine`.
# 
# ### 7. **Naming the Columns:** The script assigns names to the columns of the `engine` DataFrame using a list comprehension, with column names 'Chapitre_1' to 'Chapitre_20'.

# In[24]:


vectorizer = TfidfVectorizer()
abstractsList = [clean_sentence(abs) for abs in list(Data.abstract.values)]
vectors = vectorizer.fit_transform(abstractsList)
feature_names = vectorizer.get_feature_names_out()
#=================================================================================
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
engine = df.transpose()
#=================================================================================
colnames = ['Chapitre_'+str(i+1) for i in range(20)]
engine.columns = colnames
engine


# ### 7. **Executing a Query:** The script defines a variable `query` with the value "education best system in africa". It tokenizes the cleaned version of the query using the `word_tokenize` and `clean_sentence` functions.
# 
# ### 8. **Retrieving Relevant Data:** It retrieves the rows from the `engine` DataFrame corresponding to the tokens in the cleaned query and sums the values in each column, storing the result in the variable `tfidf`.
# 
# ### 9. **Sorting and Displaying Results:** It sorts the values in `tfidf` in descending order and displays the top 5 values.
# 

# In[25]:


query = "education best system in africa"
word_tokenize(clean_sentence(query))
query_data = engine.loc[word_tokenize(clean_sentence(query))]
#=================================================================================
query_data = engine.loc[word_tokenize(clean_sentence(query))]
tfidf = query_data.sum()
tfidf.sort_values(ascending = False).head(5)


# In[26]:


engine = pd.DataFrame(Data)
query = "education best system in africa"
tokens = word_tokenize(clean_sentence(query))


# ### 10. **Defining the book_engine function:** The script defines a function named `book_engine` that takes a query as input. Inside the function, it performs similar operations to those in steps 8 and 9, and returns the top 10 results.
# 
# ### 11. **Accessing the 'abstract' Column of the Data:** The script prints the content of the 'abstract' column from the Data DataFrame at index 1.
# 

# In[27]:


def book_engine(query):
    word_tokenize(clean_sentence(query))
    query_data = engine.loc[word_tokenize(clean_sentence(query))]
    tfidf = query_data.sum()
    return (tifidf.sort_values(ascending = False).head(10))


# In[28]:


Data['abstract'].loc[1]


# In[29]:


import matplotlib.pyplot as plt

# After obtaining the 'tfidf' variable:
top_tfidf = tfidf.sort_values(ascending=False).head(5)

# Creating a bar plot to visualize the top TF-IDF values
plt.figure(figsize=(8, 6))
top_tfidf.plot(kind='bar')
plt.title('Top 5 TF-IDF Values')
plt.xlabel('Words')
plt.ylabel('TF-IDF Scores')
plt.show()


# In[32]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# After obtaining the 'tfidf' variable:
top_words = tfidf.sort_values(ascending=False).head(5)

# Convert the top words to a dictionary for the word cloud
wordcloud_dict = top_words.to_dict()

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




