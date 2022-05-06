#!/usr/bin/env python
# coding: utf-8

# # Scopus-Step 1. Import csv files

# In[54]:


import os, glob
import pandas as pd
import re


#Scopus 
#Please change the path below:
path = "/Users/yu-liangchou/Desktop/BB3Data/Scopus"
all_files = glob.glob(os.path.join(path, "*.csv"))
all_df = []
for f in all_files:
    df = pd.read_csv(f, sep=',')
    df['file'] = f.split('/')[-1]
    all_df.append(df)
Scopus = pd.concat(all_df, ignore_index=True, sort=True)


# # Step 2. Remove the unwated columns (keep the 'Abstract')

# In[4]:


Scopus.shape


# In[5]:


Scopus.columns


# In[8]:


#Rename
Scopus_re = Scopus.rename(columns= {'Author Keywords': 'Author_Keywords'}, inplace = False)
Scopus_re.columns


# In[11]:


Scopus_re.Title
#Abstract/ Author_Keywords/ Title 


# In[12]:


#Keep Abstract
Scopus_drop = Scopus_re.drop(['Access Type', 'Affiliations', 'Art. No.',
       'Author_Keywords', 'Author(s) ID', 'Authors',
       'Authors with affiliations', 'Cited by', 'DOI', 'Document Type', 'EID',
       'Index Keywords', 'Issue', 'Link', 'Page count', 'Page end',
       'Page start', 'Publication Stage', 'Source', 'Source title', 'Title',
       'Volume', 'Year', 'file'], axis=1)


# # Step 3. Import Packages

# In[13]:


# Run in python console
import nltk; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# # Step 4. Prepare Stopwords

# In[15]:


#Split the files into 'Abstract', 'Author_Keywords', 'Document_Title'
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[16]:


Scopus_drop_Abstract_nan = Scopus_drop[Scopus_drop['Abstract'].notna()]
print(Scopus_drop_Abstract_nan)


# # Step5. Transfer the columns into a list

# In[17]:


Scopus_drop_Abstract_list = Scopus_drop_Abstract_nan.values.tolist()
print(Scopus_drop_Abstract_list)


# In[18]:


for sent in Scopus_drop_Abstract_list:
    print(sent)


# In[19]:


#Remove Emails
data = []
for sent in Scopus_drop_Abstract_list:
   for s in sent:
       process = re.sub('\S*@\S*\s?', '', s)
       process = re.sub('\s+', ' ', process)
       process = re.sub("\'", ' ', process)
       # put process to lower case
       process = process.lower()
       data.append( process )


# # Step6. Tokenize words and Clean-up text

# In[21]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
Scopus_drop_Abstract_list_tokenize = list(sent_to_words(data))

print(Scopus_drop_Abstract_list_tokenize)


# # Step7. Creating Bigram and Trigram Models

# In[22]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(Scopus_drop_Abstract_list_tokenize, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[Scopus_drop_Abstract_list_tokenize], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[Scopus_drop_Abstract_list_tokenize[1]]])


# # Step8. Remove Stopwords, Make Bigrams and Lemmatize

# In[29]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[30]:


# Remove Stop Words
Scopus_drop_Abstract_nostops = remove_stopwords(Scopus_drop_Abstract_list_tokenize)

# Form Bigrams
Scopus_drop_Abstract_bigrams = make_bigrams(Scopus_drop_Abstract_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
#python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
Scopus_drop_Abstract_lemmatized = lemmatization(Scopus_drop_Abstract_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(Scopus_drop_Abstract_lemmatized[5])


# # Step9. Create the Dictionary and Corpus needed for Topic Modeling

# In[32]:


# Create Dictionary
id2word = corpora.Dictionary(Scopus_drop_Abstract_lemmatized)

# Create Corpus
texts = Scopus_drop_Abstract_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:6])


# In[33]:


id2word[10]


# In[34]:


## Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:10]]


# # Step10. Building the Topic Model

# In[35]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# # Step11. View the topics in LDA model

# In[36]:


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[37]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=Scopus_drop_Abstract_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# # Step12. Visualize the topics-keywords

# In[53]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# # Step13. Building LDA Mallet Model

# In[39]:


# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = '/Users/yu-liangchou/Desktop/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)


# In[40]:


# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# # Step14. How to find the optimal number of topics for LDA?

# In[45]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[46]:


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=Scopus_drop_Abstract_lemmatized, start=2, limit=40, step=6)


# In[47]:


# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[48]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[49]:


# Select the model and print the topics
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# # Step15. Finding the dominant topic in each sentence

# In[50]:


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


# # Step16. Find the most representative document for each topic

# In[51]:


# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()


# # Step17. Topic distribution across documents

# In[52]:


# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

