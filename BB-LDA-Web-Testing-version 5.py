#!/usr/bin/env python
# coding: utf-8

# ## Google drive version

# In[ ]:


# Run in python console
get_ipython().system('pip install nltk')
get_ipython().system('pip install pyLDAvis')
get_ipython().system('pip install scipy')
get_ipython().system('pip install boto')

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

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','bad','subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
stop_words.extend(['of','for','and','day','due','put','beg','mfm','fbr','cra','uav','etc','far','sem','iot','tsa','one','dna','fix','new','art','set','maf','ice','yet','jam','age','auc','non','job','big','low','way','cpu','mhz','bit','eye','sub','map','key','lie','lab','rct','end','web','aim','act'])
stop_words.extend(['wsis','pbpm','thus','sc','lrmb','lp','uavs','give','user','long','work','sedc','shapc','rr'])
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# ### Web-Step 1. Import csv files

# In[ ]:


import os, glob
import pandas as pd
import re
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# use Pandas to read 2.csv
import pandas as pd
Web = pd.read_csv('Web.csv')
print(Web)


# ### Step 2. Select the columns ('Author Keywords', 'Article Title', 'Source Title' and renname)

# In[ ]:


Web_re = Web.rename(columns= {'Author Keywords': 'Author_Keywords', 'Document Title': 'Document_Title', 'Article Title': 'Article_Title'}, inplace = False)


# In[ ]:


Web_re['corpus'] = Web_re['Article_Title'] + ' ' + Web_re['Author_Keywords'] + ' ' + Web_re['Abstract']
Web_select = Web_re['corpus']


# In[ ]:


Web_drop_nan = Web_select[Web_select.notna()]
print(Web_drop_nan)


# In[ ]:


Web_list = Web_drop_nan.tolist()
print(Web_list)


# In[ ]:


#Remove Emails
data = []
for text in Web_list:
   
   # put process to lower case
   process = text.lower()
   process = process.replace('"', '')
   process = process.replace("_", " ")
   process = process.replace("causality","cause")
   process = process.replace("causation","cause")
   process = process.replace("caused","cause")
   process = process.replace("explainable","explain")
   process = process.replace("explains","explain")
   process = process.replace("explaind","explain")
   process = process.replace("explanation","explain")
   process = process.replace("interpretable","interpret")
   process = process.replace("inrepretation","interpret")
   process = process.replace("derailment","")
   process = process.replace("data", " ")
   process = process.replace('rule', '')
   process = process.replace("study", " ")
   process = process.replace('studying', '')
   process = process.replace("student", " ")
   process = process.replace('also', '')
   process = process.replace("paper", " ")
   process = process.replace("set", " ")
   process = process.replace('based', '')
   process = process.replace("result", " ")
   process = process.replace("©", " ")
   process = process.replace("elsevier", " ")
   process = process.replace("bv", " ")
   process = re.sub('\S*@\S*\s?', '', process)
   process = re.sub("\'", '', process)
   process = re.sub("[\d-]", '', process)
   process = re.sub('https?:\/\/.*[\r\n]*', '', process)
   process = re.sub('\.|:|;', '', process)
   process = re.sub('\([a-z]\)', '', process)
   process = re.sub('\(|\)', '', process)
   process = re.sub('<', '', process)
   process = re.sub('\s+', ' ', process)
   process = re.sub('url', '', process)
   process = re.sub('\*|\+', '', process)
   process = re.sub('\[|\]|\"|\'|/''|\_','', process)
   
   
   data.append( process )
print(data)


# ### Step6. Tokenize words

# In[ ]:


from nltk.tokenize import word_tokenize
nltk.download('punkt')

def sent_to_words(sentences):
                
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

#Web_drop_Abstract_list_tokenize = list(sent_to_words(data))

#print(Web_drop_Abstract_list_tokenize)

tokenized_models = [word_tokenize(str(i)) for i in data]
tokenized_models


# ### Remove stopword

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv

stopset = set(stopwords.words('english'))

Web_nostops = []
for m in tokenized_models:
    stop_m = []
    for i in m:
        if (str(i).lower() not in stopset) and (len(str(i)) > 3):
            stop_m.append(str(i))
    Web_nostops.append(stop_m)

#print(Web_nostops)
stop_models = [i for i in tokenized_models if str(i).lower() not in stopset]
print('token:'+str(stop_models))


# ### Step7. Creating Bigram and Trigram models

# In[ ]:



def make_bigrams(Web_nostops):
    bigram = gensim.models.Phrases(Web_nostops, min_count=5, threshold=100) # higher threshold fewer phrases.
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in Web_nostops]

def make_trigrams(Web_nostops):
    trigram = gensim.models.Phrases(bigram[Web_nostops], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in Web_nostops]


# In[ ]:


# Form Bigrams
Web_bigrams = make_bigrams(Web_nostops)

Web_bigrams
    


# In[ ]:


#Lemmatization


# In[ ]:


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatized_bigrams = []

for tokens in Web_bigrams:
    temp_lemma = []
    for token in tokens:
        temp_lemma.append( lemmatizer.lemmatize(token) )
    lemmatized_bigrams.append(temp_lemma)

print(lemmatized_bigrams)


# In[ ]:


texts = lemmatized_bigrams
#texts = stemmed_bigrams
print(texts)


# ### Step9. Create the Dictionary and Corpus needed for Topic modeling

# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(texts)

# Create Corpus
#texts = lemmatized_bigrams
#texts = stemmed_bigrams

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
#merge casual/ cause etc....
# View
print(corpus[:6])


# In[ ]:


## Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:10]]


# ### Step 10. Compute the Optimum Number of Topics: LDA-Coherence Score

# ### Build LDA Topic model

# In[ ]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                         id2word=id2word,
                                         num_topics=3, 
                                         random_state=100,
                                         update_every=1,
                                         chunksize=100,
                                         passes=500,
                                         alpha='auto',
                                         per_word_topics=True)


# #Step11. View the topic in LDA model

# In[ ]:


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())


# In[ ]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# ### Step12. Visualize the topics-keywords

# In[ ]:


# Visualize the topics
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# ### Step13. What is the Dominant topic and its percentage contribution in each document

# In[ ]:


def format_topics_sentences(ldamodel=None, corpus=corpus, texts=texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    #sent_topics_df = texts

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
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


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic


# In[ ]:


Web_droplicate = df_dominant_topic.drop_duplicates(subset= ['Document_No'])
print(Web_droplicate)


# In[ ]:


Web_topic_select = Web_droplicate[(Web_droplicate.Dominant_Topic == 1.0)]
print(Web_topic_select)


# In[ ]:


#Remove nan duplicate
Web_nan_droplicate = Web_drop_nan.drop_duplicates()
print(Web_nan_droplicate)


# In[ ]:


counter_causability = 0
counter_causal = 0
counter_counterf = 0
counter_interpret = 0
counter_explain = 0
counter_reasoning = 0
counter_bayesian = 0

for t, d in zip(Web_droplicate[Web_droplicate.Dominant_Topic == 1.0].Text.values, df_dominant_topic.Document_No ):
    if( ('causability' in t)):
        counter_causability = counter_causability + 1

    if( ('causal' in t) ):
        print(d)
        print(t)
        counter_causal = counter_causal + 1

    if( ('counterfactual' in t) ):
        counter_counterf = counter_counterf + 1

    if( ('interpret' in t) ):
        print(d)
        print(t)
        counter_interpret = counter_interpret + 1

    if( ('explain' in t)):
        counter_explain = counter_explain + 1
 
    if( ('reasoning' in t) ):

        counter_reasoning = counter_reasoning + 1

    if( ('bayesian' in t)):
        counter_bayesian = counter_bayesian + 1

    #print(t)


print(counter_causability, 'causability')
print(counter_causal, 'causal')
print(counter_counterf, 'counterfactual')
print(counter_interpret, 'interpret')
print(counter_explain, 'explain')
print(counter_reasoning, 'reasoning')
print(counter_bayesian, 'bayesian')


# In[ ]:





# In[ ]:





# In[ ]:




