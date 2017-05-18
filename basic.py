from gensim import corpora
from gensim.models.ldamodel import LdaModel

from docs import *
from utils import Preprocessor

p = Preprocessor()
# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
raw_docs = [doc.lower() for doc in doc_set]

print "raw: {0}".format(raw_docs)

# 1 Tokenize
# 2 Remove stop words or garbage words
# 3 Stemm out similar words
text = p.pre_process_docs(raw_docs)

print "procesed: {0}".format(text)
# Convert into ID dict
dictionary = corpora.Dictionary(text)
# Convert to Bag of words
corpus = [dictionary.doc2bow(t) for t in text]
# Construct and apply LDA
ldamdl = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=40)

print ldamdl.print_topics(num_topics=2, num_words=1)

