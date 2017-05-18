from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words

class Preprocessor:
    tokenizer=None
    stemmer=None
    stopper=None
    
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopper = get_stop_words('en')
        self.stemmer = PorterStemmer()

    def pre_process_doc(self, doc=None):
        if not doc:
            raise AttributeError("No input")

        tokens = self.tokenizer.tokenize(doc)
        stopped_tokens = [i for i in tokens if not i in self.stopper]
        stemmed = [self.stemmer.stem(i) for i in stopped_tokens]
        return stemmed
    
    def pre_process_docs(self, docs=[]):
        if len(docs) == 0:
            raise AttributeError("No input")

        return [self.pre_process_doc(d) for d in docs] 
