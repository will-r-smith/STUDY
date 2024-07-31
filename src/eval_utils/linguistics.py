import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import spacy

# Helper function to classify words
def classify_words(words):
    pos_tags = pos_tag(words)
    word_categories = {
        'nouns': [],
        'verbs': [],
        'adjectives': [],
        'adverbs': [],
        'function_words': []
    }

    function_words = set(stopwords.words('english'))
    
    for word, tag in pos_tags:
        if word.lower() in function_words:
            word_categories['function_words'].append(word)
        elif tag.startswith('NN'):
            word_categories['nouns'].append(word)
        elif tag.startswith('VB'):
            word_categories['verbs'].append(word)
        elif tag.startswith('JJ'):
            word_categories['adjectives'].append(word)
        elif tag.startswith('RB'):
            word_categories['adverbs'].append(word)
    
    return word_categories