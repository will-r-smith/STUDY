import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import spacy

# Helper function to classify words
def classify_words(words):
    pos_tags = pos_tag(words)
    function_words = set(stopwords.words('english'))
    
    word_frequencies = {
        'common_nouns': 0,
        'proper_nouns': 0,
        'verbs': 0,
        'modal_verbs': 0,
        'adjectives': 0,
        'adverbs': 0,
        'determiners': 0,
        'conjunctions': 0,
        'prepositions': 0,
        'pronouns': 0,
        'interjections': 0,
        'numerals': 0,
        'function_words': 0
    }
    
    for word, tag in pos_tags:
        if word.lower() in function_words:
            word_frequencies['function_words'] += 1
        elif tag.startswith('NNP'):
            word_frequencies['proper_nouns'] += 1
        elif tag.startswith('NN'):
            word_frequencies['common_nouns'] += 1
        elif tag.startswith('VB'):
            if tag == 'MD':
                word_frequencies['modal_verbs'] += 1
            else:
                word_frequencies['verbs'] += 1
        elif tag.startswith('JJ'):
            word_frequencies['adjectives'] += 1
        elif tag.startswith('RB'):
            word_frequencies['adverbs'] += 1
        elif tag in ('DT', 'PDT', 'WDT'):
            word_frequencies['determiners'] += 1
        elif tag in ('CC', 'IN'):
            word_frequencies['conjunctions'] += 1
        elif tag in ('IN', 'TO'):
            word_frequencies['prepositions'] += 1
        elif tag in ('PRP', 'PRP$', 'WP', 'WP$'):
            word_frequencies['pronouns'] += 1
        elif tag == 'UH':
            word_frequencies['interjections'] += 1
        elif tag == 'CD':
            word_frequencies['numerals'] += 1
    
    return word_frequencies