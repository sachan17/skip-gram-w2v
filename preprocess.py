import numpy as np
from copy import copy
# from nltk.corpus import reuters
import random
import pickle

f = open('reuters_train.pkl', 'rb')
training_words = pickle.load(f)
f.close()

print('Original Data size:', len(training_words))

g_file = open('garbage.txt', 'r')
garbage = []
for l in g_file.readlines():
	garbage.append(l.strip())
g_file.close()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

processed_words = []
for w in training_words:
	if is_number(w) or w in garbage:
		continue
	processed_words.append(w)

print('Reduced Data size:', len(processed_words))

f = open('reuters_processed_train.pkl', 'wb')
pickle.dump(processed_words, f)
f.close()
