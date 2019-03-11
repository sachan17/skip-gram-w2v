# @Author: sachan
# @Date:   2019-03-10T02:04:53+05:30
# @Last modified by:   sachan
# @Last modified time: 2019-03-10T17:38:28+05:30

import numpy as np
from copy import copy
# from nltk.corpus import reuters
import random
import pickle
import sys

if len(sys.argv) < 2:
    print("Error: No input file")
    exit(0)

f = open(sys.argv[1], 'rb')
training_words = pickle.load(f)
f.close()


VOCABULARY = []

word_count = {}

for i in training_words:
    if i.lower() not in VOCABULARY:
        VOCABULARY.append(i.lower())
        if i.lower() not in word_count.keys():
            word_count[i.lower()] = 0
    word_count[i.lower()] += 1


total_count = 0
for each in word_count.keys():
    total_count += word_count[each]


alpha = 0.75
weights = np.array(list(word_count.values()))
weights = weights**alpha
weights = weights / np.sum(weights)


VOCABULARY_SIZE = len(VOCABULARY)

print('Vocab size:', VOCABULARY_SIZE)

word_2_index = {w: idx for (idx, w) in enumerate(VOCABULARY)}
index_2_word = {idx: w for (idx, w) in enumerate(VOCABULARY)}

VOCABULARY_INDEX = []
for each in VOCABULARY:
    VOCABULARY_INDEX.append(word_2_index[each])

def processed_corpus(sentence):
    text_corpus = []
    index_corpus = []
    # for sentence in sentences:
    for word in sentence:
        if word.lower() in VOCABULARY:
            text_corpus.append(word.lower())
            index_corpus.append(word_2_index[word.lower()])
    return text_corpus, index_corpus, len(text_corpus)

training_text_corpus, training_index_corpus, training_corpus_size = processed_corpus(training_words)


print('Preprocessing Done')

uni_gram_table = copy(VOCABULARY_INDEX)
for ug in VOCABULARY_INDEX:
    uni_gram_table.extend([ug] * int(VOCABULARY_SIZE *  weights[ug]))

uni_gram_table = np.array(uni_gram_table)
np.random.shuffle(uni_gram_table)

# def negative_sample(neg_count):
#     negative_words = random.sample(uni_gram_table, neg_count)
#     return negative_words

counter = 0
uni_size = uni_gram_table.shape[0]
def negative_sample(neg_count):
    global counter
    negative_words = uni_gram_table[counter: counter+neg_count]
    counter = (counter+neg_count)%uni_size
    return negative_words


def generate_batch(skip_window, data_index, batch_size, neg_size):
    words = []
    contexts = []
    negatives = []
    for w_i in range(data_index, min(training_corpus_size, data_index+batch_size), 1):
        for j in range(w_i-skip_window, w_i+skip_window+1):
            if j >= 0 and j < training_corpus_size and j != w_i:
                words.append(training_index_corpus[w_i])
                contexts.append(training_index_corpus[j])
                if neg_size > 0:
                    negatives.append(negative_sample(neg_size))
    return words, contexts, negatives

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

skip_window = 5
data_index = 0
batch_size = 100
negative_sample_size = 10
learning_rate = 0.01

embedding_size = 100

W_target = np.random.randn(VOCABULARY_SIZE, embedding_size)
W_context = np.zeros((VOCABULARY_SIZE, embedding_size))

def train(target, context, negatives, grad_targets, grad_contexts):
    global W_context, W_target
    loss = 0
    target_vec = W_target[target]
    pos = np.dot(target_vec, W_context[context])
    if pos < -10:
        return 0, grad_targets, grad_contexts
    sig_pos = sigmoid(pos)
    loss -= np.log(sig_pos)
    grad_targets[target] += W_context[context] * (sig_pos - 1)
    grad_contexts[context] += target_vec * (sig_pos - 1)

    for i in negatives:
        neg = np.dot(target_vec, W_context[i])
        if neg > 10:
            continue
        sig_neg = sigmoid(neg)
        loss -= np.log(1 - sig_neg)
        grad_targets[target] += W_context[i] * sig_neg
        grad_contexts[i] += target_vec * sig_neg

    return loss, grad_targets, grad_contexts

num_epochs = 10

for epoch in range(num_epochs):
    data_index = 0
    while data_index < training_corpus_size:
        words, contexts, negatives = generate_batch(skip_window, data_index, batch_size, negative_sample_size)
        data_index += batch_size
        batch_loss = 0
        grad_targets = np.zeros(W_target.shape)
        grad_contexts = np.zeros(W_context.shape)
        for i in range(len(words)):
            w = words[i]
            c = contexts[i]
            n = negatives[i]
            l, grad_targets, grad_contexts = train(w,c,n, grad_targets, grad_contexts)
            batch_loss += l

        W_target -= learning_rate * grad_targets
        W_context -= learning_rate * grad_contexts

        if (data_index/batch_size) % 10 == 0:
            print('Loss at epoch {}, step {}: Training Loss:{}'.format(epoch, data_index/batch_size, batch_loss/len(words)))

f = open('Embeddings'+str(epoch)+'.txt', 'w')
for each in VOCABULARY:
    print(each, end=' ', file=f)
    for e in W_target[word_2_index[each]]:
        print(e, end=' ', file=f)
    print(file=f)
f.close()
