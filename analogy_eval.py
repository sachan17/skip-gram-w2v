# @Author: sachan
# @Date:   2019-03-07T22:20:50+05:30
# @Last modified by:   sachan
# @Last modified time: 2019-03-11T13:50:05+05:30

from sklearn.neighbors import NearestNeighbors
import sys
import gzip
import numpy
import math
from collections import Counter
from operator import itemgetter

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):
  word_vecs = {}
  if filename.endswith('.gz'): file_object = gzip.open(filename, 'r')
  else: file_object = open(filename, 'r')

  for line_num, line in enumerate(file_object):
    line = line.strip().lower()
    word = line.split()[0]
    word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vec_val in enumerate(line.split()[1:]):
      word_vecs[word][index] = float(vec_val)
    ''' normalize weight vector '''
    word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return word_vecs

if len(sys.argv) < 2:
    print("Error: No Embedding file")
    exit()


def main(vfile):
    word_vec_file = vfile
    word_vectors = read_word_vectors(vfile)

    indexed_words = []
    vector_matrix = []

    for word in word_vectors.keys():
        indexed_words.append(word)
        vector_matrix.append(word_vectors[word])

    vector_matrix = numpy.array(vector_matrix)
    print('Vectors Loaded')

    nnbr = NearestNeighbors(n_neighbors=10)
    nnbr.fit(vector_matrix)

    infile = open("questions-words.txt", 'r')
    data = infile.readlines()

    sections = {}
    sec = None
    for d in data:
        if d[0] == ':':
            if sec:
                print(sec, sections[sec])
            sec = d[2:]
            sections[sec] = {"not_found":0, "total":0, "correct":0}
            continue
        sections[sec]["total"] += 1
        l = d.strip().split(' ')
        if l[0] not in indexed_words or l[1] not in indexed_words or l[2] not in indexed_words or l[3] not in indexed_words:
            sections[sec]["not_found"] += 1
            continue
        vec1 = word_vectors[l[0]]
        vec2 = word_vectors[l[1]]
        vec4 = word_vectors[l[3]]
        vec3_comp = vec1 - vec2 + vec4
        top_3 = [indexed_words[i] for i in nnbr.kneighbors([vec3_comp], 5, return_distance=False)[0]]
        if l[2] in top_3:
            sections[sec]["correct"] += 1
    if sec:
        print(sec, sections[sec])
if __name__ == '__main__':
    if len(sys.argv) < 2:
            print("Error: No embedding file")
    main(sys.argv[1])
